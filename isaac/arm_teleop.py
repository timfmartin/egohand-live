"""Teleoperate Isaac Lab arm + gripper tasks with egohand-live, and record demonstrations.

The camera replaces the headset of Isaac Lab's hand-tracking teleoperation: the gripper
follows your right hand and a thumb-index pinch closes it (see egohand_live/gripper_teleop.py).
With ``--record`` the episodes are saved by Isaac Lab's own recorder (HDF5, successful episodes
only), the format of ``scripts/tools/record_demos.py``: ready for replay_demos.py and
GR00T-Mimic / LeRobot conversion.

    ~/isaacsim/python.sh isaac/arm_teleop.py --connect tcp://localhost:5555 --extrinsics T_world_camera.txt
    ~/isaacsim/python.sh isaac/arm_teleop.py ... --record demos/stack.hdf5 --num-demos 10
    ~/isaacsim/python.sh isaac/arm_teleop.py ... --gripper continuous     (opening follows the pinch)

Your desk maps onto the task's table with you standing behind the robot: the middle of your
work area (the board position of ``calibrate-desk-board``) lands on ``--task-centre``.

Keys (viewport focused): R discards the episode and resets, SPACE pauses / resumes.
"""

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PYDEPS = REPO / "third_party" / "isaac_pydeps"
sys.path[:0] = [str(REPO), str(PYDEPS)]

from isaaclab.app import add_launcher_args, launch_simulation  # noqa: E402
import render_settings  # noqa: E402  (isaac/)

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", default="IsaacContrib-Stack-Cube-Franka-IK-Abs",
                    help="Isaac Lab task with an absolute end-effector pose action + gripper action")  # fmt: skip
parser.add_argument("--connect", default="tcp://localhost:5555", help="egohand-live hands publisher")
parser.add_argument("--extrinsics", help="4x4 T_world_camera text file (overrides the one in messages)")
parser.add_argument("--hand", default="right", choices=["right", "left"])
parser.add_argument("--gripper", default="binary", choices=["binary", "continuous"],
                    help="binary: pinch closes (task's own gripper action); continuous: opening follows the pinch")  # fmt: skip
parser.add_argument("--rotation", default="yaw", choices=["yaw", "full"],
                    help="yaw: gripper points down and turns with the hand; full: follows the hand orientation")  # fmt: skip
parser.add_argument("--task-centre", default="0.5,0.0", help="task x,y where your workspace centre lands [m]")
parser.add_argument("--motion-scale", type=float, default=1.0, help="scale hand motion about the workspace centre")
parser.add_argument("--tcp-offset", type=float, default=0.1034,
                    help="end-effector body to fingertip centre along the approach axis [m] (Franka hand)")  # fmt: skip
parser.add_argument("--max-speed", type=float, default=0.8, help="end-effector target speed limit [m/s]")
parser.add_argument("--hold-after", type=float, default=0.5, help="seconds without the hand before the arm holds")
parser.add_argument("--debug", action="store_true", help="print commanded vs actual gripper pose once a second")
parser.add_argument("--record", help="save successful episodes to this .hdf5")
parser.add_argument("--num-demos", type=int, default=0, help="stop after this many successful demos (0 = no limit)")
parser.add_argument("--num-success-steps", type=int, default=10,
                    help="consecutive steps the success condition must hold to finish a demo")  # fmt: skip
render_settings.add_args(parser)
add_launcher_args(parser)
parser.set_defaults(device="cpu")  # one env: CPU PhysX avoids per-step GPU<->Python syncs
args, hydra_args = parser.parse_known_args()
render_settings.configure_launch(args)
sys.argv = [sys.argv[0], *hydra_args]  # the task config (Hydra) parses only its own overrides
if not getattr(args, "visualizer", None) and not getattr(args, "headless", False):
    args.visualizer = "kit"

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

import isaaclab_tasks  # noqa: E402, F401
from egohand_live.gripper_teleop import PinchGripper, desk_to_task, ee_target, step_towards, transform_hand  # noqa: E402
from egohand_live.protocol import Subscriber  # noqa: E402
from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg, JointPositionActionCfg  # noqa: E402
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg  # noqa: E402
from isaaclab.envs.utils._manual_success import _prepare_success_term  # noqa: E402
from isaaclab.managers import DatasetExportMode  # noqa: E402
from isaaclab_tasks.utils import resolve_task_config  # noqa: E402

try:
    import isaaclab_mimic.envs  # noqa: F401  (registers the Mimic variants of the tasks)
except ImportError:
    pass


class KeyWatcher:
    """Collects key presses from the Kit viewport (no-op without a Kit window)."""

    def __init__(self):
        self.pressed: set[str] = set()
        try:
            import carb
            import omni.appwindow

            self._input = carb.input.acquire_input_interface()
            self._keyboard = omni.appwindow.get_default_app_window().get_keyboard()
            self._press = carb.input.KeyboardEventType.KEY_PRESS
            self._sub = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_key)
        except Exception as exc:
            print(f"[egohand] keyboard unavailable ({exc})")

    def _on_key(self, event, *_):
        if event.type == self._press:
            self.pressed.add(event.input.name)
        return True

    def take(self, key: str) -> bool:
        if key in self.pressed:
            self.pressed.discard(key)
            return True
        return False


def make_env_cfg():
    env_cfg, _ = resolve_task_config(args.task, "")
    env_cfg.sim.device = args.device
    env_cfg.scene.num_envs = 1
    env_cfg.env_name = args.task
    if hasattr(env_cfg, "isaac_teleop"):
        env_cfg.isaac_teleop = None  # we are the input device
    env_cfg.terminations.time_out = None
    env_cfg.observations.policy.concatenate_terms = False
    success = _prepare_success_term(env_cfg)
    grip = getattr(env_cfg.actions, "gripper_action", None)
    binary = isinstance(grip, BinaryJointPositionActionCfg)
    gripper_range = None
    if args.gripper == "continuous":
        if not binary:
            raise SystemExit(f"--gripper continuous needs a binary gripper action to replace; {args.task} has {grip}")
        gripper_range = (dict(grip.close_command_expr), dict(grip.open_command_expr))
        env_cfg.actions.gripper_action = JointPositionActionCfg(
            asset_name=grip.asset_name, joint_names=grip.joint_names, scale=1.0, use_default_offset=False)  # fmt: skip
    if args.record:
        out = Path(args.record).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        env_cfg.recorders = ActionStateRecorderManagerCfg()
        env_cfg.recorders.dataset_export_dir_path = str(out.parent)
        env_cfg.recorders.dataset_filename = out.stem
        env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY
    env_cfg.sim.render_interval = env_cfg.decimation  # one viewport render per env step (task default: 2-3)
    env_cfg.viewer.eye = (-0.75, 0.0, 0.85)  # behind the robot, where the operator stands
    env_cfg.viewer.lookat = (0.5, 0.0, 0.05)
    return env_cfg, success, gripper_range


def joint_targets(names: list[str], exprs: dict[str, float]) -> np.ndarray:
    import re

    return np.array([next(v for k, v in exprs.items() if re.fullmatch(k, n)) for n in names], float)


def main():
    env_cfg, success_term, gripper_range = make_env_cfg()
    with launch_simulation(env_cfg, args):
        render_settings.apply(args.render)
        env = gym.make(args.task, cfg=env_cfg).unwrapped
        env.reset()
        am = env.action_manager
        names, dims = list(am.active_terms), list(am.action_term_dim)
        if names[:2] != ["arm_action", "gripper_action"] or dims[0] != 7:
            raise SystemExit(f"expected [arm_action (7: pose), gripper_action] actions, got {list(zip(names, dims))}")
        grip_term = am.get_term("gripper_action")
        robot = env.scene["robot"]
        ee_body = robot.find_bodies(am.get_term("arm_action").cfg.body_name)[0][0]
        if gripper_range is not None:
            jn = list(grip_term._joint_names)
            closed, opened = joint_targets(jn, gripper_range[0]), joint_targets(jn, gripper_range[1])
        origin = env.scene.env_origins[0].cpu().numpy()
        cx, cy = (float(v) for v in args.task_centre.split(","))
        T_task_desk = desk_to_task((cx, cy, 0.0), scale=args.motion_scale)
        T_cam = np.loadtxt(args.extrinsics).reshape(4, 4) if args.extrinsics else None
        sub = Subscriber(args.connect)
        pinch = PinchGripper(args.gripper)
        keys = KeyWatcher()
        step_dt = env.step_dt
        target_pos = target_quat = grip_cmd = None
        last_seen, running, success_steps, steps = 0.0, True, 0, 0
        waiting_said, next_wall = False, time.perf_counter()
        demos = 0
        print(f"[egohand] {args.task}: env step {step_dt * 1000:.0f} ms | {args.hand} hand drives the gripper, "
              f"pinch to close | R reset, SPACE pause" + (f" | recording to {args.record}" if args.record else ""))  # fmt: skip
        with torch.inference_mode():
            while env.sim.is_running():
                if keys.take("R"):
                    print("[egohand] reset (episode discarded)")
                    env.recorder_manager.reset()
                    env.reset()
                    success_steps, target_pos = 0, None
                if keys.take("SPACE"):
                    running = not running
                    print(f"[egohand] {'running' if running else 'paused'}")
                msg, _ = sub.latest(0)
                if msg is not None:
                    hand = next((h for h in msg["hands"] if h["side"] == args.hand), None)
                    T = T_cam if T_cam is not None else msg.get("T_world_camera")
                    if hand is not None and T is not None:
                        T_task_cam = T_task_desk @ np.asarray(T, float)
                        p, q = transform_hand(T_task_cam, hand["openxr26_positions"], hand["openxr26_orientations"])
                        pos, quat, width = ee_target(p, q, args.rotation, args.tcp_offset)
                        pos_w = pos + origin
                        # into the robot base frame (the IK action's frame)
                        root_p = robot.data.root_pos_w.torch[0].cpu().numpy()
                        root_q = robot.data.root_quat_w.torch[0].cpu().numpy()  # xyzw (Isaac Lab 3)
                        R_root = Rotation.from_quat(root_q)
                        pos_b = R_root.inv().apply(pos_w - root_p)
                        quat_b = (R_root.inv() * Rotation.from_quat(quat)).as_quat()
                        if target_pos is None:  # start from the current gripper pose, then follow
                            ee = robot.data.body_pos_w.torch[0, ee_body]
                            target_pos = R_root.inv().apply(ee.cpu().numpy() - root_p)
                        target_pos = step_towards(target_pos, pos_b, args.max_speed * step_dt)
                        target_quat = quat_b
                        grip_cmd = pinch(width)
                        last_seen = time.monotonic()
                if target_pos is None or not running:
                    if target_pos is None and steps == 0 and not waiting_said:
                        print(f"[egohand] waiting for the {args.hand} hand on {args.connect} ...")
                        waiting_said = True
                    env.sim.render()
                    time.sleep(1 / 30)
                    next_wall = time.perf_counter()
                    continue
                if gripper_range is not None:
                    g = closed + grip_cmd * (opened - closed)
                else:
                    g = np.array([grip_cmd])
                action = torch.tensor(np.concatenate([target_pos, target_quat, g]), dtype=torch.float32,
                                      device=env.device).unsqueeze(0)  # fmt: skip
                env.step(action)
                steps += 1
                if args.debug and steps % max(int(1 / step_dt), 1) == 0:
                    rp = robot.data.root_pos_w.torch[0].cpu().numpy()
                    ee_b = Rotation.from_quat(robot.data.root_quat_w.torch[0].cpu().numpy()).inv().apply(
                        robot.data.body_pos_w.torch[0, ee_body].cpu().numpy() - rp)
                    print(f"[egohand] target {np.round(target_pos, 3)} actual {np.round(ee_b, 3)} "
                          f"err {np.linalg.norm(target_pos - ee_b) * 1000:.0f} mm | gripper {np.round(g, 3)}")
                if success_term is not None and bool(success_term.func(env, **success_term.params)[0]):
                    success_steps += 1
                    if success_steps >= args.num_success_steps:
                        if args.record:
                            env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
                            env.recorder_manager.set_success_to_episodes(
                                [0], torch.tensor([[True]], dtype=torch.bool, device=env.device))  # fmt: skip
                            env.recorder_manager.export_episodes([0])
                            demos = env.recorder_manager.exported_successful_episode_count
                        print(f"[egohand] success! demos recorded: {demos}" if args.record else "[egohand] success!")
                        env.recorder_manager.reset()
                        env.reset()
                        success_steps, target_pos = 0, None
                        if args.num_demos and demos >= args.num_demos:
                            break
                else:
                    success_steps = 0
                if time.monotonic() - last_seen > args.hold_after and steps % 60 == 0:
                    print("[egohand] no hand: holding")
                next_wall += step_dt  # real time; keep the debt of a slow step unless far behind
                slack = next_wall - time.perf_counter()
                if slack > 0:
                    time.sleep(slack)
                elif slack < -0.25:
                    next_wall = time.perf_counter()
        sub.close()
        env.close()


if __name__ == "__main__":
    main()
