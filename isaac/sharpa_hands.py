"""Drive two Sharpa Wave hands in Isaac Lab 3 from egohand-live hand tracking.

Run with Isaac Sim's bundled Python, which already ships Isaac Lab, torch, pinocchio and
dex-retargeting (``scripts/setup_isaac.sh`` fetches the Sharpa assets and adds pyzmq +
msgpack under ``third_party/isaac_pydeps``; ``scripts/fetch_simready.py`` fetches the props):

    ~/isaacsim/python.sh isaac/sharpa_hands.py --connect tcp://GPU_HOST:5555

The GPU host runs ``egohand-live run ... --publish``. This script starts
``egohand-live retarget`` as a child process (DexPilot retargeting to Sharpa finger joints,
kept out of the render loop's GIL) unless ``--retarget-connect`` points at one already
running. Finger joints get position targets; each hand base is a free (gravity-less) body
driven toward the tracked wrist with a velocity command every physics step, so it pushes
and grasps objects through proper contacts instead of teleporting through them.

Hand poses arrive in the camera frame; ``T_world_camera`` (``--extrinsics``, else the one
sent by ``run --extrinsics``, else ``--camera-height`` / ``--camera-tilt``) places them over
a desk whose surface is z = 0: +x right, +y away from the operator, +z up (the frame of
hardware/d405_desk_mount/mount_geometry.py).

``--rig franka`` mounts the hands on the flanges of a Franka FR3 Duo (official SimReady
asset, see ``isaac/franka_rig.py``): the tracked wrist pose drives each arm by IK, the fingers
as before.

Keys (viewport focused): R resets the objects. Objects that fall off the desk respawn.
"""

import argparse
import atexit
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PYDEPS = REPO / "third_party" / "isaac_pydeps"
sys.path[:0] = [str(REPO), str(PYDEPS), str(REPO / "isaac")]

from isaaclab.app import add_launcher_args, launch_simulation  # noqa: E402
import render_settings  # noqa: E402  (isaac/)

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--connect", default="tcp://localhost:5555", help="egohand-live hands publisher")
parser.add_argument("--retarget-connect", help="use a running `egohand-live retarget` instead of starting one")
parser.add_argument("--robot", default="sharpa_wave", help="robot hand (see egohand_live.retarget.ROBOT_HANDS)")
parser.add_argument("--rig", default="floating", choices=["floating", "franka"],
                    help="floating: free hands servoed to the wrists; franka: hands on a Franka FR3 Duo (IK)")  # fmt: skip
parser.add_argument("--view", choices=["camera", "overview"],
                    help="viewer: along the real camera (default for floating) or behind the operator (franka)")  # fmt: skip
parser.add_argument("--scene", default="simready", choices=["simready", "cubes", "empty"],
                    help="objects on the desk (simready: ~20 NVIDIA SimReady props)")  # fmt: skip
parser.add_argument("--simready-root", type=Path,
                    help="folder containing nvidia/<asset>/ (default $SIMREADY_ROOT, else third_party/simready)")  # fmt: skip
parser.add_argument("--extrinsics", help="4x4 T_world_camera text file (overrides the one in messages)")
parser.add_argument("--camera-height", type=float, default=0.6, help="fallback camera height above the desk [m]")
parser.add_argument("--camera-tilt", type=float, default=45.0, help="fallback camera pitch below horizontal [deg]")
parser.add_argument("--lowpass", type=float, default=0.5, help="retargeter low-pass alpha (1 = off)")
parser.add_argument("--park-after", type=float, default=1.5,
                    help="seconds a lost hand holds its last pose before it is parked")  # fmt: skip
parser.add_argument("--follow", type=float, default=0.5,
                    help="per-physics-step blend toward the tracked wrist pose (1 = snap)")  # fmt: skip
parser.add_argument("--max-hand-speed", type=float, default=1.2, help="hand base speed limit [m/s]")
parser.add_argument("--grip-friction", type=float, default=1.5, help="friction of the robot hands' colliders")
parser.add_argument("--no-keypoints", action="store_true", help="hide the tracked human joints")
parser.add_argument("--view-distance", type=float, default=0.5,
                    help="viewer looks at the point this far along the real camera's optical axis [m]")  # fmt: skip
parser.add_argument("--screenshot", help="save a viewport screenshot to this .png after --screenshot-after s")
parser.add_argument("--screenshot-after", type=float, default=15.0)
parser.add_argument("--screenshot-every", type=float, default=0.0,
                    help="repeat screenshots this often; use {i} in --screenshot for the index")  # fmt: skip
parser.add_argument("--render-every", type=int, default=4,
                    help="render one frame every N physics steps (physics runs at 120 Hz; 4 = 30 fps)")  # fmt: skip
parser.add_argument("--control-every", type=int, default=2,
                    help="update hand/arm commands every N physics steps (2 = 60 Hz)")  # fmt: skip
parser.add_argument("--max-seconds", type=float, default=0.0, help="quit after this long (0 = until closed)")
parser.add_argument("--debug", action="store_true", help="print commanded vs simulated hand poses")
render_settings.add_args(parser)
add_launcher_args(parser)
# CPU PhysX: for a handful of articulations it beats GPU physics, whose per-step GPU<->Python
# syncs dominate here (measured ~4x faster control loop on an RTX 3080 Ti laptop).
parser.set_defaults(device="cpu")
args = parser.parse_args()
render_settings.configure_launch(args)
if not getattr(args, "visualizer", None) and not getattr(args, "headless", False):
    args.visualizer = "kit"  # Isaac Lab 3 opens no viewport unless a visualizer is requested

import numpy as np  # noqa: E402
import torch  # noqa: E402
import zmq  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

import franka_rig  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
import scenes  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg  # noqa: E402
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from isaaclab_physx.sim.schemas import PhysxRigidBodyCfg  # noqa: E402
from isaaclab_physx.sim.spawners.materials import PhysxRigidBodyMaterialCfg  # noqa: E402

from egohand_live.protocol import TOPIC_ROBOT, decode_robot_hands  # noqa: E402
from egohand_live.retarget import ROBOT_HANDS  # noqa: E402

SIDES = ("left", "right")
PARK_POS = {"left": np.array([-0.3, 0.05, 0.3]), "right": np.array([0.3, 0.05, 0.3])}  # near the operator
# Base frame (Isaac Teleop convention): +x out of the palm, +z toward the fingertips.
# Parked: palm down, fingers pointing away from the operator.
PARK_QUAT = Rotation.from_matrix([[0, -1, 0], [0, 0, 1], [-1, 0, 0]]).as_quat()
# Hand base servo: velocity proportional to the pose error, then acceleration-limited
# starting from the *simulated* velocity. A blocked hand (pressing on the desk or an object)
# therefore builds up only a bounded push (~ hand mass x MAX_LIN_ACC) instead of ramming at
# full speed. Gains satisfy gain * max_speed < max_acc, so the servo cannot overshoot.
LIN_GAIN, MAX_LIN_ACC = 15.0, 30.0  # 1/s, m/s^2
ANG_GAIN, MAX_ANG_SPEED, MAX_ANG_ACC = 15.0, 8.0, 200.0  # 1/s, rad/s, rad/s^2
MIN_BASE_HEIGHT = 0.03  # m above the desk; keeps the wrist from being driven into it
TELEPORT_DIST = 0.6  # m; farther jumps (e.g. first frame) teleport instead of flying


def default_T_world_camera(height: float, tilt_deg: float) -> np.ndarray:
    """Camera at (0, 0, height) looking along +y, pitched ``tilt_deg`` below horizontal."""
    t = np.radians(tilt_deg)
    T = np.eye(4)
    # columns: camera x (right), y (image down), z (optical axis) in world coordinates
    T[:3, :3] = [[1, 0, 0], [0, -np.sin(t), np.cos(t)], [0, -np.cos(t), -np.sin(t)]]
    T[:3, 3] = [0, 0, height]
    return T


def view_from_camera(T: np.ndarray, distance: float):
    """Viewer just behind and above the real camera, looking along its optical axis, so
    whatever the camera sees (the hands) is in view."""
    c, y_down, z = T[:3, 3], T[:3, 1], T[:3, 2]
    return tuple(c - 0.35 * z - 0.1 * y_down), tuple(c + distance * z)


def nlerp(q0: np.ndarray, q1: np.ndarray, a: float) -> np.ndarray:
    if np.dot(q0, q1) < 0:
        q1 = -q1
    q = (1 - a) * q0 + a * q1
    return q / np.linalg.norm(q)


def start_retarget_server() -> tuple[str, subprocess.Popen]:
    """Run `egohand-live retarget` with this interpreter (Isaac Sim's Python has its deps)."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    url = f"tcp://127.0.0.1:{port}"
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(REPO), str(PYDEPS), env.get("PYTHONPATH")) if p)
    # One small optimisation per hand: extra BLAS/OpenMP threads only fight Kit, PhysX and the
    # tracker for cores (seen as 10-20x slower retargeting when the laptop is busy).
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env.setdefault(var, "1")
    cmd = [sys.executable, "-m", "egohand_live.cli", "retarget", "--connect", args.connect,
           "--publish", url, "--robot", args.robot, "--lowpass", str(args.lowpass)]  # fmt: skip

    def _die_with_parent():  # Linux: SIGTERM the child even if Isaac Sim is SIGKILLed
        try:
            import ctypes
            import signal

            ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG
        except Exception:
            pass

    proc = subprocess.Popen(cmd, env=env, preexec_fn=_die_with_parent if sys.platform == "linux" else None)
    atexit.register(proc.terminate)
    return url, proc


class RobotHandFeed:
    """Non-blocking reader for robot-hands messages; keeps the newest state per side in
    world coordinates. Polled from the sim loop (no thread, so no GIL contention)."""

    def __init__(self, url, T_override, T_default):
        self.sock = zmq.Context.instance().socket(zmq.SUB)
        self.sock.setsockopt(zmq.RCVHWM, 8)
        self.sock.setsockopt(zmq.SUBSCRIBE, TOPIC_ROBOT)
        self.sock.connect(url)
        self.T_override, self.T_default = T_override, T_default
        self.T_current = T_override if T_override is not None else T_default
        self.state: dict[str, dict] = {}
        self.n_msgs, self.retarget_ms, self.latency_ms = 0, [], []

    def poll(self):
        latest = None
        while True:
            try:
                latest = self.sock.recv_multipart(zmq.NOBLOCK)[1]
            except zmq.Again:
                break
        if latest is None:
            return
        msg = decode_robot_hands(latest)
        now = time.time()
        T = self.T_override
        if T is None:
            T = msg["T_world_camera"] if msg.get("T_world_camera") is not None else self.T_default
        self.T_current = T
        R, t = T[:3, :3], T[:3, 3]
        for h in msg["hands"]:
            self.state[h["side"]] = {
                "joint_names": h["joint_names"],
                "q": h["joint_positions"],
                "pos": R @ h["base_position"] + t,
                "quat": Rotation.from_matrix(R @ Rotation.from_quat(h["base_orientation"]).as_matrix()).as_quat(),
                "joints": h["openxr26_positions"] @ R.T + t,
                "t": now,
            }
            self.retarget_ms.append(h["retarget_ms"])
        self.n_msgs += 1
        self.latency_ms.append((now - msg["t_capture"]) * 1e3)

    def pop_stats(self):
        out = self.n_msgs, self.retarget_ms, self.latency_ms
        self.n_msgs, self.retarget_ms, self.latency_ms = 0, [], []
        return out


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
            print(f"[egohand] keyboard unavailable ({exc}); R-to-reset disabled")

    def _on_key(self, event, *_):
        if event.type == self._press:
            self.pressed.add(event.input.name)
        return True

    def take(self, key: str) -> bool:
        if key in self.pressed:
            self.pressed.discard(key)
            return True
        return False


def build_hands(spec) -> dict[str, Articulation]:
    grip = PhysxRigidBodyMaterialCfg(static_friction=args.grip_friction, dynamic_friction=args.grip_friction * 0.8,
                                     restitution=0.0, friction_combine_mode="max")  # fmt: skip
    grip.func("/World/Materials/HandGrip", grip)
    hands = {}
    for side in SIDES:
        usd = spec.path("usd", side)
        if not usd.is_file():
            raise FileNotFoundError(f"{usd} missing; run scripts/setup_isaac.sh")
        cfg = ArticulationCfg(
            prim_path=f"/World/{side.capitalize()}Hand",
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(usd),
                rigid_props=PhysxRigidBodyCfg(disable_gravity=True, max_depenetration_velocity=1.0),
            ),
            init_state=ArticulationCfg.InitialStateCfg(pos=tuple(PARK_POS[side]), rot=tuple(PARK_QUAT)),
            # Sharpa ships joint drive gains tuned for Isaac Lab in the USD (stiffness/damping None = use them).
            actuators={"fingers": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=None, damping=None)},
            soft_joint_pos_limit_factor=1.0,
        )
        hands[side] = Articulation(cfg)
        make_floating(f"/World/{side.capitalize()}Hand")
        sim_utils.bind_physics_material(f"/World/{side.capitalize()}Hand", "/World/Materials/HandGrip")
    return hands


def simready_root() -> Path:
    return args.simready_root or Path(os.environ.get("SIMREADY_ROOT", scenes.default_simready_root(REPO)))


def build_franka_rig() -> dict[str, dict]:
    """FR3 Duo (one articulation) with a Sharpa hand fixed to each flange, on a riser."""
    grip = PhysxRigidBodyMaterialCfg(static_friction=args.grip_friction, dynamic_friction=args.grip_friction * 0.8,
                                     restitution=0.0, friction_combine_mode="max")  # fmt: skip
    grip.func("/World/Materials/HandGrip", grip)
    x, y, z = franka_rig.DUO_POS
    riser = sim_utils.CuboidCfg(
        size=(0.3, 0.3, z + 0.75), collision_props=sim_utils.UsdPhysicsCollisionCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.25, 0.28), roughness=0.5),
    )  # fmt: skip
    riser.func("/World/DuoRiser", riser, translation=(x, y - 0.05, (z - 0.75) / 2 - 0.002))
    robot = Articulation(franka_rig.articulation_cfg("/World/Duo", simready_root()))
    for path in franka_rig.attach_hands(sim_utils.get_current_stage(), "/World/Duo").values():
        sim_utils.bind_physics_material(path, "/World/Materials/HandGrip")
    return {side: {"robot": robot, "home": np.radians(franka_rig.HOME_DEG[side])} for side in SIDES}


def init_franka_rig(rig: dict, dt: float):
    """After sim.reset(): start posture, joint/body indices and an IK solver per arm."""
    robot = next(iter(rig.values()))["robot"]
    # Isaac Lab 3 does not apply init_state at sim.reset(): write the home posture.
    robot.write_joint_position_to_sim_index(position=robot.data.default_joint_pos.torch.clone())
    robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(robot.data.default_joint_vel.torch))
    robot.actuators.target_command.set_position_index(value=robot.data.default_joint_pos.torch.clone())
    # The Duo reports a floating-base Jacobian (a row per body, 6 base columns first).
    floating = robot.data.body_link_jacobian_w.torch.shape[1] == robot.num_bodies
    limits = robot.data.joint_pos_limits.torch[0].cpu().numpy()
    for side, r in rig.items():
        r["arm_ids"], _ = robot.find_joints(franka_rig.ARM_JOINTS[side], preserve_order=True)
        r["finger_ids"], _ = robot.find_joints(franka_rig.FINGER_JOINTS[side])
        body_ids, _ = robot.find_bodies(f"{side}_hand_C_MC")
        r["body_id"] = body_ids[0]
        r["jac_id"] = body_ids[0] if floating else body_ids[0] - 1
        r["jac_cols"] = [i + 6 for i in r["arm_ids"]] if floating else list(r["arm_ids"])
        lim = limits[r["arm_ids"]]
        r["ik"] = franka_rig.ArmIK(lim[:, 0], lim[:, 1], q_rest=r["home"])
        r["rest_fingers"] = robot.data.default_joint_pos.torch[:, r["finger_ids"]].clone()
        r["park"] = (franka_rig.PARK_POS[side], franka_rig.PALM_DOWN_FORWARD)


def drive_arm(r: dict, goal_p: np.ndarray, goal_q: np.ndarray, device) -> float:
    """One IK step toward the wrist goal; returns the remaining position error [m]."""
    robot, ids = r["robot"], r["arm_ids"]
    goal_p = np.array([goal_p[0], goal_p[1], max(goal_p[2], MIN_BASE_HEIGHT)])
    pose = robot.data.body_link_pose_w.torch[0, r["body_id"]].double().cpu().numpy()
    J = robot.data.body_link_jacobian_w.torch[0, r["jac_id"]][:, r["jac_cols"]].double().cpu().numpy()
    q = robot.data.joint_pos.torch[0, ids].double().cpu().numpy()
    q_des = r["ik"].step(q, J, pose[:3], pose[3:7], goal_p, goal_q)
    robot.actuators.target_command.set_position_index(
        value=torch.as_tensor(q_des[None], dtype=torch.float32, device=device), joint_ids=ids)
    return float(np.linalg.norm(goal_p - pose[:3]))


def make_floating(prim_path: str):
    """Free the hand base. The Sharpa USDs (Isaac URDF importer layout) carry the
    articulation root on ``root_joint``, a fixed joint to the world; deactivate it and put
    the articulation root on the hand's top prim instead. Must run before ``sim.reset()``."""
    from pxr import PhysxSchema, UsdPhysics

    stage = sim_utils.get_current_stage()
    root = stage.GetPrimAtPath(prim_path)
    for child in root.GetChildren():
        if child.IsA(UsdPhysics.Joint) and not UsdPhysics.Joint(child).GetBody0Rel().GetTargets():
            child.SetActive(False)  # world-anchoring joint (also holds the old articulation root API)
    UsdPhysics.ArticulationRootAPI.Apply(root)
    art = PhysxSchema.PhysxArticulationAPI.Apply(root)
    art.CreateEnabledSelfCollisionsAttr().Set(False)
    art.CreateSolverPositionIterationCountAttr().Set(16)
    art.CreateSolverVelocityIterationCountAttr().Set(1)


def build_objects() -> list[RigidObject]:
    if args.scene == "empty":
        return []
    if args.scene == "simready":
        root = simready_root()
        missing = scenes.missing_props(root)
        if not missing:
            return scenes.build_simready(sim_utils, RigidObject, RigidObjectCfg, root)
        print(f"[egohand] {len(missing)} SimReady props missing under {root} (e.g. {missing[0]}); "
              "run scripts/fetch_simready.py or pass --simready-root. Using --scene cubes.")  # fmt: skip
    return scenes.build_cubes(sim_utils, RigidObject, RigidObjectCfg)


def _clamp_norm(v: np.ndarray, limit: float) -> np.ndarray:
    n = np.linalg.norm(v)
    return v * (limit / n) if n > limit else v


def drive_hand(hand: Articulation, goal_p: np.ndarray, goal_q: np.ndarray, dt: float):
    """Velocity command for the free hand base: proportional servo toward the goal pose,
    acceleration-limited from the simulated velocity (bounded contact forces)."""
    data = hand.data
    p = data.root_link_pos_w.torch[0].double().cpu().numpy()
    q = data.root_link_quat_w.torch[0].double().cpu().numpy()
    dev = data.root_link_pos_w.torch.device
    goal_p = np.array([goal_p[0], goal_p[1], max(goal_p[2], MIN_BASE_HEIGHT)])
    if np.linalg.norm(goal_p - p) > TELEPORT_DIST:
        hand.write_root_pose_to_sim_index(
            root_pose=torch.as_tensor(np.concatenate([goal_p, goal_q])[None], dtype=torch.float32, device=dev))
        hand.write_root_velocity_to_sim_index(root_velocity=torch.zeros(1, 6, device=dev))
        return
    com = data.root_com_pos_w.torch[0].double().cpu().numpy()
    vel = data.root_com_vel_w.torch[0].double().cpu().numpy()  # [v_com, w] in world
    w_cur = vel[3:]
    v_cur = vel[:3] - np.cross(w_cur, com - p)  # velocity of the base link origin
    v_des = _clamp_norm(LIN_GAIN * (goal_p - p), args.max_hand_speed)
    w_des = _clamp_norm(ANG_GAIN * (Rotation.from_quat(goal_q) * Rotation.from_quat(q).inv()).as_rotvec(),
                        MAX_ANG_SPEED)  # fmt: skip
    v = v_cur + _clamp_norm(v_des - v_cur, MAX_LIN_ACC * dt)
    w = w_cur + _clamp_norm(w_des - w_cur, MAX_ANG_ACC * dt)
    v_com = v + np.cross(w, com - p)  # the API sets the centre-of-mass velocity
    hand.write_root_velocity_to_sim_index(
        root_velocity=torch.as_tensor(np.concatenate([v_com, w])[None], dtype=torch.float32, device=dev))


def reset_objects(objects: list[RigidObject], fallen_only=False) -> int:
    n = 0
    for obj in objects:
        if fallen_only and obj.data.root_pos_w.torch[0, 2].item() > -0.15:
            continue
        obj.write_root_pose_to_sim_index(root_pose=obj.data.default_root_pose.torch.clone())
        obj.write_root_velocity_to_sim_index(root_velocity=torch.zeros_like(obj.data.default_root_vel.torch))
        n += 1
    return n


def save_screenshot(path: str):
    try:
        from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport

        capture_viewport_to_file(get_active_viewport(), str(Path(path).resolve()))
        print(f"[egohand] screenshot -> {path}")
    except Exception as exc:  # no viewport (headless / non-Kit visualizer)
        print(f"[egohand] screenshot failed: {exc}")


def main():
    T_override = np.loadtxt(args.extrinsics).reshape(4, 4) if args.extrinsics else None
    T_default = default_T_world_camera(args.camera_height, args.camera_tilt)
    spec = ROBOT_HANDS[args.robot]
    retarget_url, retarget_proc = (args.retarget_connect, None) if args.retarget_connect else start_retarget_server()
    sim_cfg = sim_utils.SimulationCfg(dt=1 / 120, render_interval=args.render_every, device=args.device)
    with launch_simulation(sim_cfg, args):
        render_settings.apply(args.render)
        sim = SimulationContext(sim_cfg)
        T_view = T_override if T_override is not None else T_default
        view = args.view or ("overview" if args.rig == "franka" else "camera")
        if view == "camera":
            sim.set_camera_view(*view_from_camera(T_view, args.view_distance))
        else:  # behind and above the operator, looking at the middle of the desk
            sim.set_camera_view((0.0, -0.80, 0.95), (0.0, 0.34, 0.04))
        scenes.build_desk(sim_utils)
        objects = build_objects()
        rig = build_franka_rig() if args.rig == "franka" else None
        hands = {s: r["robot"] for s, r in rig.items()} if rig else build_hands(spec)
        markers = None
        if not args.no_keypoints:
            markers = VisualizationMarkers(VisualizationMarkersCfg(
                prim_path="/Visuals/HumanJoints",
                markers={"joint": sim_utils.SphereCfg(
                    radius=0.004, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.8, 0.1)))},
            ))  # fmt: skip
        sim.reset()
        keys = KeyWatcher()
        if rig:
            init_franka_rig(rig, sim.get_physics_dt())

        feed = RobotHandFeed(retarget_url, T_override, T_default)
        joint_ids: dict[str, list[int]] = {}
        rest_q = {side: hand.data.default_joint_pos.torch.clone() for side, hand in hands.items()}
        print(f"[egohand] {args.rig} rig, {args.scene} scene with {len(objects)} objects | hands from {args.connect} -> "
              f"retarget {retarget_url} -> sim | press R in the viewport to reset objects")  # fmt: skip

        dt = sim.get_physics_dt()
        control_dt = dt * args.control_every
        goal = {s: (PARK_POS[s].copy(), PARK_QUAT.copy()) for s in SIDES}
        if rig:
            goal = {s: (r["park"][0].copy(), r["park"][1].copy()) for s, r in rig.items()}
        ik_err = dict.fromkeys(SIDES, 0.0)
        next_wall, last_log, step = time.perf_counter(), time.time(), 0
        last_log_step, prof = 0, np.zeros(3)  # seconds in: control, sim.step, asset updates
        t_start, shot_pending, n_shots = time.time(), bool(args.screenshot), 0
        while sim.is_running():
            now = time.time()
            if shot_pending and now - t_start > args.screenshot_after:
                save_screenshot(args.screenshot.format(i=n_shots))
                n_shots += 1
                if args.screenshot_every > 0:
                    args.screenshot_after += args.screenshot_every
                else:
                    shot_pending = False
            if args.max_seconds and now - t_start > args.max_seconds:
                break
            if retarget_proc is not None and retarget_proc.poll() is not None:
                raise SystemExit(f"retarget process exited with code {retarget_proc.returncode}")
            if keys.take("R"):
                print(f"[egohand] reset {reset_objects(objects)} objects")
            elif step % 60 == 0 and objects:
                n = reset_objects(objects, fallen_only=True)
                if n:
                    print(f"[egohand] respawned {n} fallen object(s)")

            prof_t = time.perf_counter()
            if step % args.control_every == 0:  # control at physics rate / control_every
                feed.poll()
                if not np.allclose(feed.T_current, T_view):  # extrinsics arrived with the messages
                    T_view = feed.T_current
                    if view == "camera":
                        sim.set_camera_view(*view_from_camera(T_view, args.view_distance))
                keypoints = []
                for side, hand in hands.items():
                    st = feed.state.get(side)
                    if st is not None and now - st["t"] < args.park_after:
                        if side not in joint_ids:
                            joint_ids[side], _ = hand.find_joints(st["joint_names"], preserve_order=True)
                        target_p, target_q = st["pos"], st["quat"]
                        q_cmd = torch.as_tensor(st["q"], dtype=torch.float32, device=sim.device)[None]
                        ids = joint_ids[side]
                        keypoints.append(st["joints"])
                    elif rig:
                        (target_p, target_q), q_cmd, ids = rig[side]["park"], rig[side]["rest_fingers"], rig[side]["finger_ids"]
                    else:
                        target_p, target_q, q_cmd, ids = PARK_POS[side], PARK_QUAT, rest_q[side], None
                    p, q = goal[side]
                    goal[side] = (p + args.follow * (target_p - p), nlerp(q, target_q, args.follow))
                    if rig:
                        ik_err[side] = drive_arm(rig[side], *goal[side], sim.device)
                        hand.actuators.target_command.set_position_index(value=q_cmd, joint_ids=ids)
                    else:
                        drive_hand(hand, *goal[side], control_dt)
                        hand.actuators.target_command.set_position_index(value=q_cmd, joint_ids=ids)
                    hand.write_data_to_sim()
                if markers is not None:
                    markers.set_visibility(bool(keypoints))
                    if keypoints:
                        markers.visualize(translations=torch.as_tensor(np.concatenate(keypoints), dtype=torch.float32))
                for obj in objects:
                    obj.write_data_to_sim()
            t1 = time.perf_counter()
            sim.step(render=step % args.render_every == 0)  # sim.step() would render every physics step
            t2 = time.perf_counter()
            step += 1
            for asset in dict.fromkeys((*hands.values(), *objects)):  # the Duo serves both sides
                asset.update(dt)
            t3 = time.perf_counter()
            prof += np.array([t1 - prof_t, t2 - t1, t3 - t2])

            # Pace physics to wall-clock time so the dynamics look right.
            next_wall += dt
            slack = next_wall - time.perf_counter()
            if slack > 0:
                time.sleep(slack)
            elif slack < -0.1:  # far behind (a stall): drop the debt instead of racing to catch up
                next_wall = time.perf_counter()
            if now - last_log > 5.0:
                elapsed, last_log = now - last_log, now
                n_steps = max(step - last_log_step, 1)
                rtf, last_log_step = n_steps * dt / elapsed, step
                ms = prof / n_steps * 1e3
                prof[:] = 0
                n, rt_ms, lat = feed.pop_stats()
                tracked = [s for s in SIDES if s in feed.state and now - feed.state[s]["t"] < args.park_after]
                print(f"[egohand] {n / elapsed:5.1f} msg/s | sim {rtf:4.2f}x real time | tracked: {', '.join(tracked) or 'none'}"
                      + (f" | retarget {np.median(rt_ms):.1f} ms/hand" if rt_ms else "")
                      + (f" | capture->sim {np.median(lat):.0f} ms" if lat else "")
                      + (f" | per step: control {ms[0]:.1f} ms, physics+render {ms[1]:.1f} ms, update {ms[2]:.1f} ms"
                         if args.debug else ""))  # fmt: skip
                if args.debug and rig:
                    for side, r in rig.items():
                        wrist = r["robot"].data.body_link_pos_w.torch[0, r["body_id"]].cpu().numpy()
                        print(f"  {side}: goal {np.round(goal[side][0], 3)} | wrist {np.round(wrist, 3)} | "
                              f"IK error {ik_err[side] * 1000:.0f} mm")
                elif args.debug:
                    for side, hand in hands.items():
                        sim_p = hand.data.root_link_pos_w.torch[0].cpu().numpy()
                        jp = hand.data.joint_pos.torch[0].cpu().numpy()
                        print(f"  {side}: goal {np.round(goal[side][0], 3)} | sim root {np.round(sim_p, 3)} | "
                              f"joints nan={bool(np.isnan(jp).any())} max|q|={np.abs(jp).max():.2f}")
    if retarget_proc is not None:
        retarget_proc.terminate()


if __name__ == "__main__":
    main()
