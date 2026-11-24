"""Retarget OpenXR-26 hand poses to robot hands with dex-retargeting (DexPilot).

The preprocessing follows Isaac Teleop's ``DexHandRetargeter`` (NVIDIA IsaacCapture,
``src/python/isaaccapture/retargeters/dex_hand_retargeter.py``, Apache-2.0) so the
upstream Sharpa Wave DexPilot configs behave as they do with headset hand tracking:

1. take 21 of the 26 OpenXR joints (wrist + thumb chain + proximal..tip per finger),
2. express them in the OpenXR wrist frame,
3. rotate into the robot base frame with a fixed ``openxr_to_base`` matrix,
4. run the dex-retargeting optimizer.

Needs ``numpy``, ``scipy``, ``pyyaml``, ``torch`` and ``dex-retargeting`` (all bundled
with Isaac Sim 6 / Isaac Lab 3). ``torch`` is only imported to re-enable autograd, which
the DexPilot optimizer needs even if the caller runs under ``torch.inference_mode``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parents[1]

# OpenXR XR_EXT_hand_tracking indices -> the 21-joint (MANO-ordered) layout DexPilot uses.
# 0 palm, 1 wrist, 2-5 thumb metacarpal..tip, 6-10 index metacarpal..tip, 11-15 middle,
# 16-20 ring, 21-25 little. Non-thumb metacarpals and the palm are skipped.
OPENXR_TO_DEX21 = [1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 20, 22, 23, 24, 25]
OPENXR_WRIST = 1


@dataclass(frozen=True)
class RobotHandSpec:
    """Files and frame convention for one robot hand model (``{side}`` is filled in)."""

    name: str
    config: str  # dex-retargeting YAML
    urdf: str  # kinematics for the optimizer
    usd: str  # simulation asset
    # Rotation from the OpenXR wrist frame to the robot base (wrist) link frame, applied as
    # ``p_base = openxr_to_base.T @ p_openxr`` (Isaac Teleop's handtracking_to_baselink).
    openxr_to_base: tuple = (1, 0, 0, 0, 1, 0, 0, 0, 1)

    def path(self, field: str, side: str) -> Path:
        p = Path(getattr(self, field).format(side=side))
        return p if p.is_absolute() else REPO_ROOT / p


_SHARPA_DIR = "third_party/sharpa-urdf-usd-xml/wave_01/{side}_sharpa_wave"
ROBOT_HANDS = {
    "sharpa_wave": RobotHandSpec(
        name="sharpa_wave",
        config="configs/retarget/sharpa_wave_{side}_dexpilot.yml",
        urdf=f"{_SHARPA_DIR}/{{side}}_sharpa_wave.urdf",
        usd=f"{_SHARPA_DIR}/{{side}}_sharpa_wave/{{side}}_sharpa_wave.usd",
        # isaaccapture_examples.teleop_ros2.constants.DEX_HANDTRACKING_TO_BASELINK_FRAME_TRANSFORM
        openxr_to_base=(0, -1, 0, -1, 0, 0, 0, 0, -1),
    ),
}


class HandRetargeter:
    """One robot hand. Call with OpenXR-26 positions (any frame) and XYZW orientations.

    ``low_pass_alpha`` overrides the config's first-order filter. The upstream value (0.2)
    assumes ~90 Hz headset tracking; egohand-live runs at ~25 Hz and already smooths, so
    0.5 gives a similar time constant (~45 ms).
    """

    def __init__(self, robot: str | RobotHandSpec, side: str, low_pass_alpha: float | None = 0.5,
                 scaling_factor: float | None = None):  # fmt: skip
        from dex_retargeting.retargeting_config import RetargetingConfig

        self.spec = ROBOT_HANDS[robot] if isinstance(robot, str) else robot
        self.side = side
        cfg = yaml.safe_load(self.spec.path("config", side).read_text())["retargeting"]
        urdf = self.spec.path("urdf", side)
        if not urdf.is_file():
            raise FileNotFoundError(f"{urdf} missing; run scripts/setup_isaac.sh to fetch robot assets")
        cfg["urdf_path"] = str(urdf)
        if low_pass_alpha is not None:
            cfg["low_pass_alpha"] = float(low_pass_alpha)
        if scaling_factor is not None:
            cfg["scaling_factor"] = float(scaling_factor)
        self.retargeting = RetargetingConfig.from_dict(cfg).build()
        self.joint_names: list[str] = list(self.retargeting.joint_names)
        self.wrist_link: str = cfg["wrist_link_name"]
        self.openxr_to_base = np.asarray(self.spec.openxr_to_base, np.float64).reshape(3, 3)
        opt = self.retargeting.optimizer
        self._indices = np.asarray(opt.target_link_human_indices)
        self._position_type = opt.retargeting_type == "POSITION"

    def reset(self):
        self.retargeting.reset()

    def __call__(self, positions: np.ndarray, orientations_xyzw: np.ndarray) -> np.ndarray:
        """OpenXR-26 joints -> joint angles ordered as ``self.joint_names``."""
        import torch

        p = np.asarray(positions, np.float64)
        joints = p[OPENXR_TO_DEX21] - p[OPENXR_WRIST]
        wrist_rot = Rotation.from_quat(orientations_xyzw[OPENXR_WRIST]).as_matrix()
        target = joints @ wrist_rot @ self.openxr_to_base  # rows: wrist-local, then base frame
        if self._position_type:
            ref = target[self._indices]
        else:
            ref = target[self._indices[1]] - target[self._indices[0]]
        with torch.enable_grad(), torch.inference_mode(False):
            return np.asarray(self.retargeting.retarget(ref), np.float64)

    def base_pose(self, positions: np.ndarray, orientations_xyzw: np.ndarray):
        """Robot base link pose (position, XYZW quaternion) in the input frame, placed at
        the human wrist with the orientation implied by ``openxr_to_base``."""
        wrist_rot = Rotation.from_quat(orientations_xyzw[OPENXR_WRIST]).as_matrix()
        base_rot = wrist_rot @ self.openxr_to_base
        return np.asarray(positions[OPENXR_WRIST], np.float64), Rotation.from_matrix(base_rot).as_quat()


def serve(connect: str, publish: str, robot: str = "sharpa_wave", low_pass_alpha: float | None = 0.5,
          reset_after_s: float = 0.5, sides=("left", "right")):  # fmt: skip
    """Subscribe to hands messages, retarget each hand, publish robot-hands messages.

    Runs in its own process so the optimizer (Python callbacks, ~10 ms/hand) never
    competes with a simulator's render loop for the GIL.
    """
    import logging
    import time

    import zmq

    from .protocol import TOPIC_ROBOT, Subscriber, encode_robot_hands

    log = logging.getLogger("egohand_live.retarget")
    try:  # a few-variable optimisation per hand: more intra-op threads only contend for cores
        import torch

        torch.set_num_threads(1)
    except ImportError:
        pass
    retargeters = {s: HandRetargeter(robot, s, low_pass_alpha=low_pass_alpha) for s in sides}
    last_seen = dict.fromkeys(sides, 0.0)
    sub = Subscriber(connect)
    pub = zmq.Context.instance().socket(zmq.PUB)
    pub.setsockopt(zmq.SNDHWM, 4)
    pub.bind(publish)
    log.info("Retargeting %s -> %s on %s (%s, %d joints/hand)", connect, robot, publish,
             "/".join(sides), len(next(iter(retargeters.values())).joint_names))  # fmt: skip
    n, ms, t_log = 0, [], time.time()
    try:
        while True:
            msg, _ = sub.latest(200)
            if msg is not None:
                out = []
                for h in msg["hands"]:
                    side, rt = h["side"], retargeters.get(h["side"])
                    if rt is None:
                        continue
                    pos, quat = h["openxr26_positions"], h["openxr26_orientations"]
                    now = time.time()
                    if now - last_seen[side] > reset_after_s:
                        rt.reset()  # start fresh instead of filtering from a stale pose
                    last_seen[side] = now
                    t0 = time.perf_counter()
                    q = rt(pos, quat)
                    dt_ms = (time.perf_counter() - t0) * 1e3
                    base_p, base_q = rt.base_pose(pos, quat)
                    out.append({"side": side, "joint_names": rt.joint_names, "joint_positions": q,
                                "base_position": base_p, "base_orientation": base_q,
                                "openxr26_positions": pos, "retarget_ms": dt_ms})  # fmt: skip
                    ms.append(dt_ms)
                pub.send_multipart([TOPIC_ROBOT, encode_robot_hands(msg, robot, out)])
                n += 1
            if time.time() - t_log > 5.0:
                elapsed, t_log = time.time() - t_log, time.time()
                log.info("%5.1f msg/s%s", n / elapsed, f", retarget {np.median(ms):.1f} ms/hand" if ms else "")
                n, ms = 0, []
    except KeyboardInterrupt:
        pass
    finally:
        sub.close()
        pub.close(linger=0)
