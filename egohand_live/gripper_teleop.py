"""Tracked hand -> arm + parallel-gripper command (end-effector pose + gripper), simulator-free.

The rules follow NVIDIA Isaac Teleop's manipulator retargeters, so teleoperation feels as with a
headset: the end effector follows the hand and a thumb-index pinch drives the gripper (closes
below 3 cm, opens above 5 cm; the gap stops it flickering). Two additions for a desk camera:

* the gripper's fingertip centre follows the point between thumb and index fingertips, so a
  pinch closes the gripper where the fingers meet;
* ``continuous`` maps the pinch width to the gripper opening (half-pinch, half-open) instead of
  open/closed.

Frames: hands arrive as OpenXR-26 joints in the egohand-live desk frame (``T_world_camera``:
x right, y away from the operator, z up, desk at z = 0). ``desk_to_task`` places that frame in
a task's world, with the operator behind the robot (robot facing +x, as in Isaac Lab's stacking
tasks). End-effector axes: z = approach (pointing down for a top grasp), y = finger opening
direction, x = forward: Franka's ``panda_hand`` convention.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

WRIST, THUMB_TIP, INDEX_TIP = 1, 5, 10  # OpenXR-26
PINCH_CLOSE, PINCH_OPEN = 0.03, 0.05  # metres, Isaac Teleop GripperRetargeter thresholds


def desk_to_task(task_centre=(0.5, 0.0, 0.0), desk_centre=(0.0, 0.3, 0.0), scale: float = 1.0) -> np.ndarray:
    """4x4 transform from the desk frame to the task frame: desk +y -> task +x, desk +x -> task -y.

    ``desk_centre`` (the workspace centre of the egohand-live calibration) lands on ``task_centre``;
    ``scale`` scales hand motion about it (height above the desk included).
    """
    R = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]) * scale
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(task_centre, float) - R @ np.asarray(desk_centre, float)
    return T


def transform_hand(T: np.ndarray, positions: np.ndarray, quats_xyzw: np.ndarray):
    """Apply a 4x4 (rotation possibly scaled) to OpenXR joint positions and orientations."""
    R = T[:3, :3]
    s = np.cbrt(np.linalg.det(R))
    pos = np.asarray(positions, float) @ R.T + T[:3, 3]
    rot = Rotation.from_matrix(R / s) * Rotation.from_quat(quats_xyzw)
    return pos, rot.as_quat()


def ee_target(positions: np.ndarray, quats_xyzw: np.ndarray, rotation: str = "yaw", tcp_offset: float = 0.1034,
              min_height: float = 0.005):  # fmt: skip
    """End-effector body pose for one tracked hand, in the frame of ``positions``.

    Args:
        positions, quats_xyzw: OpenXR-26 joints (task frame, z up).
        rotation: "yaw" keeps the gripper pointing straight down and turns it with the hand;
            "full" follows the whole hand orientation (approach = palm normal).
        tcp_offset: distance from the end-effector body to the fingertip centre along its
            approach axis (Franka hand: 0.1034 m).
        min_height: lowest fingertip-centre height (keeps the gripper above the table).

    Returns:
        position (3,), quaternion (4,) as XYZW (Isaac Lab 3, PhysX, warp), pinch width (m).
    """
    p = np.asarray(positions, float)
    Rw = Rotation.from_quat(quats_xyzw[WRIST]).as_matrix()
    fingers, palmar = -Rw[:, 2], -Rw[:, 1]  # OpenXR: -Z distal, +Y dorsal
    if rotation == "yaw":
        z = np.array([0.0, 0.0, -1.0])
        x = fingers - (fingers @ z) * z
    else:
        z = palmar / np.linalg.norm(palmar)
        x = fingers - (fingers @ z) * z
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=1)
    tcp = 0.5 * (p[THUMB_TIP] + p[INDEX_TIP])
    tcp[2] = max(tcp[2], min_height)
    pos = tcp - tcp_offset * z
    return pos, Rotation.from_matrix(R).as_quat(), float(np.linalg.norm(p[THUMB_TIP] - p[INDEX_TIP]))


@dataclass
class PinchGripper:
    """Pinch width -> gripper command.

    binary: -1 close / +1 open with hysteresis (Isaac Lab's binary gripper action convention).
    continuous: opening fraction in [0, 1] (0 = closed), linear between ``closed_at`` and ``open_at``.
    """

    mode: str = "binary"
    close_below: float = PINCH_CLOSE
    open_above: float = PINCH_OPEN
    closed_at: float = 0.015
    open_at: float = 0.08
    _closed: bool = False

    def __call__(self, width: float) -> float:
        if self.mode == "continuous":
            return float(np.clip((width - self.closed_at) / (self.open_at - self.closed_at), 0.0, 1.0))
        if width < self.close_below:
            self._closed = True
        elif width > self.open_above:
            self._closed = False
        return -1.0 if self._closed else 1.0


def step_towards(current: np.ndarray, target: np.ndarray, max_step: float) -> np.ndarray:
    """Move ``current`` toward ``target`` by at most ``max_step`` (rate limit for the arm)."""
    d = np.asarray(target, float) - current
    n = np.linalg.norm(d)
    return target if n <= max_step else current + d * (max_step / n)
