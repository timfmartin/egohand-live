"""Synthetic hand tracking: publishes moving, grasping hands as ``egohand-live/hands/v1``.

For testing the simulator side (retargeting, robot rigs, recording) without a camera or a
GPU. The hands move in slow loops over the desk (world frame of the D405 mount: desk
surface z = 0, +y away from the operator), palms down, and close into a grasp every
``grasp_period`` seconds. Messages carry ``T_world_camera``, so the simulator places them
correctly as long as it is not given a different ``--extrinsics``.
"""

from __future__ import annotations

import logging
import time
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from .skeleton import soma25_to_openxr26

log = logging.getLogger("egohand_live.synthetic")

IMAGE_SIZE = (1280, 720)
K = np.array([[652.0, 0, 640.0], [0, 652.0, 360.0], [0, 0, 1]])
# Camera 0.6 m above the desk origin, 45 deg down (only used to express the hands in a camera frame).
DEFAULT_T_WORLD_CAMERA = np.array(
    [[1, 0, 0, 0], [0, -0.7071068, 0.7071068, 0], [0, -0.7071068, -0.7071068, 0.6], [0, 0, 0, 1]]
)


def _bend(pts: np.ndarray, pivot: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    return pivot + Rotation.from_rotvec(axis / np.linalg.norm(axis) * angle).apply(pts - pivot)


def synthetic_hand(side: str, curl: float = 0.0) -> np.ndarray:
    """SOMA-25 joints of a palm-down hand, wrist at the origin, fingers along +Y, Z up (m).

    ``curl`` in [0, 1] flexes the fingers (60 deg per joint at 1) and brings the thumb
    toward the palm, roughly a power grasp.
    """
    lateral = {"index": -0.025, "middle": 0.0, "ring": 0.02, "pinky": 0.038}  # right hand: thumb at -X
    j = np.zeros((25, 3))
    j[1:5] = [[-0.02, 0.02, 0], [-0.04, 0.045, 0], [-0.055, 0.07, 0], [-0.065, 0.09, 0]]
    ang = curl * np.pi / 3
    for f, finger in enumerate(lateral):
        x, base = lateral[finger], 5 + 5 * f
        j[base : base + 5] = [[x * 0.4, 0.02, 0], [x, 0.09, 0], [x, 0.135, 0], [x, 0.16, 0], [x, 0.18, 0]]
        for k in (1, 2, 3):  # MCP, PIP, DIP flex toward the palm (-Z)
            j[base + k + 1 : base + 5] = _bend(j[base + k + 1 : base + 5], j[base + k], np.array([-1.0, 0, 0]), ang)
    # Thumb: swing toward the palm centre about the CMC, then flex MCP and IP.
    j[2:5] = _bend(j[2:5], j[1], np.array([0, 1.0, 0.6]), -0.9 * curl)
    for k in (2, 3):
        d = j[k + 1] - j[k]
        j[k + 1 : 5] = _bend(j[k + 1 : 5], j[k], np.cross(d, [0, 0, 1.0]), -0.6 * curl)
    if side == "left":
        j[:, 0] *= -1
    return j


def hand_pose(side: str, t: float, grasp_period: float = 4.0):
    """World wrist position, rotation and curl at time ``t`` (a slow Lissajous loop)."""
    s = -1.0 if side == "left" else 1.0
    pos = np.array([
        s * 0.16 + 0.07 * np.sin(0.5 * t + (0 if s < 0 else 1.3)),
        0.30 + 0.09 * np.sin(0.37 * t + (0.6 if s < 0 else 2.0)),
        0.11 + 0.04 * np.sin(0.8 * t + (0.0 if s < 0 else 1.0)),
    ])  # fmt: skip
    rot = Rotation.from_euler("zyx", [s * -0.25 * np.sin(0.4 * t), s * 0.2 * np.sin(0.6 * t), 0.15 * np.sin(0.5 * t)])
    curl = 0.5 - 0.5 * np.cos(2 * np.pi * t / grasp_period)
    return pos, rot.as_matrix(), float(curl)


def make_result(t: float, index: int, T_world_camera: np.ndarray, sides=("left", "right")):
    """A FrameResult-like object for ``protocol.encode_hands``."""
    R_wc, t_wc = T_world_camera[:3, :3], T_world_camera[:3, 3]
    hands = {}
    for side in sides:
        pos, rot, curl = hand_pose(side, t)
        joints_w = synthetic_hand(side, curl) @ rot.T + pos
        joints_c = (joints_w - t_wc) @ R_wc  # world -> camera
        oxr_p, oxr_q, oxr_r = soma25_to_openxr26(joints_c, side)
        uv = joints_c @ K.T
        kp = uv[:, :2] / uv[:, 2:]
        lo, hi = kp.min(0), kp.max(0)
        hands[side] = SimpleNamespace(
            side=side, score=0.95, held=False,
            bbox=np.array([*(lo + hi) / 2, (hi - lo).max() * 1.3], np.float32),
            joints_cam=joints_c.astype(np.float32), keypoints_2d=kp.astype(np.float32),
            openxr_positions=oxr_p, openxr_orientations=oxr_q, openxr_radii=oxr_r,
        )  # fmt: skip
    return SimpleNamespace(index=index, t_capture=time.time(), image_size=IMAGE_SIZE, K=K.astype(np.float32),
                           timings_ms={}, hands=hands)  # fmt: skip


def serve(publish: str, T_world_camera=None, fps: float = 30.0, sides=("left", "right"), seconds: float = 0.0):
    import zmq

    from .protocol import TOPIC_HANDS, bind_or_connect, encode_hands

    T = DEFAULT_T_WORLD_CAMERA if T_world_camera is None else np.asarray(T_world_camera, float).reshape(4, 4)
    sock = zmq.Context.instance().socket(zmq.PUB)
    sock.setsockopt(zmq.SNDHWM, 4)
    bind_or_connect(sock, publish)
    log.info("Publishing synthetic %s hands on %s at %.0f Hz", "+".join(sides), publish, fps)
    t0, i = time.time(), 0
    try:
        while not seconds or time.time() - t0 < seconds:
            sock.send_multipart([TOPIC_HANDS, encode_hands(make_result(time.time() - t0, i, T, sides), T)])
            i += 1
            time.sleep(max(0.0, t0 + i / fps - time.time()))
    except KeyboardInterrupt:
        pass
    finally:
        sock.close(linger=0)
