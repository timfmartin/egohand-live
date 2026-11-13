"""Wire format: msgpack over ZeroMQ PUB/SUB.

Topics (first multipart frame):
    b"hands"  [topic, msgpack(hands message)]
    b"frame"  [topic, msgpack(frame meta), JPEG bytes]

Hands message (schema ``egohand-live/hands/v1``); arrays are little-endian float32
bytes with the shape given here:

    seq            int
    t_capture      float  capture time, seconds since epoch (sender clock)
    t_sent         float
    image_size     [w, h]
    K              (3, 3)
    frame          "opencv_camera": +X right, +Y down, +Z forward, meters
    T_world_camera (4, 4) or nil, optional fixed camera extrinsic
    latency_ms     {stage: float}
    hands          list of:
        side                 "left" | "right"
        score, held          detector confidence; held = box carried over a missed detection
        bbox                 (3,)  crop [cx, cy, size] px
        soma25_positions     (25, 3)
        keypoints_2d         (25, 2) px
        openxr26_positions   (26, 3)
        openxr26_orientations (26, 4) XYZW
        openxr26_radii       (26,)

Robot hands message (topic b"robot", schema ``egohand-live/robot-hands/v1``), published
by ``egohand-live retarget``; seq / t_capture / T_world_camera are copied from the hands
message it was computed from:

    robot          e.g. "sharpa_wave"
    t_retarget     float
    hands          list of:
        side                 "left" | "right"
        joint_names          [str]  robot joint names, same order as joint_positions
        joint_positions      (n,)   rad
        base_position        (3,)   robot base (wrist) link, camera frame
        base_orientation     (4,)   XYZW, camera frame
        openxr26_positions   (26, 3) the tracked human joints, camera frame
        retarget_ms          float
"""

import time

import cv2
import msgpack
import numpy as np

SCHEMA = "egohand-live/hands/v1"
TOPIC_HANDS = b"hands"
TOPIC_FRAME = b"frame"
ROBOT_SCHEMA = "egohand-live/robot-hands/v1"
TOPIC_ROBOT = b"robot"


def _arr(a) -> bytes:
    return np.ascontiguousarray(a, dtype="<f4").tobytes()


def _unarr(b: bytes, shape) -> np.ndarray:
    return np.frombuffer(b, dtype="<f4").reshape(shape)


def encode_hands(result, T_world_camera=None) -> bytes:
    hands = []
    for h in result.hands.values():
        hands.append(
            {
                "side": h.side,
                "score": float(h.score),
                "held": bool(h.held),
                "bbox": _arr(h.bbox),
                "soma25_positions": _arr(h.joints_cam),
                "keypoints_2d": _arr(h.keypoints_2d),
                "openxr26_positions": _arr(h.openxr_positions),
                "openxr26_orientations": _arr(h.openxr_orientations),
                "openxr26_radii": _arr(h.openxr_radii),
            }
        )
    msg = {
        "schema": SCHEMA,
        "seq": int(result.index),
        "t_capture": float(result.t_capture),
        "t_sent": time.time(),
        "image_size": list(result.image_size),
        "K": _arr(result.K),
        "frame": "opencv_camera",
        "T_world_camera": None if T_world_camera is None else _arr(T_world_camera),
        "latency_ms": {k: round(float(v), 3) for k, v in result.timings_ms.items()},
        "hands": hands,
    }
    return msgpack.packb(msg, use_bin_type=True)


_SHAPES = {
    "bbox": (3,),
    "soma25_positions": (25, 3),
    "keypoints_2d": (25, 2),
    "openxr26_positions": (26, 3),
    "openxr26_orientations": (26, 4),
    "openxr26_radii": (26,),
}


def decode_hands(buf: bytes) -> dict:
    msg = msgpack.unpackb(buf, raw=False)
    if msg.get("schema") != SCHEMA:
        raise ValueError(f"unexpected schema {msg.get('schema')!r}")
    msg["K"] = _unarr(msg["K"], (3, 3))
    if msg.get("T_world_camera") is not None:
        msg["T_world_camera"] = _unarr(msg["T_world_camera"], (4, 4))
    for h in msg["hands"]:
        for key, shape in _SHAPES.items():
            h[key] = _unarr(h[key], shape)
    return msg


def encode_robot_hands(source_msg: dict, robot: str, hands: list[dict]) -> bytes:
    T = source_msg.get("T_world_camera")
    msg = {
        "schema": ROBOT_SCHEMA,
        "robot": robot,
        "seq": source_msg["seq"],
        "t_capture": source_msg["t_capture"],
        "t_retarget": time.time(),
        "T_world_camera": None if T is None else _arr(T),
        "hands": [
            {
                "side": h["side"],
                "joint_names": list(h["joint_names"]),
                "joint_positions": _arr(h["joint_positions"]),
                "base_position": _arr(h["base_position"]),
                "base_orientation": _arr(h["base_orientation"]),
                "openxr26_positions": _arr(h["openxr26_positions"]),
                "retarget_ms": round(float(h["retarget_ms"]), 3),
            }
            for h in hands
        ],
    }
    return msgpack.packb(msg, use_bin_type=True)


def decode_robot_hands(buf: bytes) -> dict:
    msg = msgpack.unpackb(buf, raw=False)
    if msg.get("schema") != ROBOT_SCHEMA:
        raise ValueError(f"unexpected schema {msg.get('schema')!r}")
    if msg.get("T_world_camera") is not None:
        msg["T_world_camera"] = _unarr(msg["T_world_camera"], (4, 4))
    for h in msg["hands"]:
        h["joint_positions"] = _unarr(h["joint_positions"], (len(h["joint_names"]),))
        h["base_position"] = _unarr(h["base_position"], (3,))
        h["base_orientation"] = _unarr(h["base_orientation"], (4,))
        h["openxr26_positions"] = _unarr(h["openxr26_positions"], (26, 3))
    return msg


def encode_frame(bgr, K, t, index, quality=85) -> list[bytes]:
    ok, jpg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    meta = {"t": float(t), "index": int(index), "K": np.asarray(K, float).ravel().tolist(),
            "w": bgr.shape[1], "h": bgr.shape[0]}  # fmt: skip
    return [TOPIC_FRAME, msgpack.packb(meta), jpg.tobytes()]


def decode_frame(parts) -> tuple[dict, np.ndarray]:
    _, meta, jpg = parts
    bgr = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
    return msgpack.unpackb(meta, raw=False), bgr


def bind_or_connect(sock, endpoint: str) -> str:
    """Bind for wildcard endpoints (``tcp://*:PORT``, ``tcp://0.0.0.0:PORT``), else connect.

    ZMQ lets either end of a PUB/SUB pair bind, so the side that can accept incoming
    connections (e.g. the GPU host, when the camera laptop has a firewall) can listen.
    Returns ``"bind"`` or ``"connect"``.
    """
    host = endpoint.split("://", 1)[-1].rsplit(":", 1)[0]
    if host in ("*", "0.0.0.0"):
        sock.bind(endpoint)
        return "bind"
    sock.connect(endpoint)
    return "connect"


class Publisher:
    """PUB socket for hand results (and optionally JPEG frames for remote viewers)."""

    def __init__(self, bind: str = "tcp://*:5555", T_world_camera=None):
        import zmq

        self.sock = zmq.Context.instance().socket(zmq.PUB)
        self.sock.setsockopt(zmq.SNDHWM, 4)  # drop, don't queue, for slow subscribers
        self.sock.bind(bind)
        self.T_world_camera = T_world_camera

    def publish(self, result, frame_bgr=None, jpeg_quality=80):
        self.sock.send_multipart([TOPIC_HANDS, encode_hands(result, self.T_world_camera)])
        if frame_bgr is not None:
            self.sock.send_multipart(
                encode_frame(frame_bgr, result.K, result.t_capture, result.index, jpeg_quality)
            )

    def close(self):
        self.sock.close(linger=0)


class Subscriber:
    """SUB socket; ``latest()`` drains the queue and returns only the newest messages."""

    def __init__(self, connect: str = "tcp://localhost:5555", frames: bool = False):
        import zmq

        self.zmq = zmq
        self.sock = zmq.Context.instance().socket(zmq.SUB)
        self.sock.setsockopt(zmq.RCVHWM, 8)
        self.sock.setsockopt(zmq.SUBSCRIBE, TOPIC_HANDS)
        if frames:
            self.sock.setsockopt(zmq.SUBSCRIBE, TOPIC_FRAME)
        self.sock.connect(connect)

    def latest(self, timeout_ms: int = 100) -> tuple[dict | None, tuple | None]:
        """Returns (newest hands message, newest (frame_meta, bgr)) — either may be None."""
        hands = frame = None
        if not self.sock.poll(timeout_ms):
            return None, None
        while True:
            try:
                parts = self.sock.recv_multipart(flags=self.zmq.NOBLOCK)
            except self.zmq.Again:
                break
            if parts[0] == TOPIC_HANDS:
                hands = parts[1]
            elif parts[0] == TOPIC_FRAME:
                frame = parts
        return (
            decode_hands(hands) if hands is not None else None,
            decode_frame(frame) if frame is not None else None,
        )

    def close(self):
        self.sock.close(linger=0)
