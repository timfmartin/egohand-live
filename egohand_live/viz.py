"""OpenCV overlay and Rerun 3D logging."""

import logging
from collections import deque

import cv2
import numpy as np

from .skeleton import SOMA25_BONES

log = logging.getLogger("egohand_live.viz")

COLORS_BGR = {"left": (255, 160, 40), "right": (60, 60, 255)}


class RateMeter:
    def __init__(self, window=60):
        self.times = deque(maxlen=window)

    def tick(self, t):
        self.times.append(t)

    @property
    def fps(self):
        if len(self.times) < 2:
            return 0.0
        return (len(self.times) - 1) / max(self.times[-1] - self.times[0], 1e-6)


def draw_hand(img, kp2d, color, thickness=2):
    pts = np.round(kp2d).astype(int)
    for a, b in SOMA25_BONES:
        cv2.line(img, tuple(pts[a]), tuple(pts[b]), color, thickness, cv2.LINE_AA)
    for p in pts:
        cv2.circle(img, tuple(p), 3, (255, 255, 255), -1, cv2.LINE_AA)


def draw_box(img, bbox, color, label):
    cx, cy, s = bbox
    p0, p1 = (int(cx - s / 2), int(cy - s / 2)), (int(cx + s / 2), int(cy + s / 2))
    cv2.rectangle(img, p0, p1, color, 1, cv2.LINE_AA)
    cv2.putText(img, label, (p0[0], max(p0[1] - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                color, 1, cv2.LINE_AA)  # fmt: skip


def draw_overlay(bgr, hands: dict, fps: float, latency_ms: float | None, timings: dict):
    """Draw skeletons, boxes and stats. ``hands``: side -> object with ``keypoints_2d``,
    ``bbox``, ``score``, ``held``, ``joints_cam`` (FrameResult hands or decoded messages)."""
    vis = bgr.copy()
    for side, h in hands.items():
        color = COLORS_BGR[side]
        kp2d = h["keypoints_2d"] if isinstance(h, dict) else h.keypoints_2d
        bbox = h["bbox"] if isinstance(h, dict) else h.bbox
        score = h["score"] if isinstance(h, dict) else h.score
        held = h["held"] if isinstance(h, dict) else h.held
        joints = h["soma25_positions"] if isinstance(h, dict) else h.joints_cam
        draw_box(vis, bbox, color, f"{side} {score:.2f}{' held' if held else ''}  z={joints[0, 2]:.2f}m")
        draw_hand(vis, kp2d, color)
    lines = [f"{fps:5.1f} FPS"]
    if latency_ms is not None:
        lines.append(f"latency {latency_ms:5.1f} ms")
    lines += [f"{k} {v:5.1f} ms" for k, v in timings.items()]
    for i, text in enumerate(lines):
        y = 22 + 20 * i
        cv2.putText(vis, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


class RerunLogger:
    """Stream image, 2D/3D skeletons and meshes to a Rerun viewer.

    ``target``: "spawn" (local viewer), a gRPC URL such as
    ``rerun+http://LAPTOP_IP:9876/proxy`` (remote viewer), or a ``.rrd`` file path.
    """

    def __init__(self, target: str = "spawn", faces: dict | None = None, app_id="egohand_live"):
        import rerun as rr

        self.rr = rr
        rr.init(app_id)
        if target == "spawn":
            rr.spawn()
        elif target.endswith(".rrd"):
            rr.save(target)
        else:
            rr.connect_grpc(target)
        self.faces = faces or {}
        rr.log("world", rr.ViewCoordinates.RDF, static=True)  # OpenCV camera axes
        rgb = {s: tuple(int(c) for c in reversed(COLORS_BGR[s])) for s in COLORS_BGR}
        self.rgb = rgb

    def log(self, frame_bgr, result, jpeg_quality=75):
        rr = self.rr
        rr.set_time("frame", sequence=result.index)
        rr.set_time("capture", timestamp=result.t_capture)
        w, h = result.image_size
        rr.log("world/camera", rr.Pinhole(image_from_camera=result.K, resolution=[w, h]))
        ok, jpg = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        if ok:
            rr.log("world/camera/image", rr.EncodedImage(contents=jpg.tobytes(), media_type="image/jpeg"))
        for side in ("left", "right"):
            base = f"world/hands/{side}"
            h_ = result.hands.get(side)
            if h_ is None:
                rr.log(base, rr.Clear(recursive=True))
                rr.log(f"world/camera/image/{side}", rr.Clear(recursive=True))
                continue
            color = self.rgb[side]
            j = h_.joints_cam
            rr.log(f"{base}/joints", rr.Points3D(j, radii=0.004, colors=color))
            rr.log(f"{base}/bones", rr.LineStrips3D([[j[a], j[b]] for a, b in SOMA25_BONES], colors=color, radii=0.002))
            rr.log(f"world/camera/image/{side}", rr.LineStrips2D(
                [[h_.keypoints_2d[a], h_.keypoints_2d[b]] for a, b in SOMA25_BONES], colors=color))  # fmt: skip
            if h_.vertices_cam is not None and side in self.faces:
                rr.log(f"{base}/mesh", rr.Mesh3D(vertex_positions=h_.vertices_cam,
                       triangle_indices=self.faces[side], albedo_factor=[*color, 160]))  # fmt: skip
            rr.log(f"{base}/wrist_depth_m", rr.Scalars(float(j[0, 2])))
        for k, v in result.timings_ms.items():
            rr.log(f"timing_ms/{k}", rr.Scalars(v))
