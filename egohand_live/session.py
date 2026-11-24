"""Record a live session (frames + everything computed per frame) to render videos later.

``egohand-live run --save-session DIR`` writes::

    DIR/meta.json        image size, camera pose, depth scale, source
    DIR/faces.npz        hand mesh triangles (left, right)
    DIR/frames/*.jpg     the frames the tracker saw (undistorted, rotated: they match K)
    DIR/results.msgpack  one record per frame: time, K, per hand joints / OpenXR poses / mesh

Writing runs on a background thread so the live loop is not slowed down; if the disk cannot
keep up, frames are dropped from the recording (counted), never from tracking.
``egohand-live render-video DIR`` turns a session into a video (see video.py).
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from pathlib import Path

import msgpack
import numpy as np

log = logging.getLogger("egohand_live.session")


def _pack(a) -> dict:
    a = np.ascontiguousarray(a)
    return {"dtype": a.dtype.str, "shape": list(a.shape), "data": a.tobytes()}


def _unpack(d) -> np.ndarray:
    return np.frombuffer(d["data"], dtype=np.dtype(d["dtype"])).reshape(d["shape"])


def _hand_record(h) -> dict:
    rec = {
        "side": h.side,
        "score": float(h.score),
        "held": bool(h.held),
        "bbox": _pack(np.asarray(h.bbox, np.float32)),
        "joints": _pack(np.asarray(h.joints_cam, np.float32)),
        "kp2d": _pack(np.asarray(h.keypoints_2d, np.float32)),
        "oxr_pos": _pack(np.asarray(h.openxr_positions, np.float32)),
        "oxr_quat": _pack(np.asarray(h.openxr_orientations, np.float32)),
    }
    if getattr(h, "vertices_cam", None) is not None:
        rec["verts"] = _pack(np.asarray(h.vertices_cam, np.float32))
    return rec


class SessionWriter:
    def __init__(self, path: str | Path, meta: dict, faces: dict | None = None, jpeg_quality: int = 92):
        self.dir = Path(path)
        (self.dir / "frames").mkdir(parents=True, exist_ok=True)
        meta = {"version": 1, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), **meta}
        (self.dir / "meta.json").write_text(json.dumps(meta, indent=2))
        if faces:
            np.savez(self.dir / "faces.npz", **{k: np.asarray(v) for k, v in faces.items()})
        self.quality = jpeg_quality
        self._q: queue.Queue = queue.Queue(maxsize=120)
        self._results = open(self.dir / "results.msgpack", "wb")
        self.written = self.dropped = 0
        self._thread = threading.Thread(target=self._run, name="session-writer", daemon=True)
        self._thread.start()

    def add(self, bgr: np.ndarray, result) -> None:
        rec = {
            "i": int(result.index),
            "t": float(result.t_capture),
            "K": _pack(np.asarray(result.K, np.float32)),
            "timings": {k: float(v) for k, v in result.timings_ms.items()},
            "hands": [_hand_record(h) for h in result.hands.values()],
        }
        try:
            self._q.put_nowait((bgr, rec))
        except queue.Full:
            self.dropped += 1

    def _run(self):
        import cv2

        while True:
            item = self._q.get()
            if item is None:
                break
            bgr, rec = item
            name = f"{rec['i']:06d}.jpg"
            cv2.imwrite(str(self.dir / "frames" / name), bgr, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
            rec["frame"] = name
            self._results.write(msgpack.packb(rec, use_bin_type=True))
            self.written += 1

    def close(self):
        self._q.put(None)
        self._thread.join()
        self._results.close()
        log.info("session %s: %d frames written, %d dropped", self.dir, self.written, self.dropped)


def load_session(path: str | Path):
    """Returns (meta dict, faces dict or None, list of per-frame records with numpy arrays)."""
    d = Path(path)
    meta = json.loads((d / "meta.json").read_text())
    faces = None
    if (d / "faces.npz").exists():
        with np.load(d / "faces.npz") as z:
            faces = {k: z[k] for k in z.files}
    records = []
    with open(d / "results.msgpack", "rb") as f:
        for rec in msgpack.Unpacker(f, raw=False):
            rec["K"] = _unpack(rec["K"])
            rec["frame"] = str(d / "frames" / rec["frame"])
            for h in rec["hands"]:
                for k, v in list(h.items()):
                    if isinstance(v, dict) and "data" in v:
                        h[k] = _unpack(v)
            records.append(rec)
    return meta, faces, records
