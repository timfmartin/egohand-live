"""Render a recorded session (``run --save-session``) as a video that builds up the pipeline.

Layers, drawn on the camera image at the output resolution:

* ``camera``   the frame the tracker saw
* ``skeleton`` 2D keypoints and bones (left blue, right red)
* ``pose3d``   the 3D skeleton coloured by distance to the camera (near = warm), thickness by
               depth, plus the wrist's coordinate axes
* ``mesh``     the hand model EgoHand fits (SOMA mesh), shaded, rendered with pyrender

The default timeline reveals them step by step with a wipe and a caption per step; ``--layers``
renders one fixed stack instead. Output is H.264 via ffmpeg (OpenCV mp4v if ffmpeg is missing).
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess

import cv2
import numpy as np

from .skeleton import SOMA25_BONES
from .viz import COLORS_BGR

log = logging.getLogger("egohand_live.video")

LAYERS = ("camera", "skeleton", "pose3d", "mesh")
DEFAULT_STEPS = [
    (("camera",), "Webcam"),
    (("camera", "skeleton"), "Hand tracking"),
    (("camera", "pose3d"), "3D hand pose"),
    (("camera", "mesh"), "Hand model"),
]
MESH_RGBA = {"left": (0.30, 0.62, 1.0, 0.80), "right": (1.0, 0.36, 0.30, 0.80)}


def parse_size(s: str) -> tuple[int, int]:
    w, h = (int(v) for v in s.lower().split("x"))
    return w, h


def frame_indices(times: np.ndarray, fps: float, start: float = 0.0, end: float | None = None) -> np.ndarray:
    """For a constant-rate output, the index of the latest recorded frame at each output time."""
    t0 = times[0] + start
    t1 = times[-1] if end is None else min(times[-1], times[0] + end)
    out_t = t0 + np.arange(max(int((t1 - t0) * fps), 1)) / fps
    return np.clip(np.searchsorted(times, out_t, side="right") - 1, 0, len(times) - 1)


def timeline(n_frames: int, fps: float, steps, step_s: float, wipe_s: float = 0.6):
    """Per output frame: (current step, previous step or None, wipe progress 0..1).

    Steps last ``step_s`` (shortened to fit short sessions); the last one holds to the end.
    """
    k = len(steps)
    total = n_frames / fps
    step_s = min(step_s, total / (k + 0.5)) if k > 1 else total
    plan = []
    for f in range(n_frames):
        t = f / fps
        cur = min(int(t // step_s), k - 1) if step_s > 0 else 0
        into = t - cur * step_s
        prog = 1.0 if cur == 0 or into >= wipe_s else into / wipe_s
        plan.append((cur, cur - 1 if prog < 1.0 else None, prog))
    return plan


class MeshRenderer:
    """Offscreen pyrender of hand meshes given in the OpenCV camera frame."""

    def __init__(self, width: int, height: int):
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        import pyrender

        self.pr = pyrender
        self.r = pyrender.OffscreenRenderer(width, height)

    def render(self, meshes, K) -> np.ndarray:
        import trimesh

        pr = self.pr
        scene = pr.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.35, 0.35, 0.35])
        flip = np.array([1.0, -1.0, -1.0])  # OpenCV camera -> OpenGL camera
        for verts, faces, rgba in meshes:
            tm = trimesh.Trimesh(np.asarray(verts, np.float64) * flip, faces, process=False)
            mat = pr.MetallicRoughnessMaterial(baseColorFactor=rgba, metallicFactor=0.0, roughnessFactor=0.55,
                                               alphaMode="BLEND", doubleSided=True)  # fmt: skip
            scene.add(pr.Mesh.from_trimesh(tm, material=mat, smooth=True))
        scene.add(pr.IntrinsicsCamera(K[0, 0], K[1, 1], K[0, 2], K[1, 2], znear=0.02, zfar=10.0), pose=np.eye(4))
        key = np.eye(4)
        key[:3, :3] = cv2.Rodrigues(np.array([-0.5, 0.3, 0.0]))[0]  # from above-left, toward the scene
        scene.add(pr.DirectionalLight(color=np.ones(3), intensity=3.0), pose=key)
        scene.add(pr.DirectionalLight(color=np.ones(3), intensity=1.5), pose=np.eye(4))
        rgba, _ = self.r.render(scene, flags=pr.RenderFlags.RGBA)
        return rgba

    def close(self):
        self.r.delete()


def _depth_colors(z: np.ndarray, z_near: float, z_far: float) -> np.ndarray:
    u = np.clip((z_far - z) / max(z_far - z_near, 1e-6), 0, 1)  # near -> 1 (warm)
    lut = cv2.applyColorMap(np.arange(256, dtype=np.uint8)[:, None], cv2.COLORMAP_TURBO)[:, 0]
    return lut[(u * 255).astype(int)]


def _project(K, P):
    uv = P @ K.T
    return uv[:, :2] / np.maximum(uv[:, 2:], 1e-6)


def draw_skeleton(img, h, K, scale):
    pts = np.round(_project(K, h["joints"])).astype(int)
    color = COLORS_BGR[h["side"]]
    t = max(int(round(2.5 * scale)), 1)
    for a, b in SOMA25_BONES:
        cv2.line(img, tuple(pts[a]), tuple(pts[b]), color, t, cv2.LINE_AA)
    for p in pts:
        cv2.circle(img, tuple(p), max(int(3 * scale), 2), (255, 255, 255), -1, cv2.LINE_AA)


def draw_pose3d(img, h, K, scale, z_range):
    from scipy.spatial.transform import Rotation

    J = h["joints"]
    pts = _project(K, J)
    cols = _depth_colors(J[:, 2], *z_range)
    for a, b in sorted(SOMA25_BONES, key=lambda ab: -(J[ab[0], 2] + J[ab[1], 2])):  # far first
        z = 0.5 * (J[a, 2] + J[b, 2])
        t = max(int(round(5.0 * scale * 0.6 / max(z, 0.2))), 1)
        c = tuple(int(v) for v in (cols[a].astype(int) + cols[b].astype(int)) // 2)
        cv2.line(img, tuple(np.round(pts[a]).astype(int)), tuple(np.round(pts[b]).astype(int)), c, t, cv2.LINE_AA)
    for j in np.argsort(-J[:, 2]):
        r = max(int(round(5.0 * scale * 0.6 / max(J[j, 2], 0.2))), 2)
        cv2.circle(img, tuple(np.round(pts[j]).astype(int)), r, tuple(int(v) for v in cols[j]), -1, cv2.LINE_AA)
    # wrist axes (OpenXR wrist frame: x red, y green, z blue), 5 cm
    w, q = h["oxr_pos"][1], h["oxr_quat"][1]
    R = Rotation.from_quat(q).as_matrix()
    o = _project(K, w[None])[0]
    for i, c in enumerate(((0, 0, 255), (0, 200, 0), (255, 80, 0))):
        e = _project(K, (w + 0.05 * R[:, i])[None])[0]
        cv2.arrowedLine(img, tuple(np.round(o).astype(int)), tuple(np.round(e).astype(int)), c,
                        max(int(2 * scale), 1), cv2.LINE_AA, tipLength=0.25)  # fmt: skip


def draw_caption(img, text, scale, index=None, alpha=1.0):
    font, fs, th = cv2.FONT_HERSHEY_DUPLEX, 1.25 * scale, max(int(2 * scale), 1)
    label = f"{index}   {text}" if index is not None else text
    (tw, tht), base = cv2.getTextSize(label, font, fs, th)
    x, y = int(40 * scale), img.shape[0] - int(48 * scale)
    pad = int(16 * scale)
    box = img.copy()
    cv2.rectangle(box, (x - pad, y - tht - pad), (x + tw + pad, y + base + pad // 2), (20, 20, 20), -1)
    cv2.addWeighted(box, 0.55 * alpha, img, 1 - 0.55 * alpha, 0, img)
    over = img.copy()
    cv2.putText(over, label, (x, y), font, fs, (255, 255, 255), th, cv2.LINE_AA)
    cv2.addWeighted(over, alpha, img, 1 - alpha, 0, img)


class VideoWriter:
    def __init__(self, path: str, size: tuple[int, int], fps: float, crf: int = 18):
        self.size = size
        if shutil.which("ffmpeg"):
            cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                   "-s", f"{size[0]}x{size[1]}", "-r", str(fps), "-i", "-", "-c:v", "libx264",
                   "-pix_fmt", "yuv420p", "-crf", str(crf), "-preset", "medium", "-movflags", "+faststart", path]  # fmt: skip
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            self.cv = None
        else:
            log.warning("ffmpeg not found; writing mp4v with OpenCV (re-encode for social media)")
            self.proc = None
            self.cv = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

    def write(self, img):
        if self.proc:
            self.proc.stdin.write(np.ascontiguousarray(img).tobytes())
        else:
            self.cv.write(img)

    def close(self):
        if self.proc:
            self.proc.stdin.close()
            self.proc.wait()
        else:
            self.cv.release()


def render(session_dir: str, out: str | None = None, size: str = "1920x1080", fps: float = 30.0,
           layers: str | None = None, step_s: float = 4.0, aspect: str = "16:9", captions: bool = True,
           start: float = 0.0, end: float | None = None) -> str:  # fmt: skip
    from .session import load_session

    meta, faces, recs = load_session(session_dir)
    if not recs:
        raise SystemExit("empty session")
    out = out or os.path.join(session_dir, "video.mp4")
    W, H = parse_size(size)
    iw, ih = meta["image_size"]
    s = W / iw  # camera image -> canvas scale (16:9 sessions fill the canvas)
    ch = int(round(ih * s))
    times = np.array([r["t"] for r in recs])
    idx = frame_indices(times, fps, start, end)
    steps = [(tuple(layers.split(",")), None)] if layers else DEFAULT_STEPS
    plan = timeline(len(idx), fps, steps, step_s)
    used = {lay for st, _ in steps for lay in st}
    z_all = np.concatenate([h["joints"][:, 2] for r in recs for h in r["hands"]] or [np.array([0.5, 1.0])])
    z_range = (float(np.percentile(z_all, 2)), float(np.percentile(z_all, 98)))
    mesh_r = MeshRenderer(W, ch) if "mesh" in used and faces is not None else None

    # crop to the requested aspect around where the hands are
    aw, ah = (int(v) for v in aspect.split(":"))
    cw = min(W, int(round(ch * aw / ah)))
    kp_x = [float(np.median(h["kp2d"][:, 0])) * s for r in recs for h in r["hands"]]
    cx = float(np.median(kp_x)) if kp_x else W / 2
    x0 = int(np.clip(cx - cw / 2, 0, W - cw))
    out_size = (cw, ch) if aspect != "16:9" else (W, ch)
    if aspect != "16:9":  # upscale the crop so the short side stays at the requested height
        out_size = (int(round(cw * H / ch)) // 2 * 2, H // 2 * 2) if ch < H else (cw // 2 * 2, ch // 2 * 2)
    writer = VideoWriter(out, out_size, fps)
    log.info("rendering %d frames (%s, %s) -> %s", len(idx), size, aspect, out)

    cache: dict[int, np.ndarray] = {}

    def stack(rec, names):
        if rec["i"] not in cache:
            cache.clear()
            cache[rec["i"]] = cv2.resize(cv2.imread(rec["frame"]), (W, ch), interpolation=cv2.INTER_CUBIC)
        img = cache[rec["i"]].copy() if "camera" in names else np.zeros((ch, W, 3), np.uint8)
        K = rec["K"].astype(np.float64).copy()
        K[:2] *= s
        hands = rec["hands"]
        if "mesh" in names and mesh_r is not None:
            meshes = [(h["verts"], faces[h["side"]], MESH_RGBA[h["side"]]) for h in hands if "verts" in h]
            if meshes:
                rgba = mesh_r.render(meshes, K)
                a = rgba[..., 3:4].astype(np.float32) / 255.0
                img = (img * (1 - a) + rgba[..., 2::-1] * a).astype(np.uint8)
        for h in hands:
            if "pose3d" in names:
                draw_pose3d(img, h, K, s, z_range)
            elif "skeleton" in names:
                draw_skeleton(img, h, K, s)
        return img

    try:
        for f, (i, (cur, prev, prog)) in enumerate(zip(idx, plan)):
            rec = recs[i]
            img = stack(rec, steps[cur][0])
            if prev is not None:  # wipe from the left: new step on the left of the edge
                old = stack(rec, steps[prev][0])
                edge = int(prog * W)
                ramp = np.clip((edge - np.arange(W)) / (0.03 * W) + 0.5, 0, 1)[None, :, None]
                img = (img * ramp + old * (1 - ramp)).astype(np.uint8)
                cv2.line(img, (edge, 0), (edge, ch), (255, 255, 255), max(int(2 * s), 1), cv2.LINE_AA)
            if captions and steps[cur][1]:
                fade = min(1.0, prog * 1.5) if cur > 0 else min(1.0, f / (0.5 * fps))
                draw_caption(img, steps[cur][1], s * 0.75, cur + 1, fade)
            if aspect != "16:9":
                img = cv2.resize(img[:, x0 : x0 + cw], out_size, interpolation=cv2.INTER_CUBIC)
            writer.write(img)
    finally:
        writer.close()
        if mesh_r:
            mesh_r.close()
    print(f"wrote {out} ({len(idx) / fps:.1f} s, {out_size[0]}x{out_size[1]})")
    return out
