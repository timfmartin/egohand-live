"""Camera pose over a desk from RealSense depth, by fitting the desk plane.

The plane gives the camera's height, tilt and roll. Position along the desk and yaw are
not observable from a plane, so the world origin is the desk point straight below the
camera, +y is the viewing direction projected onto the desk and +z is up. With the D405
desk mount (``hardware/d405_desk_mount``), the origin is under the bar instead, as in
``mount_geometry.py``, and the equivalent collar height and tilt-scale reading are printed.

Stereo depth needs texture: the D405 has no IR projector, so a plain white desk gives
wrong depth. Lay a newspaper, magazine or printed page flat under the camera, keep hands
out of view, and check the consistency figures that are printed.
"""

from __future__ import annotations

import importlib.util
import logging
import math
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger("egohand_live.calibrate")
REPO_ROOT = Path(__file__).resolve().parents[1]
MOUNT_GEOMETRY = REPO_ROOT / "hardware" / "d405_desk_mount" / "mount_geometry.py"


def fit_plane(points: np.ndarray, thresh: float = 0.005, iters: int = 1500, seed: int = 0):
    """RANSAC + least-squares plane through ``points`` (N, 3), camera at the origin.

    Returns (n, d, inliers) with ``n @ p + d = 0`` and ``n`` pointing to the camera's
    side, so ``d`` is the camera's distance to the plane.
    """
    rng = np.random.default_rng(seed)
    best, best_k = None, -1
    for _ in range(iters):
        a, b, c = points[rng.choice(len(points), 3, replace=False)]
        n = np.cross(b - a, c - a)
        norm = np.linalg.norm(n)
        if norm < 1e-12:
            continue
        inl = np.abs((points - a) @ (n / norm)) < thresh
        if inl.sum() > best_k:
            best, best_k = inl, inl.sum()
    c0 = points[best].mean(0)
    n = np.linalg.svd(points[best] - c0, full_matrices=False)[2][2]
    d = -float(n @ c0)
    if d < 0:
        n, d = -n, -d
    return n, d, np.abs(points @ n + d) < thresh


def tilt_roll_deg(n: np.ndarray) -> tuple[float, float]:
    """Tilt (optical axis below horizontal) and roll (camera's right side up) in degrees,
    from the desk normal ``n`` (pointing up) in camera coordinates."""
    return math.degrees(math.asin(-n[2])), math.degrees(math.atan2(n[0], -n[1]))


def extrinsics_from_plane(n: np.ndarray, d: float, camera_xy=(0.0, 0.0)) -> np.ndarray:
    """4x4 T_world_camera for a desk plane (n, d) in camera coordinates; the camera sits at
    world (camera_xy, d) looking along +y."""
    up = n / np.linalg.norm(n)
    fwd = np.array([0.0, 0.0, 1.0]) - up[2] * up
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, up)
    T = np.eye(4)
    T[:3, :3] = np.stack([right, fwd, up], 1).T  # rows: world axes in camera coordinates
    T[:3, 3] = [camera_xy[0], camera_xy[1], d]
    return T


def _mount_geometry():
    spec = importlib.util.spec_from_file_location("mount_geometry", MOUNT_GEOMETRY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def d405_mount_settings(height_m: float, tilt_deg: float):
    """Invert ``mount_geometry.t_world_camera``: camera height + tilt -> collar height (mm)
    and the camera's (x, y) over the desk in metres (origin under the bar)."""
    mg = _mount_geometry()
    x, y, z = mg.camera_origin_local()
    s, c = math.sin(math.radians(tilt_deg)), math.cos(math.radians(tilt_deg))
    collar = height_m * 1000.0 - mg.PIVOT_Z + y * s - z * c
    return collar, (x / 1000.0, (mg.PIVOT_Y + y * c + z * s) / 1000.0)


@dataclass
class DeskFit:
    n: np.ndarray
    d: float
    inlier_frac: float
    rms_mm: float
    spread_deg: float  # worst normal disagreement between image quadrants
    spread_mm: float  # worst height disagreement between image quadrants

    @property
    def reliable(self) -> bool:
        return self.spread_deg < 2.0 and self.spread_mm < 10.0 and self.inlier_frac > 0.5


def fit_desk(points: np.ndarray, uv: np.ndarray, image_size, thresh: float = 0.005, iters: int = 1500) -> DeskFit:
    """Fit the desk in the lower 60 % of the image and cross-check it per quadrant."""
    w, h = image_size
    roi = uv[:, 1] > 0.4 * h
    pts, puv = points[roi], uv[roi]
    n, d, inl = fit_plane(pts, thresh, iters)
    rms = float(np.sqrt(np.mean((pts[inl] @ n + d) ** 2)) * 1e3)
    angles, heights = [], []
    mid_v = 0.7 * h
    for qu in (puv[:, 0] < w / 2, puv[:, 0] >= w / 2):
        for qv in (puv[:, 1] < mid_v, puv[:, 1] >= mid_v):
            q = pts[qu & qv]
            if len(q) < 200:
                continue
            nq, dq, _ = fit_plane(q, thresh, iters=max(100, iters // 3))
            angles.append(math.degrees(math.acos(min(1.0, abs(float(nq @ n))))))
            heights.append(abs(dq - d) * 1e3)
    return DeskFit(n, d, float(inl.mean()), rms, max(angles, default=0.0), max(heights, default=0.0))


def capture_points(serial=None, frames=30, width=1280, height=720, stride=4):
    """Median depth over ``frames`` (aligned to colour), deprojected with the colour
    intrinsics. Returns (points (N, 3), pixel uv (N, 2), (width, height), colour BGR)."""
    import pyrealsense2 as rs

    pipe, cfg = rs.pipeline(), rs.config()
    if serial:
        cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, 30)
    cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)
    profile = pipe.start(cfg)
    try:
        scale = profile.get_device().first_depth_sensor().get_depth_scale()
        align = rs.align(rs.stream.color)
        for _ in range(15):  # let auto exposure settle
            pipe.wait_for_frames()
        depths = []
        for _ in range(frames):
            fs = align.process(pipe.wait_for_frames())
            depths.append(np.asanyarray(fs.get_depth_frame().get_data())[::stride, ::stride] * scale)
            color = np.asanyarray(fs.get_color_frame().get_data()).copy()
        intr = fs.get_color_frame().profile.as_video_stream_profile().get_intrinsics()
    finally:
        pipe.stop()
    stack = np.stack(depths).astype(np.float32)
    stack[stack <= 0] = np.nan
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(stack, 0)
        stable = (np.isfinite(stack).mean(0) > 0.8) & (np.nanstd(stack, 0) < 0.004)
    vs, us = np.nonzero(stable & (med > 0.07) & (med < 1.5))
    pts = np.array([rs.rs2_deproject_pixel_to_point(intr, [float(u * stride), float(v * stride)], float(med[v, u]))
                    for v, u in zip(vs, us)])  # fmt: skip
    return pts, np.stack([us, vs], 1) * stride, (width, height), color


def run(serial=None, frames=30, d405_mount=False, out=None, preview=None) -> np.ndarray:
    pts, uv, size, color = capture_points(serial, frames)
    if len(pts) < 1000:
        raise SystemExit(f"only {len(pts)} valid depth points; is the desk in view and textured?")
    fit = fit_desk(pts, uv, size)
    tilt, roll = tilt_roll_deg(fit.n)
    print(f"camera {fit.d * 1000:.0f} mm above the desk, tilted {tilt:.1f} deg down, roll {roll:+.1f} deg")
    print(f"fit: {fit.inlier_frac:.0%} inliers, rms {fit.rms_mm:.1f} mm; quadrants agree within "
          f"{fit.spread_deg:.1f} deg / {fit.spread_mm:.0f} mm")  # fmt: skip
    if not fit.reliable:
        print("WARNING: inconsistent fit. Lay a textured sheet (newspaper, printed page) flat under the "
              "camera, keep hands out of view, and run again.")  # fmt: skip
    camera_xy = (0.0, 0.0)
    if d405_mount:
        collar, camera_xy = d405_mount_settings(fit.d, tilt)
        print(f"D405 mount: collar height {collar:.0f} mm, tilt scale {tilt:.0f} deg")
    T = extrinsics_from_plane(fit.n, fit.d, camera_xy)
    if out:
        np.savetxt(out, T, fmt="% .6f")
        print(f"wrote {out}")
    if preview:
        import cv2

        vis = color.copy()  # desk-plane points green, others red
        on = np.abs(pts @ fit.n + fit.d) < 0.005
        for (u, v), ok in zip(uv, on):
            cv2.circle(vis, (int(u), int(v)), 1, (0, 255, 0) if ok else (0, 0, 255), -1)
        cv2.imwrite(preview, vis)
    return T


# Desk outline of isaac/scenes.py (world frame of the D405 mount), drawn in the setup view.
SIM_DESK = ((-0.6, -0.05), (0.6, -0.05), (0.6, 0.75), (-0.6, 0.75))
SIM_PROPS = ((-0.42, 0.10), (0.42, 0.10), (0.42, 0.66), (-0.42, 0.66))


def _draw_world_poly(img, poly_xy, T, K, color, thickness=2):
    """Draw a closed polygon on the desk (z = 0, world frame) into the image."""
    import cv2

    R, t = T[:3, :3], T[:3, 3]
    for a, b in zip(poly_xy, poly_xy[1:] + poly_xy[:1]):
        seg = np.linspace([*a, 0.0], [*b, 0.0], 40)
        pc = (seg - t) @ R  # world -> camera
        front = pc[:, 2] > 0.05
        uv = pc[:, :2] / np.maximum(pc[:, 2:], 1e-6) * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
        for i in range(len(seg) - 1):
            if front[i] and front[i + 1]:
                cv2.line(img, tuple(int(x) for x in uv[i]), tuple(int(x) for x in uv[i + 1]), color, thickness, cv2.LINE_AA)


def live_view(serial=None, d405_mount=False, out=None, width=1280, height=720, stride=8):
    """Live colour view with the depth-measured camera pose, for setting up the mount.

    Keys: s = save T_world_camera to ``out``, d = toggle depth view, q / Esc = quit.
    """
    from collections import deque

    import cv2
    import pyrealsense2 as rs

    pipe, cfg = rs.pipeline(), rs.config()
    if serial:
        cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, 30)
    cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)
    profile = pipe.start(cfg)
    scale = profile.get_device().first_depth_sensor().get_depth_scale()
    align = rs.align(rs.stream.color)
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]])
    vs, us = np.mgrid[0 : height : stride, 0:width:stride]
    rays = np.array([rs.rs2_deproject_pixel_to_point(intr, [float(u), float(v)], 1.0)
                     for u, v in zip(us.ravel(), vs.ravel())]).reshape(*us.shape, 3)  # fmt: skip
    uv_grid = np.stack([us, vs], -1)
    hist: deque = deque(maxlen=12)
    fit = T = None
    show_depth, k = False, 0
    win = "egohand-live camera setup (s: save, d: depth, q: quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    try:
        while True:
            fs = align.process(pipe.wait_for_frames())
            color = np.asanyarray(fs.get_color_frame().get_data()).copy()
            depth = np.asanyarray(fs.get_depth_frame().get_data())
            d = depth[::stride, ::stride].astype(np.float32) * scale
            d[d <= 0] = np.nan
            hist.append(d)
            k += 1
            if k % 6 == 0 and len(hist) >= 6:
                stack = np.stack(hist)
                with np.errstate(all="ignore"), warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    med = np.nanmedian(stack, 0)
                    ok = (np.isfinite(stack).mean(0) > 0.8) & (np.nanstd(stack, 0) < 0.004)
                ok &= (med > 0.07) & (med < 1.5)
                if ok.sum() > 500:
                    fit = fit_desk(rays[ok] * med[ok][:, None], uv_grid[ok], (width, height), iters=300)
                    tilt, _ = tilt_roll_deg(fit.n)
                    xy = d405_mount_settings(fit.d, tilt)[1] if d405_mount else (0.0, 0.0)
                    T = extrinsics_from_plane(fit.n, fit.d, xy)
            if show_depth:
                vis = cv2.applyColorMap(cv2.convertScaleAbs(np.clip(depth * scale, 0, 1.2), alpha=255 / 1.2),
                                        cv2.COLORMAP_TURBO)  # fmt: skip
                vis[depth == 0] = 0
            else:
                vis = color
            cv2.drawMarker(vis, (width // 2, height // 2), (255, 255, 255), cv2.MARKER_CROSS, 40, 1)
            lines = []
            if fit is not None:
                tilt, roll = tilt_roll_deg(fit.n)
                lines.append(f"height {fit.d * 1000:.0f} mm   tilt {tilt:.1f} deg   roll {roll:+.1f} deg")
                if d405_mount:
                    lines.append(f"D405 mount: collar {d405_mount_settings(fit.d, tilt)[0]:.0f} mm")
                good = fit.reliable
                lines.append(f"fit {'OK' if good else 'UNRELIABLE (needs texture on the desk)'}: "
                             f"quadrants within {fit.spread_deg:.1f} deg / {fit.spread_mm:.0f} mm")  # fmt: skip
                if T is not None and d405_mount:
                    _draw_world_poly(vis, list(SIM_DESK), T, K, (0, 200, 255))
                    _draw_world_poly(vis, list(SIM_PROPS), T, K, (0, 255, 0), 1)
            else:
                lines.append("measuring...")
                good = True
            for i, text in enumerate(lines):
                y = 32 + 30 * i
                cv2.putText(vis, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
                col = (255, 255, 255) if good or i < len(lines) - 1 else (0, 0, 255)
                cv2.putText(vis, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2, cv2.LINE_AA)
            cv2.imshow(win, vis)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("d"):
                show_depth = not show_depth
            if key == ord("s"):
                if T is None or not out:
                    print("nothing to save" if out else "pass --out FILE to save")
                else:
                    np.savetxt(out, T, fmt="% .6f")
                    tilt, roll = tilt_roll_deg(fit.n)
                    print(f"wrote {out}: height {fit.d * 1000:.0f} mm, tilt {tilt:.1f}, roll {roll:+.1f}"
                          + ("" if fit.reliable else " (fit UNRELIABLE)"))  # fmt: skip
    finally:
        pipe.stop()
        cv2.destroyAllWindows()
