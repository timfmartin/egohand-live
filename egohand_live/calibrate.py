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


# --------------------------------------------------------------------------- desk from hand touches

PALM_JOINTS = [0, 7, 12, 17, 22]  # OpenXR palm + index/middle/ring/little proximal joints
PALM_THICKNESS = 0.015  # m from the palm joints to the skin of the palm


def desk_frame_from_points(points_cam: np.ndarray, workspace_centre=(0.0, 0.3)) -> tuple[np.ndarray, float]:
    """T_world_camera from points on the desk (camera frame), plus the plane fit RMS [m].

    World +z is the desk normal toward the camera, +y the image's "up" direction projected on
    the desk (away from the operator, whose hands enter from the bottom of the image), and the
    points' centroid lands at ``workspace_centre``.
    """
    c = points_cam.mean(0)
    n = np.linalg.svd(points_cam - c, full_matrices=False)[2][2]
    if n @ (-c) < 0:  # point toward the camera (at the origin)
        n = -n
    rms = float(np.sqrt(np.mean(((points_cam - c) @ n) ** 2)))
    up_img = np.array([0.0, -1.0, 0.0])
    y = up_img - (up_img @ n) * n
    y /= np.linalg.norm(y)
    x = np.cross(y, n)
    R_cw = np.stack([x, y, n])  # rows: world axes in camera coordinates
    T = np.eye(4)
    T[:3, :3] = R_cw
    T[:3, 3] = np.array([workspace_centre[0], workspace_centre[1], 0.0]) - R_cw @ c
    return T, rms


MIN_SPREAD = 0.06  # m, RMS spread of the touch spots along their second axis


def spread_hint(points) -> str | None:
    """None when the spots span an area (the plane is determined), else where to add one.

    Spots along a line leave the tilt about that line free, whatever the per-spot accuracy.
    """
    P = np.asarray(points, float)
    if len(P) < 3:
        return "anywhere else on the work area"
    _, s, vt = np.linalg.svd(P - P.mean(0))
    if s[1] / np.sqrt(len(P)) >= MIN_SPREAD:
        return None
    if s[0] / np.sqrt(len(P)) < MIN_SPREAD:
        return "spread out: near and far, left and right"
    main = vt[0]  # the line the spots lie along; camera x is left/right on the desk
    return "nearer to you or further away" if abs(main[0]) > np.linalg.norm(main[1:]) else "further left or right"


def desk_from_hands(connect: str, out: str, spots: int = 6, still_s: float = 1.0, min_gap: float = 0.12,
                    max_spots: int = 12) -> np.ndarray:  # fmt: skip
    """Interactive: rest a flat hand on the desk at ``spots`` places; fit the desk plane.

    Keeps asking for spots (up to ``max_spots``) until they cover an area, not a line.
    """
    from collections import deque

    from .protocol import Subscriber

    sub = Subscriber(connect)
    hist: dict[str, deque] = {}
    touches: list[np.ndarray] = []
    print(f"Rest one hand flat on the desk, palm down, and hold it still for {still_s:.0f} s. Repeat at "
          f"{spots} places at least {min_gap * 100:.0f} cm apart: near and far, left and right, and the middle.")  # fmt: skip
    hint = "anywhere"
    try:
        while len(touches) < max_spots and (len(touches) < spots or hint is not None):
            msg, _ = sub.latest(200)
            if msg is None:
                continue
            now = msg["t_capture"]
            for h in msg["hands"]:
                p = np.asarray(h["openxr26_positions"], float)
                w = p[1]
                n = np.cross(p[7] - w, p[22] - w)
                n /= np.linalg.norm(n)
                dorsal = n if h["side"] == "left" else -n
                contact = p[PALM_JOINTS].mean(0) - PALM_THICKNESS * dorsal
                q = hist.setdefault(h["side"], deque())
                q.append((now, contact))
                while q and now - q[0][0] > still_s:
                    q.popleft()
                pts = np.array([c for _, c in q])
                if len(q) < 8 or now - q[0][0] < 0.8 * still_s or pts.std(0).max() > 0.004:
                    continue
                spot = pts.mean(0)
                if any(np.linalg.norm(spot - t) < min_gap for t in touches):
                    continue
                touches.append(spot)
                q.clear()
                hint = spread_hint(touches)
                more = f"; next: {hint}" if hint and len(touches) >= 2 else ""
                print(f"  spot {len(touches)} ({h['side']} hand) at {np.round(spot, 3)} m (camera frame){more}")
    except KeyboardInterrupt:
        pass
    finally:
        sub.close()
    if len(touches) < 3:
        raise SystemExit(f"only {len(touches)} spots; need at least 3")
    P = np.array(touches)
    T, rms = desk_frame_from_points(P)
    resid = (T[:3, :3] @ P.T + T[:3, 3:4])[2]
    tilt = math.degrees(math.asin(-T[2, 2])) if abs(T[2, 2]) <= 1 else float("nan")
    print(f"desk plane fit RMS {rms * 1000:.1f} mm over {len(touches)} spots (per spot: "
          f"{', '.join(f'{r * 1000:+.0f}' for r in resid)} mm) | camera {T[2, 3] * 1000:.0f} mm above "
          f"the desk, optical axis {tilt:.0f} deg below horizontal")  # fmt: skip
    if hint is not None:
        print("WARNING: the spots lie along a line, so the desk tilt is not determined; repeat and spread them out")
    if rms > 0.015:
        print("WARNING: spots do not lie on one plane well; repeat with the hand flat and still")
    np.savetxt(out, T, fmt="% .6f")
    print(f"wrote {out}; pass it to isaac/sharpa_hands.py --extrinsics or egohand-live run --extrinsics")
    return T


def board_pose(obj: np.ndarray, img: np.ndarray, K: np.ndarray, points: np.ndarray | None = None) -> np.ndarray:
    """Board points (``points``, default ``obj``) in the camera frame from one undistorted view."""
    import cv2

    ok, rvec, tvec = cv2.solvePnP(np.asarray(obj, np.float64).reshape(-1, 1, 3),
                                  np.asarray(img, np.float64).reshape(-1, 1, 2), np.asarray(K, np.float64),
                                  None, flags=cv2.SOLVEPNP_IPPE)  # fmt: skip
    if not ok:
        raise RuntimeError("solvePnP failed")
    R, _ = cv2.Rodrigues(rvec)
    return np.asarray(obj if points is None else points, np.float64) @ R.T + tvec.ravel()


def desk_from_board(source: str, out: str, board: str, intrinsics: str | None = None, rotate: int = 0,
                    board_height_mm: float = 0.0, frames: int = 30, width: int = 1280,
                    height: int = 720) -> np.ndarray:  # fmt: skip
    """Camera pose over the desk from a board lying flat on it (printed ChArUco, or a phone).

    ``board`` is a ``--board`` spec with the square size (see :mod:`egohand_live.targets`): the
    board's orientation needs no scale, its size sets the camera height. ``board_height_mm`` is
    the board surface above the desk (paper 0, a phone on its back ~8). Put the board in the
    middle of the work area: it becomes the workspace centre, as in ``calibrate-desk-hands``.
    """
    import cv2

    from .sources import open_source
    from .targets import auto_square, identify_charuco, parse_board

    target = parse_board(board)
    if target is not None and target.square is None:
        raise SystemExit("the board spec needs the square size, e.g. checker:9x6:8.4 or charuco:auto:27")
    src = open_source(source, width=width, height=height, intrinsics=intrinsics, rotate=rotate)
    win = "desk pose from board (q: cancel)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    poses, errs, K = [], [], None
    try:
        while len(poses) < frames:
            f = src.read()
            if f is None:
                continue
            K = np.asarray(f.K, float)
            gray = cv2.cvtColor(f.bgr, cv2.COLOR_BGR2GRAY)
            if target is None:
                target = identify_charuco(gray, auto_square(board))
                if target is not None:
                    print(f"board: {target.spec}")
            det = target.detect(gray) if target is not None else None
            vis = f.bgr.copy()
            if det is not None:
                P = board_pose(det.obj, det.img, K)
                uv = P @ K.T
                errs.append(float(np.sqrt(np.mean(np.sum((uv[:, :2] / uv[:, 2:] - det.img) ** 2, 1)))))
                poses.append(board_pose(det.obj, det.img, K, target.outline()))
                for u, v in det.img.astype(int):
                    cv2.circle(vis, (u, v), 4, (0, 255, 0), -1)
            text = f"board views {len(poses)}/{frames}" + ("" if det is not None else " - lay the board flat, in view")
            cv2.putText(vis, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(vis, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow(win, vis)
            if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                raise SystemExit("cancelled")
    finally:
        src.close()
        cv2.destroyAllWindows()
    P = np.median(np.stack(poses), 0)  # board outline in the camera frame
    spread = float(np.stack(poses).std(0).max())
    T, _ = desk_frame_from_points(P)
    T[2, 3] += board_height_mm / 1000  # board surface -> desk surface
    tilt = math.degrees(math.asin(-T[2, 2]))
    dist = float(np.linalg.norm(P.mean(0)))
    print(f"board {dist * 1000:.0f} mm from the camera (reprojection {np.median(errs):.2f} px, pose spread "
          f"{spread * 1000:.1f} mm) | camera {T[2, 3] * 1000:.0f} mm above the desk, optical axis {tilt:.0f} deg "
          f"below horizontal")  # fmt: skip
    if np.median(errs) > 1.5:
        print("WARNING: large reprojection error; check --intrinsics (lens calibration) and that the board is flat")
    np.savetxt(out, T, fmt="% .6f")
    print(f"wrote {out}; pass it to isaac/sharpa_hands.py --extrinsics or egohand-live run --extrinsics")
    return T


def depth_scale_from_hands(connect: str, extrinsics: str, spots: int = 6, still_px: float = 4.0,
                           still_s: float = 1.2) -> float:  # fmt: skip
    """Monocular depth scale from flat hands resting on a desk of known pose.

    The desk plane comes from ``extrinsics`` (``calibrate-desk-board``). Each rest spot gives the
    true distance of the palm along its viewing ray; the ratio to the tracked distance is the
    correction. Stillness is judged in the image, not in depth, whose noise is what is measured.
    Prints the value for ``run --depth-scale`` (accounting for a scale already applied).
    """
    from collections import deque

    from .protocol import Subscriber

    T = np.loadtxt(extrinsics).reshape(4, 4)
    nz, off = T[2, :3], T[2, 3]  # world z = nz . p + off; the desk is z = 0
    sub = Subscriber(connect)
    buf: dict[str, deque] = {"left": deque(), "right": deque()}
    last_px: dict[str, np.ndarray | None] = {"left": None, "right": None}
    ratios, applied = [], 1.0
    print(f"Rest a flat hand on the bare desk, palm down, still for {still_s + 0.8:.0f} s; lift; repeat at "
          f"{spots} spots spread over the work area (both hands, away from the image edges).")  # fmt: skip
    try:
        while len(ratios) < spots:
            msg, _ = sub.latest(200)
            if msg is None:
                continue
            applied = float(msg.get("depth_scale", 1.0))
            now = msg["t_capture"]
            w_img, h_img = msg["image_size"]
            for h in msg["hands"]:
                s = h["side"]
                p = np.asarray(h["openxr26_positions"], float)
                kp = np.asarray(h["keypoints_2d"], float)[[0, 9, 13, 17]]  # wrist + MCPs
                n = np.cross(p[7] - p[1], p[22] - p[1])
                n /= np.linalg.norm(n)
                contact = p[PALM_JOINTS].mean(0) - PALM_THICKNESS * (n if s == "left" else -n)
                q = buf[s]
                q.append((now, kp, contact))
                while q and now - q[0][0] > still_s:
                    q.popleft()
                if len(q) < 6 or now - q[0][0] < 0.8 * still_s:
                    continue
                K = np.array([x[1] for x in q])
                px = K.mean((0, 1))
                margin = 0.06 * np.array([w_img, h_img])
                if K.std(0).max() > still_px or (px < margin).any() or (px > [w_img, h_img] - margin).any():
                    continue
                if last_px[s] is not None and np.linalg.norm(px - last_px[s]) < 60:
                    continue
                c = np.array([x[2] for x in q]).mean(0)
                ray = c / np.linalg.norm(c)
                if nz @ ray >= 0:
                    continue  # ray does not hit the desk
                true, tracked = -off / (nz @ ray), float(np.linalg.norm(c))
                ratios.append(true / tracked)
                last_px[s] = px
                q.clear()
                print(f"  spot {len(ratios)} ({s} hand): tracked {tracked:.3f} m, desk {true:.3f} m, "
                      f"error {(tracked / true - 1) * 100:+.0f}%")  # fmt: skip
    except KeyboardInterrupt:
        pass
    finally:
        sub.close()
    if len(ratios) < 3:
        raise SystemExit(f"only {len(ratios)} spots; need at least 3")
    r = np.array(ratios)
    scale = applied * float(np.median(r))
    resid = (r / np.median(r) - 1) * 100
    print(f"depth error {(1 / r.mean() - 1) * 100:+.1f}% on average, spread {np.std(1 / r) / np.mean(1 / r) * 100:.1f}% "
          f"| use --depth-scale {scale:.3f} (residual spread {resid.std():.1f}%, worst {np.abs(resid).max():.1f}%)")  # fmt: skip
    if resid.std() > 6:
        print("WARNING: the error varies a lot between spots; a fixed scale will not fix it "
              "(camera too far from the hands?)")
    return scale
