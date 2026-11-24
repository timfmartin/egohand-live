"""Camera intrinsics from a checkerboard or ChArUco board, written as an EgoPost camera JSON.

Lens calibration does not depend on the board's physical size, so a checkerboard shown on a phone
works (``--show-pattern`` shows one full-screen here). A printed ChArUco board is better for wide
lenses: it is detected when partly outside the image, so the image corners, where distortion is
largest, get covered. Views are captured automatically when the board is still and seen from a
new position, size or angle.

Three lens models are fitted and the best kept (``--model auto``): OpenCV's standard
(k1, k2, p1, p2, k3) and rational (+ k4..k6) models, stored as EgoPost RATIONAL_POLYNOMIAL, and the
Kannala-Brandt fisheye model (EgoPost KB4) for wide-angle lenses. ``egohand-live run --intrinsics``
then undistorts frames on the fly.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np

from .targets import Checkerboard, Detection, auto_square, identify_charuco, parse_board

log = logging.getLogger("egohand_live.camcal")

MODELS = ("standard", "rational", "kb4")


def pattern_image(cols: int, rows: int, square_px: int = 110) -> np.ndarray:
    """Checkerboard with ``cols`` x ``rows`` inner corners and a white quiet zone."""
    m = square_px
    img = np.full(((rows + 1) * square_px + 2 * m, (cols + 1) * square_px + 2 * m), 255, np.uint8)
    for r in range(rows + 1):
        for c in range(cols + 1):
            if (r + c) % 2 == 0:
                img[m + r * square_px : m + (r + 1) * square_px, m + c * square_px : m + (c + 1) * square_px] = 0
    return img


def board_points(cols: int, rows: int) -> np.ndarray:
    return Checkerboard(cols, rows).points


def view_descriptor(det: Detection, size, extent: float) -> np.ndarray:
    """Where the board is, how big, and how it is tilted (for picking diverse views)."""
    w, h = size
    centre = det.img.mean(0) / [w, h]
    hull = cv2.convexHull(det.img.reshape(-1, 1, 2))
    scale = np.sqrt(cv2.contourArea(hull)) / np.hypot(w, h)
    H, _ = cv2.findHomography(det.obj[:, :2], det.img)
    tilt = H[2, :2] / H[2, 2] * extent if H is not None else np.zeros(2)  # perspective terms
    return np.array([*centre, 2 * scale, *np.clip(tilt, -1, 1)])


def _fit(views: list[Detection], size, model: str):
    obj = [v.obj.astype(np.float32) for v in views]
    img = [v.img.reshape(-1, 1, 2).astype(np.float32) for v in views]
    if model == "kb4":
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-9)
        flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_FIX_SKEW
        rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
            [o.reshape(-1, 1, 3).astype(np.float64) for o in obj], [i.astype(np.float64) for i in img],
            size, None, None, flags=flags, criteria=crit)  # fmt: skip
        project = lambda o, r, t: cv2.fisheye.projectPoints(o.reshape(-1, 1, 3).astype(np.float64), r, t, K, D)[0]
        dist = D.ravel()
    else:
        flags = cv2.CALIB_RATIONAL_MODEL if model == "rational" else 0
        rms, K, D, rvecs, tvecs = cv2.calibrateCamera(obj, img, size, None, None, flags=flags)
        project = lambda o, r, t: cv2.projectPoints(o, r, t, K, D)[0]
        dist = D.ravel()[: 8 if model == "rational" else 5]
    per_view = np.array([np.sqrt(np.mean(np.sum((project(o, r, t).reshape(-1, 2) - i.reshape(-1, 2)) ** 2, 1)))
                         for o, i, r, t in zip(obj, img, rvecs, tvecs)])  # fmt: skip
    return K, dist, float(rms), per_view


def calibrate(views: list[Detection], size, model: str = "standard"):
    """K, distortion, RMS and per-view RMS reprojection error (pixels) for one lens model."""
    return _fit(views, size, model)


def calibrate_best(views: list[Detection], size, model: str = "auto"):
    """Fit the requested model, or all of them and keep the best (simpler wins within 3 %)."""
    if model != "auto":
        return (model, *calibrate(views, size, model))
    fits = {}
    for m in MODELS:
        try:
            fits[m] = calibrate(views, size, m)
            log.info("%-8s RMS %.3f px", m, fits[m][2])
        except cv2.error as e:  # fisheye fit can fail on poorly spread views
            log.info("%-8s failed: %s", m, str(e).splitlines()[-1][:80])
    best = min(fits, key=lambda m: fits[m][2])
    for m in ("standard", "kb4"):  # fewer coefficients: prefer when nearly as good
        if m in fits and m != best and fits[m][2] <= 1.03 * fits[best][2] and best == "rational":
            best = m
    return (best, *fits[best])


def egopost_json(K: np.ndarray, dist: np.ndarray, size, model: str = "standard") -> dict:
    if model == "kb4":
        name, coeffs = "KB4", np.asarray(dist, float)[:4]
    else:
        name, coeffs = "RATIONAL_POLYNOMIAL", np.zeros(14)
        coeffs[: len(dist)] = dist
    return {
        "model": name,
        "image_width": int(size[0]),
        "image_height": int(size[1]),
        "camera_matrix": {"matrix_3x3": np.asarray(K, float).round(4).tolist()},
        "distortion_coefficients": [round(float(x), 8) for x in coeffs],
    }


def _text(img, text, org, scale=1.0):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)


def run(source: str, out: str, width=1280, height=720, fps=30, board: str = "checker:9x6", n_views=20,
        show_pattern=False, write_pattern: str | None = None, model: str = "auto") -> dict:  # fmt: skip
    from .sources import open_source

    target = parse_board(board)
    checker = isinstance(target, Checkerboard)
    pattern = pattern_image(target.cols, target.rows) if checker else None
    if write_pattern and checker:
        cv2.imwrite(write_pattern, pattern)
        print(f"wrote {write_pattern} ({target.cols}x{target.rows} inner corners); show it on a phone or print it")
    src = open_source(source, width=width, height=height, fps=fps)  # raw frames, no undistortion
    win, pwin = "camera calibration (c: capture, u: undo, q: finish)", "calibration pattern"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    if show_pattern and checker:
        cv2.namedWindow(pwin, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(pwin, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    views, descs, hulls = [], [], []
    prev, last_capture, last_identify, flash = None, 0.0, 0.0, 0.0
    size = None
    try:
        while True:
            f = src.read()
            if f is None:
                continue
            img = f.bgr
            size = (img.shape[1], img.shape[0])
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if target is None and time.time() - last_identify > 0.5:  # charuco:auto
                last_identify = time.time()
                target = identify_charuco(gray, auto_square(board))
                if target is not None:
                    print(f"board: {target.spec}  (pass --board {target.spec} next time)")
            det = target.detect(gray) if target is not None else None
            vis = img.copy()
            for q in hulls:
                cv2.polylines(vis, [q], True, (0, 200, 255), 2)
            capture, hint = False, "show the whole board to the camera" if target is None else "show the board"
            if det is not None:
                for u, v in det.img.astype(int):
                    cv2.circle(vis, (u, v), 4, (0, 255, 0), -1)
                common = None if prev is None else np.intersect1d(prev.ids, det.ids)
                still = False
                if common is not None and len(common) >= 8:
                    a = prev.img[np.searchsorted(prev.ids, common)] if checker else \
                        prev.img[[np.where(prev.ids == i)[0][0] for i in common]]  # fmt: skip
                    b = det.img[[np.where(det.ids == i)[0][0] for i in common]]
                    still = np.abs(a - b).mean() < 1.5
                extent = float(np.ptp(target.outline()[:, :2], axis=0).max())
                d = view_descriptor(det, size, extent)
                novel = not descs or min(np.linalg.norm(d - e) for e in descs) > 0.12
                hint = "hold still" if not still else ("new view!" if novel else "move or tilt the board")
                capture = still and novel and time.time() - last_capture > 0.8
            prev = det
            key = cv2.waitKey(1) & 0xFF
            if key == ord("c") and det is not None:
                capture = True
            if key == ord("u") and views:
                views.pop(), descs.pop(), hulls.pop()
            if key in (ord("q"), 27):
                break
            if capture:
                views.append(det)
                descs.append(view_descriptor(det, size, float(np.ptp(target.outline()[:, :2], axis=0).max())))
                hulls.append(cv2.convexHull(det.img.reshape(-1, 1, 2)).astype(np.int32))
                last_capture = flash = time.time()
                log.info("captured view %d/%d (%d corners)", len(views), n_views, len(det.ids))
                if len(views) >= n_views:
                    break
            _text(vis, f"views {len(views)}/{n_views} - {hint}", (20, 40))
            _text(vis, "vary position, distance and tilt; cover the image corners", (20, 78), 0.7)
            cv2.imshow(win, vis)
            if show_pattern and checker:
                p = cv2.cvtColor(pattern, cv2.COLOR_GRAY2BGR)
                if time.time() - flash < 0.4:
                    cv2.rectangle(p, (0, 0), (p.shape[1] - 1, p.shape[0] - 1), (0, 200, 0), 40)
                cv2.putText(p, f"views {len(views)}/{n_views}  {hint}", (30, p.shape[0] - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3, cv2.LINE_AA)  # fmt: skip
                cv2.imshow(pwin, p)
    finally:
        src.close()
        cv2.destroyAllWindows()
    if len(views) < 8:
        raise SystemExit(f"only {len(views)} views captured; need at least 8")
    name, K, dist, rms, per_view = calibrate_best(views, size, model)
    keep = per_view < max(3 * np.median(per_view), 0.5)
    if not keep.all():
        log.info("dropping %d outlier view(s) and recalibrating", int((~keep).sum()))
        views = [v for v, k in zip(views, keep) if k]
        name, K, dist, rms, per_view = calibrate_best(views, size, name)
    w, h = size
    hfov, vfov = (np.degrees(2 * np.arctan(w / (2 * K[0, 0]))), np.degrees(2 * np.arctan(h / (2 * K[1, 1]))))
    if name == "kb4":  # angle at the image edge from the fisheye mapping r = f * theta(1 + ...)
        th = np.linspace(0, np.pi / 2, 2000)
        r = th * (1 + sum(k * th ** (2 * i + 2) for i, k in enumerate(dist[:4])))
        hfov = 2 * np.degrees(np.interp(w / 2 / K[0, 0], r, th))
        vfov = 2 * np.degrees(np.interp(h / 2 / K[1, 1], r, th))
    print(f"{len(per_view)} views, lens model {name}, RMS reprojection error {rms:.2f} px")
    print(f"fx {K[0, 0]:.1f} fy {K[1, 1]:.1f} cx {K[0, 2]:.1f} cy {K[1, 2]:.1f} | FOV {hfov:.0f} x {vfov:.0f} deg "
          f"| distortion {np.array2string(np.asarray(dist), precision=4, suppress_small=True)}")  # fmt: skip
    cal = egopost_json(K, dist, size, name)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(cal, indent=2))
    print(f"wrote {out}; use it with: egohand-live run --source {source} --intrinsics {out}")
    return cal
