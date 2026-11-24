"""Calibration targets: a plain checkerboard or a ChArUco board, behind one interface.

Spec strings (``--board``)::

    checker:9x6[:SQUARE_MM]                      inner corners; size only needed for metric poses
    charuco:COLSxROWS:SQUARE_MM:MARKER_MM:DICT   squares across x down, e.g. charuco:11x8:27:20:5X5_100
    charuco:auto:SQUARE_MM                       identify layout and dictionary from the first views

A ChArUco board is found even when partly visible or partly outside the image, which is what a
wide-angle lens needs: the corners of the image get covered. Object points are in metres when the
square size is known, otherwise in squares (enough for lens calibration).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

log = logging.getLogger("egohand_live.targets")

ARUCO_FAMILIES = ("4X4", "5X5", "6X6", "7X7")
ARUCO_SIZES = (50, 100, 250, 1000)


@dataclass
class Detection:
    ids: np.ndarray  # (N,) corner ids, stable across views (for stillness checks)
    obj: np.ndarray  # (N, 3) float32 board points
    img: np.ndarray  # (N, 2) float32 pixels


class Checkerboard:
    def __init__(self, cols: int, rows: int, square: float | None = None):
        self.cols, self.rows, self.square = cols, rows, square
        s = square or 1.0
        self.points = np.array([[c * s, r * s, 0.0] for r in range(rows) for c in range(cols)], np.float32)
        self.min_corners = cols * rows

    @property
    def spec(self) -> str:
        return f"checker:{self.cols}x{self.rows}" + (f":{self.square * 1000:g}" if self.square else "")

    def detect(self, gray: np.ndarray) -> Detection | None:
        flags = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_ACCURACY
        found, corners = cv2.findChessboardCornersSB(gray, (self.cols, self.rows), flags)
        if not found:
            return None
        return Detection(np.arange(len(self.points)), self.points.copy(), corners.reshape(-1, 2).astype(np.float32))

    def outline(self) -> np.ndarray:
        """Board extent (4 corners, object units) for drawing coverage."""
        return self.points[[0, self.cols - 1, -1, -self.cols]]


class Charuco:
    def __init__(self, squares_x: int, squares_y: int, square: float, marker: float, dictionary: str,
                 legacy: bool = False):  # fmt: skip
        self.squares_x, self.squares_y, self.square, self.marker = squares_x, squares_y, square, marker
        self.dictionary, self.legacy = dictionary, legacy
        d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, f"DICT_{dictionary}"))
        self.board = cv2.aruco.CharucoBoard((squares_x, squares_y), square, marker, d)
        self.board.setLegacyPattern(legacy)
        params = cv2.aruco.CharucoParameters()
        params.minMarkers = 1
        self.detector = cv2.aruco.CharucoDetector(self.board, params)
        self.points = np.asarray(self.board.getChessboardCorners(), np.float32)
        self.min_corners = 8

    @property
    def spec(self) -> str:
        return (f"charuco:{self.squares_x}x{self.squares_y}:{self.square * 1000:g}:{self.marker * 1000:g}:"
                f"{self.dictionary}" + (":legacy" if self.legacy else ""))  # fmt: skip

    def detect(self, gray: np.ndarray) -> Detection | None:
        corners, ids, _, _ = self.detector.detectBoard(gray)
        if ids is None or len(ids) < self.min_corners:
            return None
        ids = ids.ravel()
        obj = self.points[ids]
        if np.linalg.matrix_rank(obj[:, :2] - obj[:, :2].mean(0), tol=1e-6) < 2:
            return None  # one row of corners: no constraint on the plane
        return Detection(ids, obj, corners.reshape(-1, 2).astype(np.float32))

    def outline(self) -> np.ndarray:
        x, y = (self.squares_x - 2) * self.square, (self.squares_y - 2) * self.square
        o = self.points[0]
        return np.array([o, o + [x, 0, 0], o + [x, y, 0], o + [0, y, 0]], np.float32)


def parse_board(spec: str):
    """Target from a ``--board`` spec; ``charuco:auto`` returns None (identify from images)."""
    kind, _, rest = spec.partition(":")
    parts = rest.split(":") if rest else []
    if kind == "checker":
        cols, rows = (int(v) for v in parts[0].lower().split("x"))
        return Checkerboard(cols, rows, float(parts[1]) / 1000 if len(parts) > 1 else None)
    if kind == "charuco":
        if parts and parts[0] == "auto":
            return None
        sx, sy = (int(v) for v in parts[0].lower().split("x"))
        legacy = len(parts) > 4 and parts[4] == "legacy"
        return Charuco(sx, sy, float(parts[1]) / 1000, float(parts[2]) / 1000, parts[3].upper(), legacy)
    raise ValueError(f"unknown board {spec!r}; expected checker:COLSxROWS or charuco:...")


def auto_square(spec: str) -> float | None:
    parts = spec.split(":")
    return float(parts[2]) / 1000 if len(parts) > 2 else None


def identify_charuco(gray: np.ndarray, square: float | None = None) -> Charuco | None:
    """Dictionary, layout and marker size of a ChArUco board seen whole (or nearly) in ``gray``."""
    best = None
    for fam in ARUCO_FAMILIES:
        d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, f"DICT_{fam}_1000"))
        corners, ids, _ = cv2.aruco.ArucoDetector(d).detectMarkers(gray)
        if ids is not None and (best is None or len(ids) > len(best[2])):
            best = (fam, corners, ids.ravel())
    if best is None or len(best[2]) < 4:
        return None
    fam, corners, ids = best
    size = next(s for s in ARUCO_SIZES if s > ids.max())
    centres = {i: c.reshape(4, 2).mean(0) for i, c in zip(ids, corners)}
    sides = [np.linalg.norm(np.diff(c.reshape(4, 2)[[0, 1, 2, 3, 0]], axis=0), axis=1).mean() for c in corners]
    # markers sit in every other square, so consecutive ids in a row are two squares apart
    gaps = [np.linalg.norm(centres[i + 1] - centres[i]) for i in centres if i + 1 in centres]
    ratio = float(np.median(sides) / (np.median(gaps) / 2)) if gaps else 0.7
    n = int(ids.max()) + 1
    sq = square or 1.0
    marker = round(ratio * sq * 2000) / 2000 if square else ratio  # metres, 0.5 mm steps
    cands = []
    for sx in range(3, 25):
        for sy in range(3, 25):
            if (sx * sy) // 2 < n or (sx * sy) // 2 > n + sx:  # allow an unseen last row of markers
                continue
            for legacy in (False, True):
                b = Charuco(sx, sy, sq, marker, f"{fam}_{size}", legacy)
                corners_, ids_, _, _ = b.detector.detectBoard(gray)
                if ids_ is None or len(ids_) < 6:
                    continue
                obj = b.points[ids_.ravel()][:, :2]
                img = corners_.reshape(-1, 2)
                H, _ = cv2.findHomography(obj, img, cv2.RANSAC, 5.0)
                if H is None:
                    continue
                proj = cv2.perspectiveTransform(obj.reshape(-1, 1, 2), H).reshape(-1, 2)
                cands.append((len(ids_), float(np.median(np.linalg.norm(proj - img, axis=1))), b))
    if not cands:
        return None
    good = [c for c in cands if c[1] < 3.0] or cands  # wrong layouts misplace corners
    # a taller/wider guess also "finds" corners along the board's outer edge: prefer the board
    # whose marker count matches the markers seen
    found = max(good, key=lambda c: ((c[2].squares_x * c[2].squares_y) // 2 == n, c[0], -c[1]))
    log.info("identified %s (%d corners, %.1f px)", found[2].spec, found[0], found[1])
    return found[2]
