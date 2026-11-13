"""Causal temporal filters for live hand tracking.

EgoPost's adaptive smoother needs the whole clip (it filters in both temporal
directions), so live use needs a causal replacement. The One Euro filter
(Casiez et al., CHI 2012) trades jitter at rest against lag during fast motion.
"""

import math
from dataclasses import dataclass

import numpy as np


def _alpha(cutoff_hz, dt):
    tau = 1.0 / (2.0 * math.pi * np.asarray(cutoff_hz))
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    """One Euro filter over arrays of points shaped (..., D).

    The adaptive cutoff uses the speed magnitude of each point (norm over the
    last axis), so all coordinates of a point share one cutoff.

    Args:
        min_cutoff: Cutoff at rest, Hz. Lower = smoother but laggier when slow.
        beta: Cutoff increase per unit of speed (units/s). Higher = less lag when fast.
        d_cutoff: Cutoff for the derivative estimate, Hz.
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.reset()

    def reset(self):
        self._x = None
        self._dx = None
        self._t = None

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if self._x is None:
            self._x, self._dx, self._t = x.copy(), np.zeros_like(x), t
            return x.copy()
        dt = max(t - self._t, 1e-4)
        dx = (x - self._x) / dt
        self._dx = self._dx + _alpha(self.d_cutoff, dt) * (dx - self._dx)
        speed = np.linalg.norm(self._dx, axis=-1, keepdims=True)
        a = _alpha(self.min_cutoff + self.beta * speed, dt)
        self._x = self._x + a * (x - self._x)
        self._t = t
        return self._x.copy()


@dataclass
class SmoothingConfig:
    """Filter settings for a hand. Monocular depth is noisier than image-plane
    position, so the wrist's camera-Z gets its own, lower cutoff."""

    wrist_xy_min_cutoff: float = 1.5
    wrist_xy_beta: float = 4.0
    wrist_z_min_cutoff: float = 0.8
    wrist_z_beta: float = 2.0
    fingers_min_cutoff: float = 2.0
    fingers_beta: float = 15.0
    reset_after_s: float = 0.3
    enabled: bool = True


class HandSmoother:
    """Smooth one hand's camera-frame joints as wrist translation + wrist-relative shape."""

    def __init__(self, cfg: SmoothingConfig | None = None):
        self.cfg = cfg or SmoothingConfig()
        c = self.cfg
        self._xy = OneEuroFilter(c.wrist_xy_min_cutoff, c.wrist_xy_beta)
        self._z = OneEuroFilter(c.wrist_z_min_cutoff, c.wrist_z_beta)
        self._local = OneEuroFilter(c.fingers_min_cutoff, c.fingers_beta)
        self._last_t = None

    def reset(self):
        for f in (self._xy, self._z, self._local):
            f.reset()
        self._last_t = None

    def __call__(self, joints_cam: np.ndarray, t: float) -> np.ndarray:
        """Args: joints_cam (25, 3) meters, OpenCV camera frame. Returns filtered copy."""
        if not self.cfg.enabled:
            return np.asarray(joints_cam, dtype=np.float64).copy()
        if self._last_t is not None and t - self._last_t > self.cfg.reset_after_s:
            self.reset()
        self._last_t = t
        joints = np.asarray(joints_cam, dtype=np.float64)
        wrist = joints[0]
        local = joints - wrist
        wrist_f = np.empty(3)
        wrist_f[:2] = self._xy(wrist[None, :2], t)[0]
        wrist_f[2] = self._z(wrist[None, 2:], t)[0, 0]
        return self._local(local, t) + wrist_f
