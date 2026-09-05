"""Detector, sky and rotation conventions for W09 geometry."""

from __future__ import annotations

import numpy as np


def detector_xy_grids(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    """Pixel-centre coordinates: x column, y row, origin at the top-left corner."""
    yy, xx = np.indices((int(height), int(width)), dtype=np.float64)
    return xx + 0.5, yy + 0.5


def detector_to_sky(x, y, cx: float, cy: float):
    """Sky frame: +x right, +y up, origin at (cx, cy) in detector pixels."""
    return np.asarray(x, dtype=np.float64) - float(cx), float(cy) - np.asarray(y, dtype=np.float64)


def sky_to_detector(sx, sy, cx: float, cy: float):
    return np.asarray(sx, dtype=np.float64) + float(cx), float(cy) - np.asarray(sy, dtype=np.float64)


def rotate_sky(sx, sy, angle_rad: float):
    """Rotate sky coordinates counter-clockwise by ``angle_rad``."""
    ang = float(angle_rad)
    c = np.cos(ang)
    s = np.sin(ang)
    sx = np.asarray(sx, dtype=np.float64)
    sy = np.asarray(sy, dtype=np.float64)
    return c * sx - s * sy, s * sx + c * sy
