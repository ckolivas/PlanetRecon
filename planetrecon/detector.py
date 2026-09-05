"""CFA sampling, green proxy, and joint RGB accumulation adjoints."""

from __future__ import annotations

import numpy as np

from scipy.ndimage import shift as ndshift


BAYER_PATTERNS = {
    "RGGB": (("R", "G"), ("G", "B")),
    "GRBG": (("G", "R"), ("B", "G")),
    "GBRG": (("G", "B"), ("R", "G")),
    "BGGR": (("B", "G"), ("G", "R")),
}
CHANNEL_INDEX = {"R": 0, "G": 1, "B": 2}


def is_bayer(color_mode: str) -> bool:
    return color_mode in BAYER_PATTERNS


def cfa_labels(height: int, width: int, pattern: str, origin_xy=(0, 0)) -> np.ndarray:
    if pattern not in BAYER_PATTERNS:
        raise ValueError(f"unknown Bayer pattern {pattern!r}")
    ox, oy = int(origin_xy[0]) % 2, int(origin_xy[1]) % 2
    tiles = np.array(BAYER_PATTERNS[pattern], dtype=object)
    yy, xx = np.indices((height, width))
    return tiles[(yy + oy) % 2, (xx + ox) % 2]


def channel_mask(labels: np.ndarray, channel: str) -> np.ndarray:
    return labels == channel


def extract_green_proxy(raw: np.ndarray, pattern: str, origin_xy=(0, 0)) -> np.ndarray:
    """Full-res green field: measured G sites, neighbour mean elsewhere."""
    raw = np.asarray(raw, dtype=np.float64)
    labels = cfa_labels(raw.shape[0], raw.shape[1], pattern, origin_xy)
    gmask = channel_mask(labels, "G")
    out = np.zeros_like(raw, dtype=np.float64)
    out[gmask] = raw[gmask]
    filled = out.copy()
    # 4-neighbour fill for missing green
    acc = np.zeros_like(out)
    w = np.zeros_like(out)
    acc[:-1] += out[1:]
    w[:-1] += gmask[1:]
    acc[1:] += out[:-1]
    w[1:] += gmask[:-1]
    acc[:, :-1] += out[:, 1:]
    w[:, :-1] += gmask[:, 1:]
    acc[:, 1:] += out[:, :-1]
    w[:, 1:] += gmask[:, :-1]
    missing = ~gmask
    filled[missing] = acc[missing] / np.clip(w[missing], 1.0, None)
    return filled


def bilinear_demosaic(raw: np.ndarray, pattern: str, origin_xy=(0, 0)) -> np.ndarray:
    """Conventional demosaic-first comparison path. Labelled, not the joint solve."""
    raw = np.asarray(raw, dtype=np.float64)
    labels = cfa_labels(raw.shape[0], raw.shape[1], pattern, origin_xy)
    rgb = np.zeros(raw.shape + (3,), dtype=np.float64)
    for name, idx in CHANNEL_INDEX.items():
        mask = channel_mask(labels, name)
        plane = np.zeros_like(raw)
        plane[mask] = raw[mask]
        acc = np.zeros_like(raw)
        w = np.zeros_like(raw)
        acc[:-1] += plane[1:]
        w[:-1] += mask[1:]
        acc[1:] += plane[:-1]
        w[1:] += mask[:-1]
        acc[:, :-1] += plane[:, 1:]
        w[:, :-1] += mask[:, 1:]
        acc[:, 1:] += plane[:, :-1]
        w[:, 1:] += mask[:, :-1]
        acc[:-1, :-1] += plane[1:, 1:]
        w[:-1, :-1] += mask[1:, 1:]
        acc[1:, 1:] += plane[:-1, :-1]
        w[1:, 1:] += mask[:-1, :-1]
        acc[:-1, 1:] += plane[1:, :-1]
        w[:-1, 1:] += mask[1:, :-1]
        acc[1:, :-1] += plane[:-1, 1:]
        w[1:, :-1] += mask[:-1, 1:]
        filled = plane.copy()
        missing = ~mask
        filled[missing] = acc[missing] / np.clip(w[missing], 1.0, None)
        rgb[..., idx] = filled
    return rgb


def cfa_accumulate(
    raw: np.ndarray,
    shift_xy: tuple[float, float],
    pattern: str,
    origin_xy=(0, 0),
) -> tuple[np.ndarray, np.ndarray]:
    """Scatter each CFA sample onto its colour plane after the inverse shift.

    Green G1 and G2 update the same green radiance field.
    """
    raw = np.asarray(raw, dtype=np.float64)
    h, w = raw.shape
    labels = cfa_labels(h, w, pattern, origin_xy)
    rgb = np.zeros((h, w, 3), dtype=np.float64)
    weight = np.zeros((h, w, 3), dtype=np.float64)
    sy, sx = float(shift_xy[1]), float(shift_xy[0])
    ones = np.ones((h, w), dtype=np.float64)
    for name, idx in CHANNEL_INDEX.items():
        mask = channel_mask(labels, name).astype(np.float64)
        plane = raw * mask
        rgb[..., idx] = ndshift(plane, shift=(-sy, -sx), order=1, prefilter=False, mode="grid-constant")
        weight[..., idx] = ndshift(mask * ones, shift=(-sy, -sx), order=1, prefilter=False, mode="grid-constant")
    return rgb, weight
