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
    # A four-site lattice needs neither full detector coordinate grids nor
    # Python objects. Fixed-width labels also make per-channel masks vectorized
    # native comparisons instead of a Python string comparison at every pixel.
    tiles = BAYER_PATTERNS[pattern]
    labels = np.empty((height, width), dtype='U1')
    for y in range(2):
        for x in range(2):
            labels[y::2, x::2] = tiles[(y + oy) % 2][(x + ox) % 2]
    return labels


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


def nearest_debayer_preview(raw: np.ndarray, pattern: str, origin_xy=(0, 0), stride=1) -> np.ndarray:
    """Display-only RGB: copy the closest measured site for each colour.

    Sampling is evaluated on the original detector lattice before preview
    reduction. No averaging or interpolation; ties choose a deterministic site.
    Temporary storage scales with the preview, not a full-resolution RGB image.
    """
    raw = np.asarray(raw)
    if raw.ndim != 2 or min(raw.shape) < 2 or pattern not in BAYER_PATTERNS:
        raise ValueError('nearest Bayer display requires a 2-D mosaic of at least 2 by 2 and a known pattern')
    if type(stride) is not int or stride < 1:
        raise ValueError('preview stride must be a positive integer')
    h,w = raw.shape
    y,x = np.arange(0,h,stride),np.arange(0,w,stride)
    labels = cfa_labels(2,2,pattern,origin_xy)
    rgb = np.empty((len(y),len(x),3),dtype=raw.dtype)
    for channel,index in CHANNEL_INDEX.items():
        best_distance = np.full((len(y),len(x)),np.iinfo(np.int64).max,dtype=np.int64)
        for py,px in np.argwhere(labels==channel):
            yy = py + 2*np.clip((y-py+1)//2,0,(h-1-py)//2)
            xx = px + 2*np.clip((x-px+1)//2,0,(w-1-px)//2)
            distance = (y-yy)[:,None]**2 + (x-xx)[None,:]**2
            closer = distance < best_distance
            values = raw[yy[:,None],xx[None,:]]
            rgb[...,index][closer] = values[closer]
            best_distance[closer] = distance[closer]
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
