"""Bilinear sampling, its adjoint, and measurement scatter."""

from __future__ import annotations

import numpy as np


def _weights(y, x):
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    y, x = np.broadcast_arrays(y, x)
    finite = np.isfinite(y) & np.isfinite(x)
    y = np.where(finite, y, -2.0)
    x = np.where(finite, x, -2.0)
    # Round-off from composed rotations must not invent direct CFA support at
    # another colour site. Use the same coordinates for sampling and adjoints.
    y = np.where(abs(y - np.rint(y)) < 1e-10, np.rint(y), y)
    x = np.where(abs(x - np.rint(x)) < 1e-10, np.rint(x), x)
    y0 = np.floor(y)
    x0 = np.floor(x)
    wy = y - y0
    wx = x - x0
    return y0, x0, wy, wx


def _gather(src: np.ndarray, yi: np.ndarray, xi: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = np.zeros(yi.shape + src.shape[2:], dtype=np.float64)
    if not np.any(valid):
        return out
    out[valid] = src[yi[valid], xi[valid]]
    return out


def bilinear_sample(src: np.ndarray, y, x, fill: float = 0.0) -> np.ndarray:
    """Sample ``src[y, x]`` with bilinear interpolation. Out-of-bounds is ``fill``."""
    src = np.asarray(src, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    y, x = np.broadcast_arrays(y, x)
    h, w = src.shape[:2]
    y0, x0, wy, wx = _weights(y, x)
    corners = (
        (y0, x0, (1.0 - wy) * (1.0 - wx)),
        (y0, x0 + 1.0, (1.0 - wy) * wx),
        (y0 + 1.0, x0, wy * (1.0 - wx)),
        (y0 + 1.0, x0 + 1.0, wy * wx),
    )
    acc = np.zeros(y.shape + src.shape[2:], dtype=np.float64)
    covered = np.zeros(y.shape, dtype=np.float64)
    for yc, xc, wt in corners:
        yi = np.rint(yc).astype(np.int64)
        xi = np.rint(xc).astype(np.int64)
        valid = (yi >= 0) & (yi < h) & (xi >= 0) & (xi < w)
        scale = wt[...,None] if src.ndim == 3 else wt
        acc = acc + scale * _gather(src, yi, xi, valid)
        covered = covered + wt * valid.astype(np.float64)
    if float(fill) != 0.0:
        missing = 1.0 - covered
        acc += (missing[...,None] if src.ndim == 3 else missing) * float(fill)
    return acc


def bilinear_sample_adjoint(dest: np.ndarray, y, x, src_shape: tuple[int, int]) -> np.ndarray:
    """Adjoint of :func:`bilinear_sample` with fill 0, mapping dest → src."""
    dest = np.asarray(dest, dtype=np.float64)
    if dest.ndim == 3:
        planes = [
            bilinear_sample_adjoint(dest[..., c], y, x, src_shape)
            for c in range(dest.shape[-1])
        ]
        return np.stack(planes, axis=-1)
    h, w = int(src_shape[0]), int(src_shape[1])
    y0, x0, wy, wx = _weights(y, x)
    out = np.zeros((h, w), dtype=np.float64)
    corners = (
        (y0, x0, (1.0 - wy) * (1.0 - wx)),
        (y0, x0 + 1.0, (1.0 - wy) * wx),
        (y0 + 1.0, x0, wy * (1.0 - wx)),
        (y0 + 1.0, x0 + 1.0, wy * wx),
    )
    for yc, xc, wt in corners:
        yi = np.rint(yc).astype(np.int64)
        xi = np.rint(xc).astype(np.int64)
        valid = (yi >= 0) & (yi < h) & (xi >= 0) & (xi < w)
        if not np.any(valid):
            continue
        np.add.at(out, (yi[valid], xi[valid]), (wt * dest)[valid])
    return out


def bilinear_push(
    src: np.ndarray,
    y_dest,
    x_dest,
    dest_shape: tuple[int, int],
    valid: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Scatter source samples onto a destination canvas with bilinear weights.

    This is the measurement backprojection (push) used by geometry-aware
    accumulation. ``valid`` masks samples that must not contribute.
    """
    src = np.asarray(src, dtype=np.float64)
    y_dest = np.asarray(y_dest, dtype=np.float64)
    x_dest = np.asarray(x_dest, dtype=np.float64)
    if valid is None:
        valid = np.ones(y_dest.shape, dtype=bool)
    else:
        valid = np.asarray(valid, dtype=bool)
    h, w = int(dest_shape[0]), int(dest_shape[1])
    accum = np.zeros((h, w) + src.shape[2:], dtype=np.float64)
    weight = np.zeros((h, w), dtype=np.float64)
    y0, x0, wy, wx = _weights(y_dest, x_dest)
    corners = (
        (y0, x0, (1.0 - wy) * (1.0 - wx)),
        (y0, x0 + 1.0, (1.0 - wy) * wx),
        (y0 + 1.0, x0, wy * (1.0 - wx)),
        (y0 + 1.0, x0 + 1.0, wy * wx),
    )
    src_flat = src
    for yc, xc, wt in corners:
        yi = np.rint(yc).astype(np.int64)
        xi = np.rint(xc).astype(np.int64)
        inb = valid & (yi >= 0) & (yi < h) & (xi >= 0) & (xi < w) & np.isfinite(wt)
        if not np.any(inb):
            continue
        contrib = ((wt[...,None] if src.ndim == 3 else wt) * src_flat)[inb]
        wcontrib = wt[inb]
        np.add.at(accum, (yi[inb], xi[inb]), contrib)
        np.add.at(weight, (yi[inb], xi[inb]), wcontrib)
    if src.ndim == 3:
        weight = np.broadcast_to(weight[...,None],accum.shape).copy()
    return accum, weight
