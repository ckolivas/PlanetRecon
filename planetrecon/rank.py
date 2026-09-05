"""Locked practical ranking and diagnostic rankings (R9 §7)."""

from __future__ import annotations

import hashlib
import json

import numpy as np
from scipy.ndimage import convolve

from planetrecon import constants as C


LAPLACIAN_KERNEL = np.array(C.RANK_LAPLACIAN_KERNEL, dtype=np.float64)


def ranking_config_hash() -> str:
    payload = {
        "kernel": [list(row) for row in C.RANK_LAPLACIAN_KERNEL],
        "border_px": C.RANK_BORDER_PX,
        "sky_subtraction": "median of known sky-mask pixels if any are present",
        "score": "mean squared Laplacian over valid interior pixels",
        "higher_is_better": True,
        "denoise": False,
        "presharpen": False,
        "p_grid": list(C.RANK_P_GRID),
        "decision_p": C.DECISION_P,
        "registration": "exact known subpixel Fourier shift before scoring",
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def laplacian_score(
    image: np.ndarray,
    sky_mask: np.ndarray | None = None,
    border: int = C.RANK_BORDER_PX,
) -> float:
    work = np.asarray(image, dtype=np.float64)
    if sky_mask is not None:
        sky = np.asarray(sky_mask, dtype=bool)
        if sky.any():
            work = work - float(np.median(work[sky]))
    lap = convolve(work, LAPLACIAN_KERNEL, mode="nearest")
    valid = np.ones(lap.shape, dtype=bool)
    if border > 0:
        valid[:border, :] = False
        valid[-border:, :] = False
        valid[:, :border] = False
        valid[:, -border:] = False
    return float(np.mean(lap[valid] ** 2))


def score_sequence(
    images: np.ndarray,
    sky_mask: np.ndarray | None = None,
    border: int = C.RANK_BORDER_PX,
) -> np.ndarray:
    return np.array(
        [laplacian_score(img, sky_mask=sky_mask, border=border) for img in images],
        dtype=np.float64,
    )


def top_fraction_indices(
    scores: np.ndarray,
    p: float,
    higher_is_better: bool = True,
) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    n = scores.size
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    if p >= 100:
        n_keep = n
    else:
        n_keep = max(1, int(round(p * n / 100.0)))
        n_keep = min(n_keep, n)
    order = np.argsort(scores, kind="stable")
    if higher_is_better:
        order = order[::-1]
    return np.sort(order[:n_keep].astype(np.int64))


def subsets_from_scores(scores: np.ndarray, p_grid=C.RANK_P_GRID) -> dict[int, np.ndarray]:
    return {int(p): top_fraction_indices(scores, p) for p in p_grid}


def diagnostic_scores(
    strehl: np.ndarray,
    phase_rms: np.ndarray,
    otf: np.ndarray,
    high_band: np.ndarray,
) -> dict[str, np.ndarray]:
    hmask = np.asarray(high_band, dtype=bool)
    mean_abs_h = np.array(
        [float(np.mean(np.abs(otf[k])[hmask])) for k in range(otf.shape[0])],
        dtype=np.float64,
    )
    return {
        "strehl_proxy": np.asarray(strehl, dtype=np.float64),
        "phase_rms_rad": np.asarray(phase_rms, dtype=np.float64),
        "highband_mean_abs_H": mean_abs_h,
    }
