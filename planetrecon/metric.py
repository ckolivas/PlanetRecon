"""Fourier high-band metric helpers used by simulator tests.

Prompt 2 owns estimator E_H. Prompt 1 uses the same window, MTF, and
frequency coordinates to judge simulator convergence.
"""

from __future__ import annotations

import numpy as np

from planetrecon import constants as C
from planetrecon.config import SimConfig
from planetrecon.object import tukey_2d
from planetrecon.optics import otf_from_centered_psf


def frequency_radii(cfg: SimConfig, n: int) -> np.ndarray:
    fx = np.fft.fftfreq(n, d=cfg.detector_pixel_scale_rad)
    fy = np.fft.fftfreq(n, d=cfg.detector_pixel_scale_rad)
    FX, FY = np.meshgrid(fx, fy, indexing="xy")
    return np.hypot(FX, FY)


def high_band_mask(cfg: SimConfig, mtf: np.ndarray) -> np.ndarray:
    rho = frequency_radii(cfg, mtf.shape[0]) / cfg.f_c
    return (
        (rho >= C.HIGH_BAND_RHO_MIN)
        & (rho <= 1.0)
        & (mtf >= C.HIGH_BAND_MTF_MIN)
    )


def ideal_mtf_from_dl_psf(psf_det: np.ndarray) -> np.ndarray:
    return np.abs(otf_from_centered_psf(psf_det)).astype(np.float64)


def relative_high_band(
    a: np.ndarray,
    b: np.ndarray,
    window: np.ndarray,
    mtf: np.ndarray,
    mask: np.ndarray,
) -> float:
    """E_H-style weighted high-band discrepancy of a relative to b."""
    fa = np.fft.fft2(window * a, norm="ortho")
    fb = np.fft.fft2(window * b, norm="ortho")
    w = mtf**2
    num = np.sum(mask * w * np.abs(fa - fb) ** 2)
    den = np.sum(mask * w * np.abs(fb) ** 2)
    if den <= 0:
        return float("inf")
    return float(num / den)


def high_band_power(
    img: np.ndarray, window: np.ndarray, mtf: np.ndarray, mask: np.ndarray
) -> float:
    f = np.fft.fft2(window * img, norm="ortho")
    return float(np.sum(mask * mtf**2 * np.abs(f) ** 2))


def eval_window(n: int = C.EVAL_SIZE) -> np.ndarray:
    return tukey_2d(n, C.TUKEY_ALPHA)
