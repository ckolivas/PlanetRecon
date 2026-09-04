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


def mid_band_mask(cfg: SimConfig, mtf: np.ndarray) -> np.ndarray:
    rho = frequency_radii(cfg, mtf.shape[0]) / cfg.f_c
    return (
        (rho >= C.MID_BAND_RHO_MIN)
        & (rho <= C.MID_BAND_RHO_MAX)
        & (mtf >= C.HIGH_BAND_MTF_MIN)
    )


def support_mask(cfg: SimConfig, n: int) -> np.ndarray:
    rho = frequency_radii(cfg, n) / cfg.f_c
    return rho <= C.SUPPORT_RHO_MAX


def high_band_truth_fraction(
    img: np.ndarray, window: np.ndarray, mtf: np.ndarray, mask: np.ndarray
) -> float:
    """R_H: conditioned high-band truth power over total windowed power."""
    f = np.fft.fft2(window * img, norm="ortho")
    total = float(np.sum(np.abs(f) ** 2))
    if total <= 0:
        return 0.0
    return float(np.sum(mask * mtf**2 * np.abs(f) ** 2) / total)


def eh_metric(
    o_hat: np.ndarray,
    o_true: np.ndarray,
    window: np.ndarray,
    mtf: np.ndarray,
    mask: np.ndarray,
) -> float:
    return relative_high_band(o_hat, o_true, window, mtf, mask)


def image_rel_mse(o_hat: np.ndarray, o_true: np.ndarray) -> float:
    den = float(np.mean(np.asarray(o_true, dtype=np.float64) ** 2))
    if den <= 0:
        return float("inf")
    err = np.asarray(o_hat, dtype=np.float64) - np.asarray(o_true, dtype=np.float64)
    return float(np.mean(err**2) / den)


def signed_contrast(img: np.ndarray, aperture: np.ndarray, annulus: np.ndarray) -> float:
    ap = np.asarray(aperture, dtype=bool)
    an = np.asarray(annulus, dtype=bool)
    if not ap.any() or not an.any():
        return float("nan")
    bg = float(np.mean(img[an]))
    fg = float(np.mean(img[ap]))
    if abs(bg) < 1e-15:
        return float("nan")
    return (fg - bg) / bg


def contrast_relative_error(c_hat: float, c_true: float) -> float:
    if not np.isfinite(c_hat) or not np.isfinite(c_true) or abs(c_true) < 1e-15:
        return float("nan")
    return float((c_hat - c_true) / abs(c_true))


def oval_in_metric_region(aperture: np.ndarray, window: np.ndarray) -> bool:
    ap = np.asarray(aperture, dtype=bool)
    if not ap.any():
        return False
    return bool(np.min(window[ap]) >= 1.0 - 1e-12)
