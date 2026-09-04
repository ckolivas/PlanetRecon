"""Pupil, PSF, detector binning, and diffraction-limited references."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import j1

from planetrecon.config import SimConfig


@dataclass
class Pupil:
    amplitude: np.ndarray
    x_m: np.ndarray
    y_m: np.ndarray
    frequency_x: np.ndarray
    frequency_y: np.ndarray
    mask: np.ndarray
    dx_m: float
    grid_size: int


def make_pupil(cfg: SimConfig) -> Pupil:
    n = cfg.pupil_grid_size
    dx = cfg.pupil_dx_m
    coords = (np.arange(n) - n // 2) * dx
    x, y = np.meshgrid(coords, coords, indexing="xy")
    r = np.hypot(x, y)
    outer = 0.5 * cfg.d_m
    inner = cfg.obstruction_ratio * outer
    amp = ((r <= outer) & (r >= inner)).astype(np.float64)
    freq_x = x / cfg.wavelength_m
    freq_y = y / cfg.wavelength_m
    return Pupil(
        amplitude=amp,
        x_m=x,
        y_m=y,
        frequency_x=freq_x,
        frequency_y=freq_y,
        mask=amp > 0.5,
        dx_m=dx,
        grid_size=n,
    )


def instantaneous_psf(amplitude: np.ndarray, phase: np.ndarray) -> np.ndarray:
    field = amplitude * np.exp(1j * phase)
    psi = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field)))
    psf = np.abs(psi) ** 2
    total = psf.sum()
    if total <= 0:
        raise RuntimeError("PSF has zero energy")
    return psf / total


def diffraction_limited_psf(pupil: Pupil) -> np.ndarray:
    return instantaneous_psf(pupil.amplitude, np.zeros_like(pupil.amplitude))


def center_crop(image: np.ndarray, n: int) -> np.ndarray:
    ny, nx = image.shape
    if ny < n or nx < n:
        raise ValueError(f"cannot crop {image.shape} to {n}x{n}")
    y0 = ny // 2 - n // 2
    x0 = nx // 2 - n // 2
    return image[y0 : y0 + n, x0 : x0 + n]


def bin_box(image: np.ndarray, factor: int) -> np.ndarray:
    if factor == 1:
        return image
    if image.shape[0] % factor or image.shape[1] % factor:
        raise ValueError(
            f"image shape {image.shape} is not divisible by bin factor {factor}"
        )
    ny, nx = image.shape
    return image.reshape(ny // factor, factor, nx // factor, factor).sum(axis=(1, 3))


def centroid_px(psf: np.ndarray) -> tuple[float, float]:
    total = psf.sum()
    ny, nx = psf.shape
    yy, xx = np.indices(psf.shape)
    cx = float((psf * xx).sum() / total)
    cy = float((psf * yy).sum() / total)
    return cx - (nx // 2), cy - (ny // 2)


def otf_from_centered_psf(psf: np.ndarray) -> np.ndarray:
    """OTF with DC at [0, 0] (numpy FFT layout)."""
    return np.fft.fft2(np.fft.ifftshift(psf))


def annular_airy_psf(cfg: SimConfig, n: int, pixel_scale_rad: float) -> np.ndarray:
    """Analytic obscured-aperture PSF on a centered grid, sum-normalised."""
    yy, xx = np.indices((n, n))
    x = (xx - n // 2) * pixel_scale_rad
    y = (yy - n // 2) * pixel_scale_rad
    theta = np.hypot(x, y)
    k = np.pi * cfg.d_m * theta / cfg.wavelength_m
    eps = cfg.obstruction_ratio
    with np.errstate(divide="ignore", invalid="ignore"):
        term = 2.0 * j1(k) / k - eps * 2.0 * j1(eps * k) / k
        term = np.where(k == 0.0, 1.0 - eps**2, term)
    i = (term / (1.0 - eps**2)) ** 2
    return i / i.sum()


def detector_pixel_mtf(cfg: SimConfig, n: int) -> np.ndarray:
    fx = np.fft.fftfreq(n, d=cfg.detector_pixel_scale_rad)
    fy = np.fft.fftfreq(n, d=cfg.detector_pixel_scale_rad)
    FX, FY = np.meshgrid(fx, fy, indexing="xy")
    sx = cfg.detector_pixel_scale_rad * FX
    sy = cfg.detector_pixel_scale_rad * FY
    mx = np.sinc(sx)
    my = np.sinc(sy)
    return mx * my
