"""Kolmogorov phase screens and frozen-flow finite-exposure PSFs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import map_coordinates

from planetrecon import constants as C
from planetrecon.config import SimConfig
from planetrecon.optics import Pupil, instantaneous_psf


@dataclass
class PhaseScreen:
    phi: np.ndarray
    dx_m: float
    x_start_m: float
    y_centre_m: float
    subharmonic_levels: int
    r0_m: float


def kolmogorov_psd(fx: np.ndarray, fy: np.ndarray, r0: float) -> np.ndarray:
    f = np.hypot(fx, fy)
    psd = np.zeros_like(f, dtype=np.float64)
    nz = f > 0
    psd[nz] = 0.023 * r0 ** (-5.0 / 3.0) * f[nz] ** (-11.0 / 3.0)
    return psd


def _ft_screen(r0: float, nx: int, ny: int, dx: float, rng: np.random.Generator) -> np.ndarray:
    dfx = 1.0 / (nx * dx)
    dfy = 1.0 / (ny * dx)
    fx = (np.arange(nx) - nx // 2) * dfx
    fy = (np.arange(ny) - ny // 2) * dfy
    FX, FY = np.meshgrid(fx, fy, indexing="xy")
    psd = kolmogorov_psd(FX, FY, r0)
    cn = (
        rng.normal(size=(ny, nx)) + 1j * rng.normal(size=(ny, nx))
    ) * np.sqrt(psd * dfx * dfy)
    phs = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(cn))) * (nx * ny)
    return np.real(phs)


def _subharmonics(
    r0: float,
    nx: int,
    ny: int,
    dx: float,
    n_sub: int,
    rng: np.random.Generator,
) -> np.ndarray:
    lx = nx * dx
    ly = ny * dx
    x = (np.arange(nx) - nx / 2.0) * dx
    y = (np.arange(ny) - ny / 2.0) * dx
    X, Y = np.meshgrid(x, y, indexing="xy")
    phs_lo = np.zeros((ny, nx), dtype=np.complex128)
    for p in range(1, n_sub + 1):
        dfx = 1.0 / (3**p * lx)
        dfy = 1.0 / (3**p * ly)
        fx = np.arange(-1, 2) * dfx
        fy = np.arange(-1, 2) * dfy
        FX, FY = np.meshgrid(fx, fy, indexing="xy")
        psd = kolmogorov_psd(FX, FY, r0)
        psd[1, 1] = 0.0
        cn = (
            rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
        ) * np.sqrt(np.maximum(psd, 0.0) * dfx * dfy)
        for iy in range(3):
            for ix in range(3):
                phs_lo += cn[iy, ix] * np.exp(
                    1j * 2.0 * np.pi * (FX[iy, ix] * X + FY[iy, ix] * Y)
                )
    out = np.real(phs_lo)
    out -= out.mean()
    return out


def generate_square_screen(
    r0_m: float,
    n: int,
    dx: float,
    n_sub: int,
    rng: np.random.Generator,
) -> np.ndarray:
    phi = _ft_screen(C.R0_REF_M, n, n, dx, rng)
    phi = phi + _subharmonics(C.R0_REF_M, n, n, dx, n_sub, rng)
    phi -= phi.mean()
    return phi * (C.R0_REF_M / r0_m) ** (5.0 / 6.0)


def generate_screen(cfg: SimConfig, rng: np.random.Generator) -> PhaseScreen:
    """Generate a unit-r0 Kolmogorov screen and scale to cfg.r0_m.

    Paired regimes share the unit field because ``rng`` is seeded only from
    the integer seed, not from D/r0. Amplitude scales as (r0_ref/r0)^{5/6}.
    """
    nx, ny = cfg.screen_nx, cfg.screen_ny
    dx = cfg.screen_dx_m
    phi = _ft_screen(C.R0_REF_M, nx, ny, dx, rng)
    phi = phi + _subharmonics(
        C.R0_REF_M, nx, ny, dx, cfg.subharmonic_levels, rng
    )
    phi -= phi.mean()
    scale = (C.R0_REF_M / cfg.r0_m) ** (5.0 / 6.0)
    phi = phi * scale
    x_start = (nx - 1) * dx - 0.5 * cfg.d_m - cfg.interp_margin_m
    y_centre = 0.5 * (ny - 1) * dx
    return PhaseScreen(
        phi=phi,
        dx_m=dx,
        x_start_m=x_start,
        y_centre_m=y_centre,
        subharmonic_levels=cfg.subharmonic_levels,
        r0_m=cfg.r0_m,
    )


def sample_times(t0: float, texp: float, j: int) -> np.ndarray:
    if j < 1:
        raise ValueError("J must be >= 1")
    return t0 + (np.arange(j) + 0.5) * texp / j


def extraction_centres(screen: PhaseScreen, t: float, wind_m_s: float) -> tuple[float, float]:
    return screen.x_start_m - wind_m_s * t, screen.y_centre_m


def extract_phase(
    pupil: Pupil, screen: PhaseScreen, t: float, wind_m_s: float
) -> np.ndarray:
    xc, yc = extraction_centres(screen, t, wind_m_s)
    mask = pupil.mask
    x = xc + pupil.x_m[mask]
    y = yc + pupil.y_m[mask]
    ix = x / screen.dx_m
    iy = y / screen.dx_m
    ny, nx = screen.phi.shape
    if ix.min() < 0.0 or iy.min() < 0.0 or ix.max() > nx - 1 or iy.max() > ny - 1:
        raise RuntimeError(
            f"phase-screen wrap at t={t:.6e}s: "
            f"ix=[{ix.min():.3f},{ix.max():.3f}] iy=[{iy.min():.3f},{iy.max():.3f}] "
            f"shape=({ny},{nx})"
        )
    samples = map_coordinates(
        screen.phi, np.vstack([iy, ix]), order=1, mode="nearest", prefilter=False
    )
    phase = np.zeros_like(pupil.amplitude)
    phase[mask] = samples
    return phase


def finite_exposure_psf(
    pupil: Pupil,
    screen: PhaseScreen,
    t0: float,
    texp: float,
    j: int,
    wind_m_s: float,
) -> np.ndarray:
    acc = None
    for t in sample_times(t0, texp, j):
        phase = extract_phase(pupil, screen, t, wind_m_s)
        h = instantaneous_psf(pupil.amplitude, phase)
        acc = h if acc is None else acc + h
    return acc / float(j)


def no_wrap_ok(cfg: SimConfig, screen: PhaseScreen, n_frames: int | None = None) -> bool:
    n = cfg.n_frames if n_frames is None else n_frames
    t_end = (n - 1) * cfg.dt_s + cfg.texp_s
    xc, yc = extraction_centres(screen, t_end, cfg.wind_m_s)
    half = 0.5 * cfg.d_m + cfg.interp_margin_m
    ny, nx = screen.phi.shape
    xmax = (nx - 1) * screen.dx_m
    ymax = (ny - 1) * screen.dx_m
    return (
        xc - half >= 0.0
        and xc + half <= xmax
        and yc - half >= 0.0
        and yc + half <= ymax
    )


def structure_function(
    phi: np.ndarray, dx: float, rho: np.ndarray
) -> np.ndarray:
    """Non-wrapping lag structure function along x and y, interpolated to rho."""
    ny, nx = phi.shape
    lags = np.arange(1, min(nx, ny) // 4)
    measured = []
    rhos = []
    for lag in lags:
        dx2 = (phi[:, lag:] - phi[:, :-lag]) ** 2
        dy2 = (phi[lag:, :] - phi[:-lag, :]) ** 2
        sf = 0.5 * (dx2.mean() + dy2.mean())
        measured.append(sf)
        rhos.append(lag * dx)
    rhos = np.asarray(rhos)
    measured = np.asarray(measured)
    return np.interp(rho, rhos, measured)


def kolmogorov_structure_function(rho: np.ndarray, r0: float) -> np.ndarray:
    return 6.88 * (rho / r0) ** (5.0 / 3.0)
