"""First-60 inference modes on the obstructed pupil.

Modes are Noll Zernikes QR-orthonormalised on the annular mask, excluding
piston. They are used only for residual diagnostics; the Kolmogorov screen
is not generated in this basis.
"""

from __future__ import annotations

import math

import numpy as np

from planetrecon import constants as C
from planetrecon.optics import Pupil


def noll_nm(j: int) -> tuple[int, int]:
    """Noll (1976) index j >= 1 -> radial order n, azimuthal order m."""
    if j < 1:
        raise ValueError("Noll indices start at 1")
    n = 0
    count = 0
    while True:
        n_in_order = n + 1
        if count + n_in_order >= j:
            break
        count += n_in_order
        n += 1
    p = j - count
    idx = 1
    for am in range(n % 2, n + 1, 2):
        if am == 0:
            if idx == p:
                return n, 0
            idx += 1
            continue
        start_j = count + idx
        pair = (-am, am) if (start_j % 2 == 1) else (am, -am)
        for m in pair:
            if idx == p:
                return n, m
            idx += 1
    raise RuntimeError(f"noll_nm failed for j={j}")


def _radial(n: int, m: int, rho: np.ndarray) -> np.ndarray:
    m = abs(m)
    r = np.zeros_like(rho)
    for k in range((n - m) // 2 + 1):
        num = math.factorial(n - k)
        den = (
            math.factorial(k)
            * math.factorial((n + m) // 2 - k)
            * math.factorial((n - m) // 2 - k)
        )
        r += ((-1) ** k) * (num / den) * rho ** (n - 2 * k)
    return r


def zernike_noll(j: int, rho: np.ndarray, theta: np.ndarray) -> np.ndarray:
    n, m = noll_nm(j)
    r = _radial(n, m, rho)
    if m == 0:
        return np.sqrt(n + 1) * r
    norm = np.sqrt(2 * (n + 1))
    if m > 0:
        return norm * r * np.cos(m * theta)
    return norm * r * np.sin(-m * theta)


class KLBasis:
    def __init__(self, pupil: Pupil, n_modes: int = C.KL_MODES):
        mask = pupil.mask
        x = pupil.x_m[mask]
        y = pupil.y_m[mask]
        rho = np.hypot(x, y) / (0.5 * C.D_M)
        theta = np.arctan2(y, x)
        n_build = n_modes + 12
        cols = [zernike_noll(j, rho, theta) for j in range(1, n_build + 1)]
        z = np.column_stack(cols)
        q, _ = np.linalg.qr(z, mode="reduced")
        self.modes = q[:, 1 : n_modes + 1]
        self.mask = mask
        self.n_modes = n_modes
        self.shape = mask.shape

    def expand(self, coeff: np.ndarray) -> np.ndarray:
        """Synthesize a 2-D piston-removed phase from the first len(coeff) modes."""
        a = np.asarray(coeff, dtype=np.float64).reshape(-1)
        if a.size > self.n_modes:
            raise ValueError(f"need {a.size} modes, basis has {self.n_modes}")
        phase = np.zeros(self.shape, dtype=np.float64)
        if a.size:
            phase[self.mask] = self.modes[:, : a.size] @ a
        return phase

    def project(self, phase: np.ndarray) -> tuple[np.ndarray, float]:
        v = phase[self.mask]
        v = v - v.mean()
        coeff = self.modes.T @ v
        recon = self.modes @ coeff
        resid = v - recon
        rms = float(np.sqrt(np.mean(resid**2)))
        return coeff.astype(np.float64), rms
