"""Gate-1 known-transfer estimators: E1, E2a0, E2a, A1o."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import fourier_shift
from scipy.sparse.linalg import LinearOperator, cg

from planetrecon import constants as C
from planetrecon.config import SimConfig
from planetrecon.metric import frequency_radii


def _fourier_shift_array(freq: np.ndarray, shift_xy) -> np.ndarray:
    """Shift a DC-at-[0,0] Fourier array by ``(sx, sy)`` detector pixels."""
    sx, sy = float(shift_xy[0]), float(shift_xy[1])
    return np.asarray(fourier_shift(freq, shift=(sy, sx)), dtype=np.complex128)


def fourier_shift_image(image: np.ndarray, shift_xy) -> np.ndarray:
    return np.fft.ifft2(_fourier_shift_array(np.fft.fft2(image), shift_xy)).real


def shift_otf(otf: np.ndarray, shift_xy) -> np.ndarray:
    return _fourier_shift_array(otf, shift_xy)


def register_images(images: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    out = np.empty_like(images, dtype=np.float64)
    for k in range(images.shape[0]):
        out[k] = fourier_shift_image(images[k], (-shifts[k, 0], -shifts[k, 1]))
    return out


def register_otfs(otfs: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    out = np.empty_like(otfs, dtype=np.complex128)
    for k in range(otfs.shape[0]):
        out[k] = shift_otf(otfs[k], (-shifts[k, 0], -shifts[k, 1]))
    return out


def lambda_field(
    cfg: SimConfig,
    n: int,
    lam_rel: float,
    lam_grad: float = 0.0,
) -> np.ndarray:
    rho = frequency_radii(cfg, n) / cfg.f_c
    return np.asarray(lam_rel + lam_grad * rho**2, dtype=np.float64)


def frame_noise_variance(expected: np.ndarray, read_rms: float = C.READ_NOISE_E) -> np.ndarray:
    """Spatially averaged Poisson+read variance per frame (e^-²)."""
    means = np.mean(np.clip(expected, 0.0, None), axis=(-2, -1))
    return np.asarray(means + read_rms**2, dtype=np.float64)


def wiener_num_den(
    otfs: np.ndarray,
    images: np.ndarray,
    sigma2: np.ndarray,
    lam_f: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    ifs = np.fft.fft2(images, axes=(-2, -1))
    w = 1.0 / np.clip(sigma2.astype(np.float64), C.DEN_FLOOR, None)
    w = w[:, None, None]
    num = np.sum(np.conj(otfs) * ifs * w, axis=0)
    den = np.sum(np.abs(otfs) ** 2 * w, axis=0) + lam_f
    den = np.maximum(den, C.DEN_FLOOR)
    return num, den


def e1(
    otfs: np.ndarray,
    images: np.ndarray,
    sigma2: np.ndarray,
    lam_f: np.ndarray,
) -> np.ndarray:
    """Closed-form multi-frame quadratic/Wiener estimator (R9 §8.1)."""
    num, den = wiener_num_den(otfs, images, sigma2, lam_f)
    return np.fft.ifft2(num / den).real


def _normal_matvec(vec, shape, otfs, sigma2, lam_f):
    o = vec.reshape(shape)
    of = np.fft.fft2(o)
    w = 1.0 / np.clip(sigma2.astype(np.float64), C.DEN_FLOOR, None)
    w = w[:, None, None]
    den = np.sum(np.abs(otfs) ** 2 * w, axis=0) + lam_f
    return np.fft.ifft2(den * of).real.ravel()


def e2a0(
    otfs: np.ndarray,
    images: np.ndarray,
    sigma2: np.ndarray,
    lam_f: np.ndarray,
    tol: float = 1e-10,
    maxiter: int = C.E2A0_CG_MAXITER,
    x0: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """Conjugate-gradient numerical realisation of the E1 objective (§8.2)."""
    num, _den = wiener_num_den(otfs, images, sigma2, lam_f)
    b = np.fft.ifft2(num).real.ravel()
    shape = images.shape[-2:]
    n = int(np.prod(shape))

    def matvec(v):
        return _normal_matvec(v, shape, otfs, sigma2, lam_f)

    A = LinearOperator((n, n), matvec=matvec, dtype=np.float64)
    x0v = None if x0 is None else np.asarray(x0, dtype=np.float64).ravel()
    x, info = cg(A, b, rtol=tol, maxiter=maxiter, x0=x0v)
    recon = np.asarray(x, dtype=np.float64).reshape(shape)
    return recon, {"cg_info": int(info), "n_iter_cap": int(maxiter)}


def _project_e2a(image: np.ndarray, support: np.ndarray) -> np.ndarray:
    o = np.maximum(np.asarray(image, dtype=np.float64), 0.0)
    of = np.fft.fft2(o) * support
    o = np.fft.ifft2(of).real
    return np.maximum(o, 0.0)


def e2a(
    otfs: np.ndarray,
    images: np.ndarray,
    sigma2: np.ndarray,
    lam_f: np.ndarray,
    support: np.ndarray,
    maxiter: int = C.E2A_FISTA_MAXITER,
    tol: float = C.E2A_FISTA_TOL,
) -> tuple[np.ndarray, dict]:
    """Positivity + spectral-support reconstruction of the E1 quadratic (§8.3)."""
    num, den = wiener_num_den(otfs, images, sigma2, lam_f)
    support = np.asarray(support, dtype=np.float64)
    den = np.where(support > 0.5, den, 1.0)
    num = num * support
    o = _project_e2a(np.fft.ifft2(num / den).real, support)
    lip = float(np.max(den * support))
    step = 1.0 / max(lip, C.DEN_FLOOR)
    y = o.copy()
    t = 1.0
    last_delta = 0.0
    n_iter = 0
    for n_iter in range(1, maxiter + 1):
        of = np.fft.fft2(y)
        grad = np.fft.ifft2((den * of - num) * support).real
        o_next = _project_e2a(y - step * grad, support)
        t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        y = o_next + ((t - 1.0) / t_next) * (o_next - o)
        last_delta = float(np.linalg.norm(o_next - o) / max(np.linalg.norm(o_next), 1e-12))
        o = o_next
        t = t_next
        if last_delta < tol:
            break
    return o, {
        "n_iter": n_iter,
        "rel_delta": last_delta,
        "step": step,
        "converged": bool(last_delta < tol),
    }


def a1o(
    otfs: np.ndarray,
    images: np.ndarray,
    sigma2: np.ndarray,
    lam_f: np.ndarray,
    support: np.ndarray,
    shifts: np.ndarray | None = None,
    maxiter: int = C.E2A_FISTA_MAXITER,
    tol: float = C.E2A_FISTA_TOL,
) -> tuple[np.ndarray, dict]:
    """Registered uniform stack reconstructed with the true mean transfer (§8.6).

    If ``shifts`` is None the arrays are assumed already registered.
    """
    if shifts is not None:
        images = register_images(images, shifts)
        otfs = register_otfs(otfs, shifts)
    n_sel = images.shape[0]
    stack = np.mean(images, axis=0, dtype=np.float64)
    h_eff = np.mean(otfs, axis=0)
    sigma2_stack = float(np.mean(sigma2) / max(n_sel, 1))
    recon, info = e2a(
        h_eff[None, ...],
        stack[None, ...],
        np.array([sigma2_stack]),
        lam_f,
        support,
        maxiter=maxiter,
        tol=tol,
    )
    info = dict(info)
    info["H_eff"] = h_eff
    info["n_selected"] = int(n_sel)
    info["sigma2_stack"] = sigma2_stack
    return recon, info


@dataclass(frozen=True)
class Regularisation:
    e1_lambda_rel: float = C.E1_LAMBDA_REL
    e2a_lambda_rel: float = C.E2A_LAMBDA_REL
    e1_lambda_grad: float = C.E1_LAMBDA_GRAD

    def field_e1(self, cfg: SimConfig, n: int) -> np.ndarray:
        return lambda_field(cfg, n, self.e1_lambda_rel, self.e1_lambda_grad)

    def field_e2a(self, cfg: SimConfig, n: int) -> np.ndarray:
        return lambda_field(cfg, n, self.e2a_lambda_rel, self.e1_lambda_grad)
