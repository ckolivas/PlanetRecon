"""Gate-1 known-transfer estimators: E1, E2a0, E2a, E2b, A1o."""

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


def _apply_spectral_support(image: np.ndarray, support: np.ndarray) -> np.ndarray:
    of = np.fft.fft2(np.asarray(image, dtype=np.float64)) * support
    return np.fft.ifft2(of).real


def positivity_violation(image: np.ndarray) -> float:
    return float(max(0.0, -np.min(np.asarray(image, dtype=np.float64))))


def out_of_support_fraction(image: np.ndarray, support: np.ndarray) -> float:
    """Out-of-support Fourier norm / total Fourier norm."""
    freq = np.fft.fft2(np.asarray(image, dtype=np.float64))
    total = float(np.linalg.norm(freq))
    if total <= 0.0:
        return 0.0
    leaked = freq[np.asarray(support) < 0.5]
    return float(np.linalg.norm(leaked) / total)


def project_positivity_support(
    image: np.ndarray,
    support: np.ndarray,
    maxiter: int = C.DYKSTRA_MAXITER,
    tol: float = C.DYKSTRA_TOL,
    p: np.ndarray | None = None,
    q: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """Euclidean projection onto {x ≥ 0} ∩ bandlimited(support) via Dykstra.

    Sequential clip-then-support-then-clip is not a projection onto the
    intersection: the final positivity clip restores out-of-support power.
    """
    x = np.asarray(image, dtype=np.float64).copy()
    support = np.asarray(support, dtype=np.float64)
    if p is None:
        p = np.zeros_like(x)
    else:
        p = np.asarray(p, dtype=np.float64).copy()
    if q is None:
        q = np.zeros_like(x)
    else:
        q = np.asarray(q, dtype=np.float64).copy()
    last = 0.0
    n_iter = 0
    pos = positivity_violation(x)
    leak = out_of_support_fraction(x, support)
    for n_iter in range(1, int(maxiter) + 1):
        y = np.maximum(x + p, 0.0)
        p = x + p - y
        z = _apply_spectral_support(y + q, support)
        q = y + q - z
        denom = max(float(np.linalg.norm(z)), 1e-12)
        last = float(np.linalg.norm(z - x) / denom)
        x = z
        pos = positivity_violation(x)
        leak = out_of_support_fraction(x, support)
        if pos <= C.FEASIBLE_POS_TOL and leak <= C.FEASIBLE_SUPPORT_TOL:
            break
        if last < tol:
            break
    info = {
        "n_iter": int(n_iter),
        "rel_delta": last,
        "converged": bool(
            last < tol or (pos <= C.FEASIBLE_POS_TOL and leak <= C.FEASIBLE_SUPPORT_TOL)
        ),
        "method": "dykstra",
        "positivity_violation": pos,
        "out_of_support": leak,
        "p": p,
        "q": q,
    }
    return x, info


def _project_e2a(image: np.ndarray, support: np.ndarray) -> np.ndarray:
    projected, _info = project_positivity_support(image, support)
    return projected


def constraint_diagnostics(image: np.ndarray, support: np.ndarray) -> dict:
    pos = positivity_violation(image)
    leak = out_of_support_fraction(image, support)
    return {
        "positivity_violation": pos,
        "out_of_support": leak,
        "feasible": bool(pos <= C.FEASIBLE_POS_TOL and leak <= C.FEASIBLE_SUPPORT_TOL),
    }


def isotropic_tv(image: np.ndarray, eps: float = C.E2B_TV_EPS) -> float:
    """Charbonnier isotropic TV with forward differences and replicate edges."""
    o = np.asarray(image, dtype=np.float64)
    dx = np.zeros_like(o)
    dy = np.zeros_like(o)
    dx[:, :-1] = o[:, 1:] - o[:, :-1]
    dy[:-1, :] = o[1:, :] - o[:-1, :]
    return float(np.sum(np.sqrt(dx * dx + dy * dy + eps * eps)))


def isotropic_tv_grad(image: np.ndarray, eps: float = C.E2B_TV_EPS) -> np.ndarray:
    """Gradient of :func:`isotropic_tv`."""
    o = np.asarray(image, dtype=np.float64)
    dx = np.zeros_like(o)
    dy = np.zeros_like(o)
    dx[:, :-1] = o[:, 1:] - o[:, :-1]
    dy[:-1, :] = o[1:, :] - o[:-1, :]
    nrm = np.sqrt(dx * dx + dy * dy + eps * eps)
    px = dx / nrm
    py = dy / nrm
    grad = np.zeros_like(o)
    grad[:, 1:] += px[:, :-1]
    grad[:, :-1] -= px[:, :-1]
    grad[1:, :] += py[:-1, :]
    grad[:-1, :] -= py[:-1, :]
    return grad


def e2a(
    otfs: np.ndarray,
    images: np.ndarray,
    sigma2: np.ndarray,
    lam_f: np.ndarray,
    support: np.ndarray,
    maxiter: int = C.E2A_FISTA_MAXITER,
    tol: float = C.E2A_FISTA_TOL,
    x0: np.ndarray | None = None,
    tv_mu: float = 0.0,
    tv_eps: float = C.E2B_TV_EPS,
) -> tuple[np.ndarray, dict]:
    """Positivity + spectral-support reconstruction of the E1 quadratic (§8.3).

    Optional Charbonnier TV (``tv_mu``) is the E2b production prior. A1o must
    call this with ``tv_mu=0`` so it keeps E2a object assumptions (§8.6).
    """
    num, den = wiener_num_den(otfs, images, sigma2, lam_f)
    support = np.asarray(support, dtype=np.float64)
    den = np.where(support > 0.5, den, 1.0)
    num = num * support
    start = np.fft.ifft2(num / den).real if x0 is None else np.asarray(x0, dtype=np.float64)
    o, proj_info = project_positivity_support(start, support, maxiter=max(C.DYKSTRA_MAXITER, 128))
    dual_p = proj_info["p"]
    dual_q = proj_info["q"]
    lip = float(np.max(den * support))
    if tv_mu > 0.0:
        lip = lip + 8.0 * float(tv_mu) / max(float(tv_eps), 1e-12)
    step = 1.0 / max(lip, C.DEN_FLOOR)
    y = o.copy()
    t = 1.0
    last_delta = 0.0
    n_iter = 0
    last_grad_norm = 0.0
    for n_iter in range(1, maxiter + 1):
        of = np.fft.fft2(y)
        grad = np.fft.ifft2((den * of - num) * support).real
        if tv_mu > 0.0:
            grad = grad + float(tv_mu) * isotropic_tv_grad(y, tv_eps)
        last_grad_norm = float(np.linalg.norm(grad))
        o_next, proj_info = project_positivity_support(
            y - step * grad,
            support,
            maxiter=C.DYKSTRA_WARM_MAXITER,
            p=dual_p,
            q=dual_q,
        )
        dual_p = proj_info["p"]
        dual_q = proj_info["q"]
        t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        y = o_next + ((t - 1.0) / t_next) * (o_next - o)
        last_delta = float(np.linalg.norm(o_next - o) / max(np.linalg.norm(o_next), 1e-12))
        o = o_next
        t = t_next
        if last_delta < tol:
            break
    o, proj_info = project_positivity_support(o, support, maxiter=400)
    of = np.fft.fft2(o)
    grad_o = np.fft.ifft2((den * of - num) * support).real
    if tv_mu > 0.0:
        grad_o = grad_o + float(tv_mu) * isotropic_tv_grad(o, tv_eps)
    projected, _kkt = project_positivity_support(
        o - step * grad_o,
        support,
        maxiter=max(C.DYKSTRA_MAXITER, 128),
        p=dual_p,
        q=dual_q,
    )
    kkt = float(np.linalg.norm(projected - o) / max(np.linalg.norm(o), 1e-12))
    feas = constraint_diagnostics(o, support)
    return o, {
        "n_iter": n_iter,
        "rel_delta": last_delta,
        "step": step,
        "converged": bool(last_delta < tol),
        "tv_mu": float(tv_mu),
        "positivity_violation": feas["positivity_violation"],
        "out_of_support": feas["out_of_support"],
        "feasible": feas["feasible"],
        "kkt_residual": kkt,
        "grad_norm": last_grad_norm,
        "projection_iters": int(proj_info["n_iter"]),
        "projection_method": "dykstra",
        "estimator_operator_version": C.ESTIMATOR_OPERATOR_VERSION,
    }


def e2b(
    otfs: np.ndarray,
    images: np.ndarray,
    sigma2: np.ndarray,
    lam_f: np.ndarray,
    support: np.ndarray,
    mu: float | None = None,
    tv_eps: float = C.E2B_TV_EPS,
    maxiter: int = C.E2A_FISTA_MAXITER,
    tol: float = C.E2A_FISTA_TOL,
    x0: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """E2a plus the locked production object prior (R9 §8.5)."""
    tv_mu = C.E2B_TV if mu is None else float(mu)
    recon, info = e2a(
        otfs,
        images,
        sigma2,
        lam_f,
        support,
        maxiter=maxiter,
        tol=tol,
        x0=x0,
        tv_mu=tv_mu,
        tv_eps=tv_eps,
    )
    info = dict(info)
    info["prior"] = "charbonnier_isotropic_tv"
    info["tv_eps"] = float(tv_eps)
    return recon, info


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
    e2b_lambda_rel: float = C.E2B_LAMBDA_REL
    e2b_tv: float = C.E2B_TV
    e2b_tv_eps: float = C.E2B_TV_EPS

    def field_e1(self, cfg: SimConfig, n: int) -> np.ndarray:
        return lambda_field(cfg, n, self.e1_lambda_rel, self.e1_lambda_grad)

    def field_e2a(self, cfg: SimConfig, n: int) -> np.ndarray:
        return lambda_field(cfg, n, self.e2a_lambda_rel, self.e1_lambda_grad)

    def field_e2b(self, cfg: SimConfig, n: int) -> np.ndarray:
        return lambda_field(cfg, n, self.e2b_lambda_rel, self.e1_lambda_grad)
