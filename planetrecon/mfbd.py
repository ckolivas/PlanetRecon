"""Blind MFBD estimator D and mode-continuation D-tail (R9 §18).

The solver uses a snapshot (mid-exposure) pupil phase in the first ``M_fit``
QR-orthonormal Noll modes on the obstructed pupil. It does not share a closed
low-order subspace with the Kolmogorov screen: residual phase beyond 60 modes
is documented by Prompt 1. Finite-exposure averaging is *not* modelled; that
mismatch is part of the Q2 test.

Observations remain unregistered. Known Gate-1 translations initialise and
freeze pupil tip/tilt, while higher modes remain fitted variables.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
from scipy.fft import fft2, ifft2
from scipy.optimize import minimize, least_squares

from planetrecon import constants as C
from planetrecon.config import SimConfig
from planetrecon.estimators import e2a, e2b
from planetrecon.kl import KLBasis
from planetrecon.optics import (
    bin_box,
    center_crop,
    centroid_px,
    make_pupil,
    otf_from_centered_psf,
)


def _bin_box_adj(grad_det: np.ndarray, factor: int) -> np.ndarray:
    if factor == 1:
        return np.asarray(grad_det, dtype=np.float64)
    return np.repeat(np.repeat(np.asarray(grad_det, dtype=np.float64), factor, axis=0), factor, axis=1)


def _center_crop_adj(grad_crop: np.ndarray, full_hw: tuple[int, int]) -> np.ndarray:
    ny, nx = full_hw
    cy, cx = grad_crop.shape
    out = np.zeros((ny, nx), dtype=np.float64)
    y0 = ny // 2 - cy // 2
    x0 = nx // 2 - cx // 2
    out[y0 : y0 + cy, x0 : x0 + cx] = grad_crop
    return out


@dataclass
class PupilForward:
    """Snapshot PSF / OTF from KL coefficients on the simulator pupil."""

    amplitude: np.ndarray
    mask: np.ndarray
    modes: np.ndarray
    bin_factor: int
    eval_size: int

    @classmethod
    def from_config(cls, cfg: SimConfig, n_modes: int = C.KL_MODES) -> "PupilForward":
        pupil = make_pupil(cfg)
        basis = KLBasis(pupil, n_modes=n_modes)
        return cls(
            amplitude=np.asarray(pupil.amplitude, dtype=np.float64),
            mask=np.asarray(pupil.mask, dtype=bool),
            modes=np.asarray(basis.modes, dtype=np.float64),
            bin_factor=int(cfg.bin_factor),
            eval_size=int(cfg.eval_size),
        )

    @property
    def n_modes(self) -> int:
        return int(self.modes.shape[1])

    def phase(self, alpha: np.ndarray) -> np.ndarray:
        a = np.asarray(alpha, dtype=np.float64).reshape(-1)
        ph = np.zeros(self.amplitude.shape, dtype=np.float64)
        if a.size:
            ph[self.mask] = self.modes[:, : a.size] @ a
        return ph

    def psf_det(self, alpha: np.ndarray) -> np.ndarray:
        field = self.amplitude * np.exp(1j * self.phase(alpha))
        psi = np.fft.fftshift(fft2(np.fft.ifftshift(field), workers=1))
        iopt = np.abs(psi) ** 2
        total = float(iopt.sum())
        if total <= 0:
            raise RuntimeError("PSF has zero energy")
        psf_opt = iopt / total
        psf = center_crop(bin_box(psf_opt, self.bin_factor), self.eval_size)
        s = float(psf.sum())
        return psf if s <= 0 else psf / s

    def otf(self, alpha: np.ndarray) -> np.ndarray:
        return otf_from_centered_psf(self.psf_det(alpha))

    def otfs(self, alphas: np.ndarray, frame_workers: int = 1) -> np.ndarray:
        k = alphas.shape[0]
        n = self.eval_size
        out = np.empty((k, n, n), dtype=np.complex128)
        if frame_workers <= 1 or k <= 1:
            for i in range(k):
                out[i] = self.otf(alphas[i])
            return out

        def one(i: int):
            return i, self.otf(alphas[i])

        with ThreadPoolExecutor(max_workers=int(frame_workers)) as pool:
            for i, otf in pool.map(one, range(k)):
                out[i] = otf
        return out


def frame_loss_and_grad(
    alpha: np.ndarray,
    fwd: PupilForward,
    obj_f: np.ndarray,
    image: np.ndarray,
    sigma2: float,
) -> tuple[float, np.ndarray]:
    """Data-fidelity loss and gradient wrt KL coefficients for one frame."""
    a = np.asarray(alpha, dtype=np.float64).reshape(-1)
    m = a.size
    phase = fwd.phase(a)
    field = fwd.amplitude * np.exp(1j * phase)
    psi = np.fft.fftshift(fft2(np.fft.ifftshift(field), workers=1))
    iopt = np.abs(psi) ** 2
    total = float(iopt.sum())
    if total <= 0:
        raise RuntimeError("PSF has zero energy")
    psf_opt = iopt / total
    binned = bin_box(psf_opt, fwd.bin_factor)
    psf = center_crop(binned, fwd.eval_size)
    sdet = float(psf.sum())
    if sdet > 0:
        psf = psf / sdet
    else:
        sdet = 1.0

    n = fwd.eval_size
    h = fft2(np.fft.ifftshift(psf), workers=1)
    pred = ifft2(h * obj_f, workers=1).real
    img = np.asarray(image, dtype=np.float64)
    resid = pred - img
    inv_var = 1.0 / max(float(sigma2), C.DEN_FLOOR)
    loss = float(np.sum(resid * resid) * inv_var)

    dpred = (2.0 * resid) * inv_var
    dh = fft2(dpred, workers=1) * np.conj(obj_f) / n
    dpsf = np.fft.ifftshift(n * ifft2(dh, workers=1)).real
    dpsf = (dpsf - float(np.sum(dpsf * psf))) / sdet
    dbinned = _center_crop_adj(dpsf, binned.shape)
    dopt_norm = _bin_box_adj(dbinned, fwd.bin_factor)
    # Adjoint of I / sum(I): subtract the scalar inner product, not a
    # PSF-shaped term. The latter is the untransposed Jacobian and corrupts
    # higher-mode gradients, particularly for large calibrated pupil tilts.
    dopt = (dopt_norm - float(np.sum(dopt_norm * psf_opt))) / total
    dpsi = 2.0 * psi * dopt
    n_opt = field.size
    dfield = np.fft.fftshift(n_opt * ifft2(np.fft.ifftshift(dpsi), workers=1))
    dphase = np.real(np.conj(dfield) * (1j * field))
    grad = fwd.modes[:, :m].T @ dphase[fwd.mask]
    return loss, np.asarray(grad, dtype=np.float64)


def tip_tilt_jacobian(fwd: PupilForward, phase_base=None) -> tuple[np.ndarray, np.ndarray]:
    """Local centroid Jacobian; an initializer for the nonlinear calibration."""
    base = np.zeros(2) if phase_base is None else np.asarray(phase_base, dtype=float).copy()
    if base.ndim != 1 or not 2 <= base.size <= fwd.n_modes or not np.isfinite(base).all():
        raise ValueError('tip/tilt calibration requires at least two finite pupil modes')
    base[:2] = 0.
    c0 = np.array(centroid_px(fwd.psf_det(base)), dtype=np.float64)
    columns = []
    for axis in range(2):
        plus, minus = base.copy(), base.copy()
        plus[axis], minus[axis] = 1e-3, -1e-3
        columns.append((np.array(centroid_px(fwd.psf_det(plus))) -
                        np.array(centroid_px(fwd.psf_det(minus)))) / 2e-3)
    jac = np.column_stack(columns)
    return jac, c0


def tip_tilt_from_shifts(fwd: PupilForward, shifts: np.ndarray, *, phase_base=None,
                        return_info: bool = False):
    """Fit first-two KL coefficients to the actual cropped detector PSF centroid.

    A one-radian secant is inaccurate after detector binning/cropping. The local
    Jacobian only initializes a bounded nonlinear solve, whose centroid error
    must pass before the coefficients can be used. Optional fixed higher modes
    qualify the mapping under aberration; centroid agreement alone does not
    establish a physical pupil tilt from measured image motion.
    """
    shifts = np.asarray(shifts, dtype=float)
    if shifts.ndim != 2 or shifts.shape[1] != 2 or not np.isfinite(shifts).all():
        raise ValueError('shifts must be a finite (N, 2) array')
    if np.any(shifts < -(fwd.eval_size//2)) or np.any(shifts > (fwd.eval_size-1)//2):
        raise ValueError('requested centroid is outside the detector crop')
    bases = np.zeros((len(shifts), 2)) if phase_base is None else np.asarray(phase_base, dtype=float)
    if (bases.ndim != 2 or bases.shape[0] != len(shifts) or not 2 <= bases.shape[1] <= fwd.n_modes
            or not np.isfinite(bases).all()):
        raise ValueError('phase_base must contain finite (N, M) coefficients with at least two modes')
    out = np.zeros((len(shifts), 2), dtype=np.float64)
    diagnostics = []
    common = tip_tilt_jacobian(fwd) if phase_base is None else None
    for k, target in enumerate(shifts):
        base = bases[k].copy()
        jac, c0 = common if common is not None else tip_tilt_jacobian(fwd, base)
        if not np.isfinite(jac).all() or np.linalg.cond(jac) > 1e8:
            raise ValueError('tip/tilt centroid calibration is ill-conditioned')
        initial = np.linalg.solve(jac, target-c0)
        def residual(coefficients):
            base[:2] = coefficients
            return np.array(centroid_px(fwd.psf_det(base))) - target
        fit = least_squares(residual, initial, max_nfev=64, ftol=1e-11, xtol=1e-11, gtol=1e-11)
        error = float(np.linalg.norm(residual(fit.x)))
        if not fit.success or not np.isfinite(error) or error > 1e-6:
            raise ValueError(f'tip/tilt centroid calibration failed for frame {k}: error {error:.6g} px')
        out[k] = fit.x
        diagnostics.append({'success': True, 'status': int(fit.status), 'nfev': int(fit.nfev),
                            'centroid_error_px': error, 'optimality': float(fit.optimality)})
    info = {'method': 'nonlinear cropped detector-centroid calibration v2',
            'tolerance_px': 1e-6, 'frames': diagnostics}
    return (out, info) if return_info else out


def fit_frame_alpha(
    alpha: np.ndarray,
    fwd: PupilForward,
    obj_f: np.ndarray,
    image: np.ndarray,
    sigma2: float,
    n_iter: int = C.Q2_ALPHA_ITERS,
    freeze_tip_tilt: bool = False,
    *,
    return_info: bool = False,
):
    """L-BFGS-B with explicit frozen coordinates and actual optimizer evidence."""
    a = np.asarray(alpha, dtype=np.float64).reshape(-1).copy()
    if (not np.all(np.isfinite(a)) or a.size > fwd.n_modes or
            not np.all(np.isfinite(obj_f)) or not np.all(np.isfinite(image)) or
            not np.isfinite(sigma2) or sigma2 <= 0 or type(n_iter) is not int or n_iter < 1):
        raise ValueError("phase fit requires finite inputs, positive variance and iteration budget")
    frozen = min(2, a.size) if freeze_tip_tilt else 0
    scale = max(float(np.sum(np.asarray(image, dtype=np.float64)**2)/sigma2), 1.)
    trace = []

    def fun(free):
        full = np.concatenate([a[:frozen], np.asarray(free, dtype=np.float64)])
        loss, grad = frame_loss_and_grad(full, fwd, obj_f, image, sigma2)
        if not np.isfinite(loss) or not np.all(np.isfinite(grad)):
            raise ValueError("non-finite phase objective or gradient")
        return loss, grad[frozen:]

    loss0, _ = fun(a[frozen:])
    trace.append(float(loss0))
    if a.size == frozen:
        info = {"success": True, "status": 0, "message": "all coordinates frozen",
                "nit": 0, "nfev": 1, "gradient_norm": 0.0, "frozen_modes": frozen,
                "data_energy_scale": scale, "relative_gradient_inf": 0.0,
                "objective_trace": trace}
        return (a, loss0, info) if return_info else (a, loss0)
    def scaled_fun(free):
        loss, grad = fun(free)
        return loss/scale, grad/scale

    res = minimize(scaled_fun, a[frozen:], method="L-BFGS-B", jac=True,
        callback=lambda x: trace.append(float(fun(x)[0])),
        options={"maxiter": n_iter, "ftol": 1e-14, "gtol": 1e-10, "maxls": 40})
    if not np.all(np.isfinite(res.x)) or not np.isfinite(res.fun):
        raise ValueError("non-finite phase optimizer result")
    a[frozen:] = res.x
    loss, gradient = fun(res.x)
    info = {"success": bool(res.success), "status": int(res.status), "message": str(res.message),
            "nit": int(res.nit), "nfev": int(res.nfev),
            "gradient_norm": float(np.linalg.norm(gradient)), "frozen_modes": frozen,
            "data_energy_scale": scale,
            "relative_gradient_inf": float(np.max(np.abs(gradient), initial=0.)/scale),
            "objective_trace": trace}
    return (a, loss, info) if return_info else (a, loss)


def data_residual(
    otfs: np.ndarray,
    obj: np.ndarray,
    images: np.ndarray,
    sigma2: np.ndarray,
) -> float:
    obj_f = fft2(obj, workers=1)
    pred = ifft2(otfs * obj_f[None, ...], workers=1).real
    resid = pred - images
    w = 1.0 / np.clip(np.asarray(sigma2, dtype=np.float64), C.DEN_FLOOR, None)
    return float(np.mean(np.sum(resid * resid, axis=(-2, -1)) * w))


def initial_object(
    fwd: PupilForward,
    images: np.ndarray,
    sigma2: np.ndarray,
    lam_f: np.ndarray,
    support: np.ndarray,
    idx: np.ndarray,
    tv_mu: float,
    alphas: np.ndarray | None = None,
) -> np.ndarray:
    """Object start from selected frames. ``alphas`` are snapshot KL coefficients."""
    idx = np.asarray(idx, dtype=np.int64)
    if alphas is None:
        otfs = fwd.otfs(np.zeros((idx.size, 1)))
    else:
        otfs = fwd.otfs(np.asarray(alphas, dtype=np.float64)[idx])
    return reconstruct_object(
        otfs,
        images[idx],
        sigma2[idx],
        lam_f,
        support,
        tv_mu,
        x0=np.clip(np.mean(images[idx], axis=0), 0.0, None),
    )[0]


def reconstruct_object(
    otfs: np.ndarray,
    images: np.ndarray,
    sigma2: np.ndarray,
    lam_f: np.ndarray,
    support: np.ndarray,
    tv_mu: float,
    x0: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    if tv_mu > 0.0:
        return e2b(otfs, images, sigma2, lam_f, support, mu=tv_mu, x0=x0)
    return e2a(otfs, images, sigma2, lam_f, support, x0=x0, tv_mu=0.0)


def _fit_frames(
    fwd: PupilForward,
    alphas: np.ndarray,
    m: int,
    obj_f: np.ndarray,
    images: np.ndarray,
    sigma2: np.ndarray,
    idx: np.ndarray,
    alpha_iters: int,
    frame_workers: int,
    freeze_tip_tilt: bool = False,
) -> list[dict]:
    idx = np.asarray(idx, dtype=np.int64)
    if idx.size == 0:
        return []

    def one(k: int):
        a, _loss, info = fit_frame_alpha(
            alphas[k, :m],
            fwd,
            obj_f,
            images[k],
            float(sigma2[k]),
            n_iter=alpha_iters,
            return_info=True,
            freeze_tip_tilt=freeze_tip_tilt,
        )
        return int(k), a, {"frame": int(k), **info}

    diagnostics = []
    if frame_workers <= 1 or idx.size == 1:
        for k in idx:
            kk, a, info = one(int(k))
            alphas[kk, :m] = a
            diagnostics.append(info)
        return diagnostics
    with ThreadPoolExecutor(max_workers=min(32, int(frame_workers))) as pool:
        for kk, a, info in pool.map(one, (int(k) for k in idx)):
            alphas[kk, :m] = a
            diagnostics.append(info)
    return diagnostics


def d_tail(
    fwd: PupilForward,
    images: np.ndarray,
    sigma2: np.ndarray,
    lam_f: np.ndarray,
    support: np.ndarray,
    *,
    tv_mu: float,
    obj0: np.ndarray,
    train_idx: np.ndarray,
    holdout_idx: np.ndarray,
    m_grid: tuple[int, ...] = C.M_FIT_GRID,
    outer_iters: tuple[int, ...] = C.Q2_OUTER_ITERS,
    alpha_iters: int = C.Q2_ALPHA_ITERS,
    alpha0: np.ndarray | None = None,
    frame_workers: int = C.Q2_FRAME_WORKERS,
    freeze_tip_tilt: bool = False,
) -> dict:
    """Mode-continuation MFBD: D at the first M, then D-tail through ``m_grid``.

    Object and coefficients are fitted on ``train_idx``. Hold-out frames, if
    any, receive a phase-only fit against the frozen object after each stage.
    """
    n_frames = images.shape[0]
    if (not m_grid or any(type(m) is not int or m < 1 for m in m_grid)
            or list(m_grid) != sorted(set(m_grid)) or not outer_iters
            or any(type(i) is not int or i < 1 for i in outer_iters)):
        raise ValueError("mode stages and positive iteration budgets must be ordered")
    m_max = int(m_grid[-1])
    if m_max > fwd.n_modes:
        raise ValueError(f"M_fit={m_max} exceeds basis ({fwd.n_modes})")
    alphas = np.zeros((n_frames, m_max), dtype=np.float64)
    if alpha0 is not None:
        w = min(alpha0.shape[1], m_max)
        alphas[:, :w] = alpha0[:, :w]
    obj = np.asarray(obj0, dtype=np.float64)
    train_idx = np.asarray(train_idx, dtype=np.int64)
    holdout_idx = np.asarray(holdout_idx, dtype=np.int64)
    for indices in (train_idx, holdout_idx):
        if np.any(indices < 0) or np.any(indices >= n_frames) or np.unique(indices).size != indices.size:
            raise ValueError("invalid or duplicate fit indices")
    if train_idx.size == 0 or np.intersect1d(train_idx, holdout_idx).size:
        raise ValueError("training set must be nonempty and disjoint from holdout")
    stages = []
    for stage, m in enumerate(m_grid):
        outer_cap = int(outer_iters[stage]) if stage < len(outer_iters) else int(outer_iters[-1])
        last_info: dict = {}
        phase_diagnostics = []
        trace = []
        otfs = fwd.otfs(alphas[:, :m], frame_workers=frame_workers)
        obj, last_info = reconstruct_object(otfs[train_idx], images[train_idx], sigma2[train_idx],
                                            lam_f, support, tv_mu, x0=obj)
        for n_outer in range(1, outer_cap+1):
            old_obj, old_otfs = obj.copy(), otfs[train_idx].copy()
            obj_f = fft2(obj, workers=1)
            phase_info = _fit_frames(
                fwd, alphas, m, obj_f, images, sigma2, train_idx, alpha_iters, frame_workers,
                freeze_tip_tilt=freeze_tip_tilt,
            )
            phase_diagnostics.append({"partition": "train", "outer": n_outer, "frames": phase_info})
            otfs = fwd.otfs(alphas[:, :m], frame_workers=frame_workers)
            obj, last_info = reconstruct_object(otfs[train_idx], images[train_idx], sigma2[train_idx],
                                                lam_f, support, tv_mu, x0=obj)
            current_phase = phase_stationarity(fwd, alphas[:, :m], obj, images, sigma2, train_idx,
                                                freeze_tip_tilt=freeze_tip_tilt)
            obj_delta = float(np.linalg.norm(obj-old_obj)/max(np.linalg.norm(obj), 1e-12))
            otf_delta = float(np.linalg.norm(otfs[train_idx]-old_otfs)/max(np.linalg.norm(otfs[train_idx]), 1e-12))
            joint = bool(last_info.get('converged', False) and current_phase['stationary']
                         and obj_delta < C.Q2_JOINT_REL_TOL and otf_delta < C.Q2_JOINT_REL_TOL)
            trace.append({'outer': n_outer, 'object_rel_delta': obj_delta, 'otf_rel_delta': otf_delta,
                          'train_loss': data_residual(otfs[train_idx], obj, images[train_idx], sigma2[train_idx]),
                          'phase_relative_gradient_inf': current_phase['max_relative_gradient_inf'],
                          'object_converged': bool(last_info.get('converged', False)), 'converged': joint})
            if joint:
                break
        obj_f = fft2(obj, workers=1)
        holdout_loss = None
        holdout_stationarity = None
        if holdout_idx.size:
            phase_info = _fit_frames(
                fwd, alphas, m, obj_f, images, sigma2, holdout_idx, alpha_iters, frame_workers,
                freeze_tip_tilt=freeze_tip_tilt,
            )
            phase_diagnostics.append({"partition": "model_selection", "frames": phase_info})
            otfs[holdout_idx] = fwd.otfs(alphas[holdout_idx, :m])
            holdout_loss = data_residual(
                otfs[holdout_idx], obj, images[holdout_idx], sigma2[holdout_idx]
            )
            holdout_stationarity = phase_stationarity(fwd, alphas[:, :m], obj, images, sigma2, holdout_idx,
                                                      freeze_tip_tilt=freeze_tip_tilt)
        train_loss = data_residual(otfs[train_idx], obj, images[train_idx], sigma2[train_idx])
        stages.append(
            {
                "M": int(m),
                "object": obj.copy(),
                "alphas": alphas[:, :m].copy(),
                "otfs": otfs,
                "train_loss": train_loss,
                "holdout_loss": holdout_loss,
                "n_outer": n_outer,
                "outer_cap": outer_cap,
                "convergence": {'converged': joint and (holdout_stationarity is None or holdout_stationarity['stationary']),
                                'training_converged': joint, 'phase': current_phase,
                                'model_selection_phase': holdout_stationarity,
                                'trace': trace, 'relative_change_tolerance': C.Q2_JOINT_REL_TOL},
                "phase_fits": phase_diagnostics,
                "object_info": {
                    k: v for k, v in last_info.items() if k != "H_eff"
                },
            }
        )
    return {
        "object": obj,
        "alphas": alphas,
        "stages": stages,
        "train_idx": train_idx,
        "holdout_idx": holdout_idx,
    }


def phase_stationarity(fwd, alphas, obj, images, sigma2, indices, *, freeze_tip_tilt=False):
    """Evaluate free-coordinate gradients at the final object/phase pair.

    Gradients are relative to measured weighted image energy, independent of
    truth and of arbitrary brightness units. A stationary fit can still have
    large residual/model error; those require separate acceptance checks.
    """
    obj_f = fft2(obj, workers=1)
    rows = []
    for k in indices:
        loss, gradient = frame_loss_and_grad(alphas[k], fwd, obj_f, images[k], sigma2[k])
        if freeze_tip_tilt:
            gradient = gradient[min(2, gradient.size):]
        scale = max(float(np.sum(np.asarray(images[k], dtype=np.float64)**2)/sigma2[k]), 1.)
        raw = float(np.max(np.abs(gradient), initial=0.))
        rows.append({'frame': int(k), 'loss': loss, 'gradient_inf': raw,
                     'data_energy_scale': scale, 'relative_gradient_inf': raw/scale})
    maximum = max((r['relative_gradient_inf'] for r in rows), default=float('inf'))
    return {'stationary': bool(np.isfinite(maximum) and maximum <= C.Q2_PHASE_GRAD_REL_TOL),
            'max_relative_gradient_inf': maximum, 'tolerance': C.Q2_PHASE_GRAD_REL_TOL,
            'normalization': 'weighted observed image energy, floored at one', 'frames': rows}


def fit_converged(fit):
    """Require current joint checks; old optimizer status flags are insufficient."""
    return bool(fit['stages']) and all(st.get('convergence', {}).get('converged', False)
                                      and st['object_info'].get('converged', False) for st in fit['stages'])


def holdout_split(n: int, frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    n_ho = int(round(frac * n))
    n_ho = min(max(n_ho, 0), max(n - 1, 0))
    rng = np.random.default_rng(int(seed) + 917_401)
    if n_ho == 0:
        return np.arange(n, dtype=np.int64), np.zeros(0, dtype=np.int64)
    ho = np.sort(rng.choice(n, n_ho, replace=False).astype(np.int64))
    train = np.setdiff1d(np.arange(n, dtype=np.int64), ho, assume_unique=False)
    return train, ho


def assessment_split(n: int, frac: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Preassign disjoint training, selection and untouched assessment frames.

    This separates uses of data within a capture; it does not claim independent
    nights or independent atmospheric samples from temporally correlated frames.
    """
    if n < 3 or not np.isfinite(frac) or not 0 < frac < .5:
        raise ValueError('three-way assessment requires at least three frames and 0 < fraction < 0.5')
    count = min(max(1, int(round(frac*n))), (n-1)//2)
    shuffled = np.random.default_rng(int(seed)+917_402).permutation(n)
    assess = np.sort(shuffled[:count])
    select = np.sort(shuffled[count:2*count])
    train = np.sort(shuffled[2*count:])
    return train, select, assess


def assess_selected_fit(fwd, fit, images, sigma2, indices, alpha0, *,
                        alpha_iters=C.Q2_ALPHA_ITERS, frame_workers=1):
    """Profile phase nuisance parameters once after freezing the selected model.

    The training object is never updated and assessment phases start from the
    externally specified shift calibration, not an all-frame fit. The residual
    is a phase-profiled diagnostic, not an unfitted predictive likelihood.
    """
    indices = np.asarray(indices, dtype=np.int64)
    used = np.union1d(fit['train_idx'], fit['holdout_idx'])
    if (indices.ndim != 1 or indices.size == 0 or np.unique(indices).size != indices.size
            or np.any(indices < 0) or np.any(indices >= images.shape[0])
            or np.intersect1d(indices, used).size):
        raise ValueError('assessment indices must be nonempty, unique and excluded from training and selection')
    m = fit['stages'][-1]['M']
    alphas = np.asarray(alpha0, dtype=np.float64).copy()
    obj = np.asarray(fit['object'], dtype=np.float64).copy()
    phase_info = _fit_frames(fwd, alphas, m, fft2(obj, workers=1), images, sigma2,
                             indices, alpha_iters, frame_workers, freeze_tip_tilt=True)
    otfs = fwd.otfs(alphas[indices, :m])
    loss = data_residual(otfs, obj, images[indices], sigma2[indices])
    finite = bool(np.isfinite(loss) and loss >= 0)
    current_phase = phase_stationarity(fwd, alphas[:, :m], obj, images, sigma2, indices, freeze_tip_tilt=True)
    converged = fit_converged(fit) and current_phase['stationary']
    return {'partition_role': 'assessment only after model selection',
            'metric_role': 'phase-profiled residual with frozen training object; not unfitted prediction',
            'sampling_limit': 'disjoint frames in one capture; temporal correlation remains',
            'indices': indices.tolist(), 'n_frames': int(indices.size), 'M': int(m),
            'loss': float(loss) if finite else None, 'phase_fits': phase_info, 'phase_stationarity': current_phase,
            'status': 'invalid' if not finite else 'valid' if converged else 'incomplete'}


def select_init(results: dict[str, dict]) -> str:
    """Choose an initialisation from held-out loss, else training loss. No truth."""
    if not results:
        raise ValueError("no initialization candidates")
    use_holdout = any(r["stages"][-1].get("holdout_loss") is not None for r in results.values())
    candidates = {}
    for name, result in results.items():
        last = result["stages"][-1]
        value = last.get("holdout_loss") if use_holdout else last.get("train_loss")
        if value is not None and np.isfinite(value) and value >= 0:
            candidates[name] = float(value)
    if not candidates:
        raise ValueError("all initialization candidates have invalid losses")
    return min(candidates, key=candidates.get)
