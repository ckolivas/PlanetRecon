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
from scipy.optimize import minimize

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
    dopt = dopt_norm / total - psf_opt * (float(np.sum(dopt_norm)) / total)
    dpsi = 2.0 * psi * dopt
    n_opt = field.size
    dfield = np.fft.fftshift(n_opt * ifft2(np.fft.ifftshift(dpsi), workers=1))
    dphase = np.real(np.conj(dfield) * (1j * field))
    grad = fwd.modes[:, :m].T @ dphase[fwd.mask]
    return loss, np.asarray(grad, dtype=np.float64)


def tip_tilt_jacobian(fwd: PupilForward) -> tuple[np.ndarray, np.ndarray]:
    """Linear map α_tip/tilt → PSF centroid, evaluated at zero."""
    c0 = np.array(centroid_px(fwd.psf_det(np.zeros(2))), dtype=np.float64)
    cx = np.array(centroid_px(fwd.psf_det(np.array([1.0, 0.0]))), dtype=np.float64)
    cy = np.array(centroid_px(fwd.psf_det(np.array([0.0, 1.0]))), dtype=np.float64)
    jac = np.column_stack([cx - c0, cy - c0])
    return jac, c0


def tip_tilt_from_shifts(fwd: PupilForward, shifts: np.ndarray) -> np.ndarray:
    """Known detector-pixel translations as first-two KL coefficients."""
    jac, c0 = tip_tilt_jacobian(fwd)
    out = np.zeros((shifts.shape[0], 2), dtype=np.float64)
    for k in range(shifts.shape[0]):
        out[k] = np.linalg.lstsq(jac, np.asarray(shifts[k], dtype=np.float64) - c0, rcond=None)[0]
    return out


def fit_frame_alpha(
    alpha: np.ndarray,
    fwd: PupilForward,
    obj_f: np.ndarray,
    image: np.ndarray,
    sigma2: float,
    n_iter: int = C.Q2_ALPHA_ITERS,
    freeze_tip_tilt: bool = False,
) -> tuple[np.ndarray, float]:
    """L-BFGS-B on KL coefficients. Known translations may freeze tip/tilt."""
    a = np.asarray(alpha, dtype=np.float64).reshape(-1).copy()
    if freeze_tip_tilt and a.size > 2:
        tt = a[:2].copy()

        def fun_ho(ho):
            full = np.concatenate([tt, np.asarray(ho, dtype=np.float64)])
            loss, grad = frame_loss_and_grad(full, fwd, obj_f, image, sigma2)
            return loss, grad[2:]

        res = minimize(
            fun_ho,
            a[2:],
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": int(n_iter), "ftol": 1e-10, "gtol": 1e-8},
        )
        a[2:] = res.x
        return a, float(res.fun)

    def fun(x):
        return frame_loss_and_grad(x, fwd, obj_f, image, sigma2)

    res = minimize(
        fun,
        a,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": int(n_iter), "ftol": 1e-10, "gtol": 1e-8},
    )
    return np.asarray(res.x, dtype=np.float64), float(res.fun)


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
) -> None:
    idx = np.asarray(idx, dtype=np.int64)
    if idx.size == 0:
        return

    def one(k: int):
        a, _loss = fit_frame_alpha(
            alphas[k, :m],
            fwd,
            obj_f,
            images[k],
            float(sigma2[k]),
            n_iter=alpha_iters,
            freeze_tip_tilt=freeze_tip_tilt,
        )
        return int(k), a

    if frame_workers <= 1 or idx.size == 1:
        for k in idx:
            kk, a = one(int(k))
            alphas[kk, :m] = a
        return
    with ThreadPoolExecutor(max_workers=int(frame_workers)) as pool:
        for kk, a in pool.map(one, (int(k) for k in idx)):
            alphas[kk, :m] = a


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
    stages = []
    for stage, m in enumerate(m_grid):
        n_outer = int(outer_iters[stage]) if stage < len(outer_iters) else int(outer_iters[-1])
        last_info: dict = {}
        for _it in range(n_outer):
            otfs = fwd.otfs(alphas[:, :m], frame_workers=frame_workers)
            obj, last_info = reconstruct_object(
                otfs[train_idx],
                images[train_idx],
                sigma2[train_idx],
                lam_f,
                support,
                tv_mu,
                x0=obj,
            )
            obj_f = fft2(obj, workers=1)
            _fit_frames(
                fwd, alphas, m, obj_f, images, sigma2, train_idx, alpha_iters, frame_workers,
                freeze_tip_tilt=freeze_tip_tilt,
            )
        otfs = fwd.otfs(alphas[:, :m], frame_workers=frame_workers)
        obj, last_info = reconstruct_object(
            otfs[train_idx],
            images[train_idx],
            sigma2[train_idx],
            lam_f,
            support,
            tv_mu,
            x0=obj,
        )
        obj_f = fft2(obj, workers=1)
        holdout_loss = None
        if holdout_idx.size:
            _fit_frames(
                fwd, alphas, m, obj_f, images, sigma2, holdout_idx, alpha_iters, frame_workers,
                freeze_tip_tilt=freeze_tip_tilt,
            )
            otfs[holdout_idx] = fwd.otfs(alphas[holdout_idx, :m])
            holdout_loss = data_residual(
                otfs[holdout_idx], obj, images[holdout_idx], sigma2[holdout_idx]
            )
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


def holdout_split(n: int, frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    n_ho = int(round(frac * n))
    n_ho = min(max(n_ho, 0), max(n - 1, 0))
    rng = np.random.default_rng(int(seed) + 917_401)
    if n_ho == 0:
        return np.arange(n, dtype=np.int64), np.zeros(0, dtype=np.int64)
    ho = np.sort(rng.choice(n, n_ho, replace=False).astype(np.int64))
    train = np.setdiff1d(np.arange(n, dtype=np.int64), ho, assume_unique=False)
    return train, ho


def select_init(results: dict[str, dict]) -> str:
    """Choose an initialisation from held-out loss, else training loss. No truth."""
    def key(name: str) -> tuple[int, float]:
        last = results[name]["stages"][-1]
        ho = last.get("holdout_loss")
        if ho is not None and np.isfinite(ho):
            return (0, float(ho))
        return (1, float(last["train_loss"]))

    return min(results, key=key)
