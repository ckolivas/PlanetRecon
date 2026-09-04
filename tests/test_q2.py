import numpy as np
import pytest

from planetrecon import constants as C
from planetrecon.config import make_config
from planetrecon.estimators import e2a, e2b, isotropic_tv, isotropic_tv_grad
from planetrecon.metric import support_mask
from planetrecon.mfbd import (
    PupilForward,
    d_tail,
    data_residual,
    fit_frame_alpha,
    frame_loss_and_grad,
    holdout_split,
    select_init,
    tip_tilt_from_shifts,
)
from planetrecon.optics import centroid_px
from planetrecon.optics import otf_from_centered_psf
from planetrecon.q2 import (
    aggregate_q2,
    closure_C,
    e2_star_name,
    prior_limited,
    select_reported_init,
)


def _blob(n: int) -> np.ndarray:
    yy, xx = np.indices((n, n))
    r2 = (xx - n / 2.0) ** 2 / 5.0**2 + (yy - n / 2.0) ** 2 / 4.0**2
    obj = np.exp(-0.5 * r2)
    obj += 0.2 * np.exp(-0.5 * ((xx - n / 2.0 - 2) ** 2 + (yy - n / 2.0 + 1) ** 2) / 1.8**2)
    return np.clip(obj, 0.0, None)


def _tiny_fwd() -> tuple[PupilForward, object]:
    cfg = make_config(1001, 4.0, n_diam=8, pupil_pad_factor=8.0, eval_size=16)
    return PupilForward.from_config(cfg, n_modes=8), cfg


def test_e2b_mu0_matches_e2a():
    n = 32
    rng = np.random.default_rng(4)
    obj = _blob(n)
    psf = np.exp(
        -0.5
        * ((np.arange(n) - n // 2)[None, :] ** 2 + (np.arange(n) - n // 2)[:, None] ** 2)
        / 1.6**2
    )
    psf /= psf.sum()
    otf = otf_from_centered_psf(psf)[None, ...]
    img = np.fft.ifft2(otf[0] * np.fft.fft2(obj)).real[None, ...]
    img = img + rng.normal(scale=0.02, size=img.shape)
    support = np.ones((n, n))
    lam = np.full((n, n), 0.03)
    a, _ia = e2a(otf, img, np.array([0.1]), lam, support)
    b, _ib = e2b(otf, img, np.array([0.1]), lam, support, mu=0.0)
    rel = np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-12)
    assert rel < 1e-12


def test_e2b_positivity_and_tv_reduces_variation():
    n = 32
    rng = np.random.default_rng(5)
    obj = _blob(n)
    yy, xx = np.indices((n, n))
    psf = np.exp(-0.5 * ((xx - n // 2) ** 2 + (yy - n // 2) ** 2) / 1.5**2)
    psf /= psf.sum()
    otf = otf_from_centered_psf(psf)[None, ...]
    img = np.fft.ifft2(otf[0] * np.fft.fft2(obj)).real[None, ...]
    img = img + rng.normal(scale=0.05, size=img.shape)
    support = np.ones((n, n))
    lam = np.full((n, n), 0.03)
    noisy, _ = e2a(otf, img, np.array([0.2]), lam, support)
    smooth, info = e2b(otf, img, np.array([0.2]), lam, support, mu=3e-2)
    assert smooth.min() >= -1e-12
    assert info["n_iter"] >= 1
    assert isotropic_tv(smooth) < isotropic_tv(noisy)


def test_tv_gradient_matches_finite_difference():
    rng = np.random.default_rng(6)
    o = rng.normal(size=(12, 11))
    eps = 0.4
    analytic = isotropic_tv_grad(o, eps)
    fd = np.empty_like(o)
    delta = 1e-6
    for i in range(o.shape[0]):
        for j in range(o.shape[1]):
            up = o.copy()
            dn = o.copy()
            up[i, j] += delta
            dn[i, j] -= delta
            fd[i, j] = (isotropic_tv(up, eps) - isotropic_tv(dn, eps)) / (2 * delta)
    rel = np.linalg.norm(analytic - fd) / max(np.linalg.norm(fd), 1e-12)
    assert rel < 2e-5


def test_mfbd_gradient_matches_finite_difference():
    fwd, _cfg = _tiny_fwd()
    n = fwd.eval_size
    obj = _blob(n)
    obj_f = np.fft.fft2(obj)
    image = np.fft.ifft2(fwd.otf(np.array([0.8, -0.5, 0.4, 0.3, -0.2, 0.1])) * obj_f).real
    alpha = np.zeros(6, dtype=np.float64)
    loss, grad = frame_loss_and_grad(alpha, fwd, obj_f, image, 1.0)
    assert np.isfinite(loss)
    fd = np.empty_like(alpha)
    delta = 1e-6
    for i in range(alpha.size):
        up = alpha.copy()
        dn = alpha.copy()
        up[i] += delta
        dn[i] -= delta
        lu, _ = frame_loss_and_grad(up, fwd, obj_f, image, 1.0)
        ld, _ = frame_loss_and_grad(dn, fwd, obj_f, image, 1.0)
        fd[i] = (lu - ld) / (2 * delta)
    significant = np.abs(fd) > 1e-8
    rel = np.linalg.norm(grad[significant] - fd[significant]) / max(
        np.linalg.norm(fd[significant]), 1e-12
    )
    assert significant.any()
    assert rel < 5e-6, rel


def test_true_alpha_has_near_zero_loss():
    fwd, _cfg = _tiny_fwd()
    n = fwd.eval_size
    obj = _blob(n)
    obj_f = np.fft.fft2(obj)
    alpha = np.array([0.3, -0.2, 0.1, 0.05], dtype=np.float64)
    image = np.fft.ifft2(fwd.otf(alpha) * obj_f).real
    loss, grad = frame_loss_and_grad(alpha, fwd, obj_f, image, 1.0)
    assert loss < 1e-18
    assert np.linalg.norm(grad) < 1e-8


def test_fit_alpha_recovers_small_phase():
    fwd, _cfg = _tiny_fwd()
    n = fwd.eval_size
    obj = _blob(n)
    obj_f = np.fft.fft2(obj)
    true = np.array([0.25, -0.15, 0.08], dtype=np.float64)
    image = np.fft.ifft2(fwd.otf(true) * obj_f).real
    rec, loss = fit_frame_alpha(np.zeros(3), fwd, obj_f, image, 1.0, n_iter=20)
    assert loss < 1e-6
    # The first two (tip/tilt-like) modes are well observed; a higher mode can
    # be weakly identifiable on this compact blob at 16×16.
    assert np.linalg.norm(rec[:2] - true[:2]) < 1e-4


def test_d_tail_improves_data_residual():
    fwd, cfg = _tiny_fwd()
    n = fwd.eval_size
    rng = np.random.default_rng(7)
    obj = _blob(n)
    k, m = 4, 5
    alphas = 0.15 * rng.normal(size=(k, m))
    otfs = fwd.otfs(alphas)
    images = np.fft.ifft2(otfs * np.fft.fft2(obj)[None, ...]).real
    sigma2 = np.full(k, 1e-6)
    lam = np.full((n, n), 1e-4)
    support = support_mask(cfg, n)
    stack = np.mean(images, axis=0)
    fit = d_tail(
        fwd,
        images,
        sigma2,
        lam,
        support,
        tv_mu=0.0,
        obj0=np.clip(stack, 0.0, None),
        train_idx=np.arange(k),
        holdout_idx=np.zeros(0, dtype=np.int64),
        m_grid=(3, 5),
        outer_iters=(2, 2),
        alpha_iters=6,
    )
    assert fit["stages"][0]["M"] == 3
    assert fit["stages"][-1]["M"] == 5
    assert fit["stages"][-1]["train_loss"] <= fit["stages"][0]["train_loss"] + 1e-12
    rel = np.linalg.norm(fit["object"] - obj) / np.linalg.norm(obj)
    assert rel < 0.25


def test_holdout_split_disjoint():
    train, ho = holdout_split(20, 0.1, seed=1001)
    assert ho.size == 2
    assert train.size == 18
    assert set(train).isdisjoint(ho)
    assert set(train).union(ho) == set(range(20))
    train2, ho2 = holdout_split(20, 0.1, seed=1001)
    assert np.array_equal(ho, ho2) and np.array_equal(train, train2)


def test_closure_and_prior_limited():
    assert closure_C(1.0, 0.7, 0.4) == pytest.approx(0.5)
    assert np.isnan(closure_C(1.0, 0.7, 1.0))
    assert prior_limited(0.04, 0.10) is True
    assert prior_limited(0.09, 0.10) is False
    assert e2_star_name(0.04, 0.10) == "E2a"
    assert e2_star_name(0.09, 0.10) == "E2b"


def test_select_init_prefers_holdout():
    results = {
        "zero": {
            "stages": [
                {"train_loss": 1.0, "holdout_loss": 3.0},
            ]
        },
        "subset": {
            "stages": [
                {"train_loss": 1.2, "holdout_loss": 1.1},
            ]
        },
    }
    assert select_init(results) == "subset"


def test_reported_init_uses_holdout_selection_when_available():
    all_frame = {
        "zero": {"stages": [{"train_loss": 1.0, "holdout_loss": None}]},
        "subset": {"stages": [{"train_loss": 2.0, "holdout_loss": None}]},
    }
    holdout = {
        "zero": {"stages": [{"train_loss": 1.0, "holdout_loss": 3.0}]},
        "subset": {"stages": [{"train_loss": 2.0, "holdout_loss": 1.0}]},
    }
    assert select_reported_init(all_frame) == "zero"
    assert select_reported_init(all_frame, holdout) == "subset"


def test_aggregate_q2_records_holdout_diagnostic():
    result = {
        "seed": 1001,
        "dr0": 4.0,
        "crops": {
            "feature": {
                "C": 0.5,
                "E_H_D": 0.7,
                "E_H_A1o": 1.0,
                "E_H_E2_star": 0.4,
                "E2_star": "E2b",
                "prior_limited": False,
                "chosen_init": "subset",
                "holdout": {
                    "chosen_init": "subset",
                    "holdout_over_train": 1.01,
                },
            }
        },
    }
    block = aggregate_q2([result], crop="feature")["4.0"]
    assert block["holdout_n"] == 1
    assert block["median_holdout_over_train"] == pytest.approx(1.01)
    assert block["holdout_chosen_inits"] == ["subset"]


def test_tip_tilt_from_shifts_matches_centroid():
    fwd, _cfg = _tiny_fwd()
    shifts = np.array([[0.15, -0.10], [0.0, 0.0], [-0.12, 0.08]])
    tt = tip_tilt_from_shifts(fwd, shifts)
    assert tt.shape == (3, 2)
    assert np.all(np.isfinite(tt))
    tt2 = tip_tilt_from_shifts(fwd, shifts)
    assert np.allclose(tt, tt2)
    # Larger requested x-shift maps to a different first coefficient.
    assert tt[0, 0] != tt[2, 0]


def test_data_residual_zero_on_perfect_prediction():
    n = 8
    obj = _blob(n)
    otf = np.ones((2, n, n), dtype=np.complex128)
    images = np.stack([obj, obj])
    assert data_residual(otf, obj, images, np.ones(2)) == pytest.approx(0.0)
