import numpy as np
import pytest

from planetrecon import constants as C
from planetrecon.config import make_config
from planetrecon.estimators import (
    a1o,
    e1,
    e2a,
    e2a0,
    fourier_shift_image,
    lambda_field,
    register_otfs,
    shift_otf,
)
from planetrecon.evaluate import classify_gap, oracle_sensitive
from planetrecon.metric import (
    eh_metric,
    high_band_truth_fraction,
    signed_contrast,
    support_mask,
)
from planetrecon.optics import otf_from_centered_psf
from planetrecon.rank import (
    laplacian_score,
    ranking_config_hash,
    subsets_from_scores,
    top_fraction_indices,
)


def _gaussian_psf(n: int, sigma: float, shift_xy=(0.0, 0.0)) -> np.ndarray:
    yy, xx = np.indices((n, n))
    x = xx - n // 2 - shift_xy[0]
    y = yy - n // 2 - shift_xy[1]
    psf = np.exp(-0.5 * (x**2 + y**2) / sigma**2)
    return psf / psf.sum()


def _blob_object(n: int) -> np.ndarray:
    yy, xx = np.indices((n, n))
    r2 = (xx - n // 2) ** 2 / 18.0**2 + (yy - n // 2) ** 2 / 14.0**2
    obj = np.exp(-0.5 * r2)
    obj += 0.15 * np.exp(
        -0.5 * ((xx - n // 2 - 8) ** 2 + (yy - n // 2 + 4) ** 2) / 3.5**2
    )
    return obj


def test_ranking_hash_stable():
    a = ranking_config_hash()
    b = ranking_config_hash()
    assert a == b
    assert len(a) == 64


def test_laplacian_prefers_sharp():
    n = 32
    yy, xx = np.indices((n, n))
    sharp = np.exp(-0.5 * ((xx - 16) ** 2 + (yy - 16) ** 2) / 1.5**2)
    blur = np.exp(-0.5 * ((xx - 16) ** 2 + (yy - 16) ** 2) / 6.0**2)
    assert laplacian_score(sharp) > laplacian_score(blur)


def test_top_fraction_sizes():
    scores = np.arange(500, dtype=np.float64)
    assert top_fraction_indices(scores, 10).size == 50
    assert top_fraction_indices(scores, 5).size == 25
    assert top_fraction_indices(scores, 100).size == 500
    top10 = top_fraction_indices(scores, 10)
    assert set(top10) == set(range(450, 500))
    subsets = subsets_from_scores(scores)
    assert list(subsets) == [5, 10, 25, 50, 100]


def test_fourier_shift_roundtrip():
    n = 32
    img = _blob_object(n)
    shifted = fourier_shift_image(img, (2.0, -3.0))
    back = fourier_shift_image(shifted, (-2.0, 3.0))
    assert np.allclose(img, back, atol=1e-12)
    f = np.fft.fft2(img)
    fx = np.fft.fftfreq(n)
    fy = np.fft.fftfreq(n)
    FX, FY = np.meshgrid(fx, fy, indexing="xy")
    f[np.hypot(FX, FY) > 0.25] = 0
    bandlimited = np.fft.ifft2(f).real
    shifted = fourier_shift_image(bandlimited, (1.5, -2.25))
    back = fourier_shift_image(shifted, (-1.5, 2.25))
    assert np.allclose(bandlimited, back, atol=1e-10)


def test_otf_shift_convention():
    n = 32
    obj = _blob_object(n)
    psf = _gaussian_psf(n, 1.8, shift_xy=(2.0, -1.0))
    otf = otf_from_centered_psf(psf)
    img = np.fft.ifft2(otf * np.fft.fft2(obj)).real
    psf0 = _gaussian_psf(n, 1.8)
    otf0 = otf_from_centered_psf(psf0)
    assert np.allclose(shift_otf(otf, (-2.0, 1.0)), otf0, atol=1e-10)
    img_reg = fourier_shift_image(img, (-2.0, 1.0))
    img0 = np.fft.ifft2(otf0 * np.fft.fft2(obj)).real
    assert np.allclose(img_reg, img0, atol=1e-10)


def test_e1_recovers_noiseless_object():
    n = 32
    rng = np.random.default_rng(0)
    obj = _blob_object(n)
    k = 6
    otfs = np.empty((k, n, n), dtype=np.complex128)
    images = np.empty((k, n, n), dtype=np.float64)
    for i in range(k):
        sigma = 1.4 + 0.3 * i
        shift = rng.normal(scale=0.8, size=2)
        psf = _gaussian_psf(n, sigma, shift_xy=shift)
        otfs[i] = otf_from_centered_psf(psf)
        images[i] = np.fft.ifft2(otfs[i] * np.fft.fft2(obj)).real
    lam = np.full((n, n), 1e-8)
    recon = e1(otfs, images, np.full(k, 1.0), lam)
    rel = np.linalg.norm(recon - obj) / np.linalg.norm(obj)
    assert rel < 0.02


def test_e1_matches_e2a0():
    n = 32
    rng = np.random.default_rng(1)
    obj = _blob_object(n)
    k = 4
    otfs = np.empty((k, n, n), dtype=np.complex128)
    images = np.empty((k, n, n), dtype=np.float64)
    for i in range(k):
        psf = _gaussian_psf(n, 1.6 + 0.2 * i, shift_xy=rng.normal(scale=0.4, size=2))
        otfs[i] = otf_from_centered_psf(psf)
        img = np.fft.ifft2(otfs[i] * np.fft.fft2(obj)).real
        images[i] = img + rng.normal(scale=0.01, size=img.shape)
    lam = np.full((n, n), 1e-3)
    sigma2 = np.full(k, 0.01**2)
    o1 = e1(otfs, images, sigma2, lam)
    o0, info = e2a0(otfs, images, sigma2, lam, x0=o1)
    assert info["cg_info"] == 0
    img_rel = np.linalg.norm(o0 - o1) / np.linalg.norm(o1)
    window = np.ones((n, n))
    mtf = np.ones((n, n))
    mask = np.ones((n, n), dtype=bool)
    mask[0, 0] = False
    eh1 = eh_metric(o1, obj, window, mtf, mask)
    eh0 = eh_metric(o0, obj, window, mtf, mask)
    rel_eh = abs(eh0 - eh1) / max(eh1, 1e-12)
    assert img_rel < 1e-6
    assert rel_eh < C.E1_E2A0_EH_REL_TOL


def test_a1o_heff_is_registered_mean():
    n = 32
    rng = np.random.default_rng(2)
    obj = np.clip(_blob_object(n), 0, None)
    k = 5
    otfs = np.empty((k, n, n), dtype=np.complex128)
    images = np.empty((k, n, n), dtype=np.float64)
    shifts = np.zeros((k, 2))
    for i in range(k):
        shifts[i] = rng.normal(scale=1.2, size=2)
        psf = _gaussian_psf(n, 1.7, shift_xy=shifts[i])
        otfs[i] = otf_from_centered_psf(psf)
        images[i] = np.clip(np.fft.ifft2(otfs[i] * np.fft.fft2(obj)).real, 0, None)
    support = np.ones((n, n))
    lam = np.full((n, n), 1e-4)
    _recon, info = a1o(
        otfs,
        images,
        np.full(k, 0.02),
        lam,
        support,
        shifts=shifts,
        maxiter=40,
    )
    h_mean = np.mean(register_otfs(otfs, shifts), axis=0)
    assert np.allclose(info["H_eff"], h_mean, atol=1e-10)
    assert info["n_selected"] == k
    assert abs(info["sigma2_stack"] - (0.02 / k)) < 1e-15


def test_g3_equals_g1_plus_g2():
    e_a1 = 0.40
    e_s10 = 0.25
    e_s100 = 0.10
    g1 = e_s10 - e_s100
    g2 = e_a1 - e_s10
    g3 = e_a1 - e_s100
    assert abs(g3 - (g1 + g2)) <= C.G3_SUM_TOL


def test_classification_thresholds():
    strong = classify_gap([0.12] * 8 + [0.11] * 4, [False] * 12)
    assert strong["label"] == "strong"
    negative = classify_gap([0.01] * 10 + [-0.02, 0.02], [False] * 12)
    assert negative["label"] == "negative"
    inconclusive = classify_gap([0.07] * 12, [False] * 12)
    assert inconclusive["label"] == "inconclusive"
    inconclusive_pos = classify_gap([0.0] * 8 + [0.15] * 4, [False] * 12)
    assert inconclusive_pos["label"] == "inconclusive"
    path = classify_gap([0.2] * 12, [True] + [False] * 11)
    assert path["label"] == "pathological"


def test_oracle_sensitivity_rule():
    assert oracle_sensitive(0.20, 0.12) is True
    assert oracle_sensitive(0.11, 0.12) is False
    assert oracle_sensitive(0.50, 0.04) is False


def test_e2a_positivity_and_support():
    n = 32
    obj = np.clip(_blob_object(n), 0, None)
    psf = _gaussian_psf(n, 1.8)
    otf = otf_from_centered_psf(psf)[None, ...]
    img = np.fft.ifft2(otf[0] * np.fft.fft2(obj)).real[None, ...]
    rng = np.random.default_rng(3)
    img = img + rng.normal(scale=0.02, size=img.shape)
    support = np.ones((n, n))
    fy = np.fft.fftfreq(n)
    fx = np.fft.fftfreq(n)
    FX, FY = np.meshgrid(fx, fy, indexing="xy")
    support[np.hypot(FX, FY) > 0.45] = 0.0
    recon, info = e2a(otf, img, np.array([0.02**2]), np.full((n, n), 1e-3), support)
    assert recon.min() >= -1e-12
    rf = np.fft.fft2(recon)
    energy = np.abs(rf) ** 2
    leaked = float(energy[support < 0.5].sum() / max(energy.sum(), 1e-12))
    assert leaked < 1e-3
    assert info["n_iter"] >= 1


def test_signed_contrast_recovers_dip():
    n = 64
    img = np.ones((n, n))
    yy, xx = np.indices((n, n))
    ap = (xx - 32) ** 2 + (yy - 32) ** 2 <= 4**2
    an = ((xx - 32) ** 2 + (yy - 32) ** 2 >= 6**2) & (
        (xx - 32) ** 2 + (yy - 32) ** 2 <= 10**2
    )
    img[ap] = 0.8
    c = signed_contrast(img, ap, an)
    assert c == pytest.approx(-0.2, abs=1e-12)


def test_support_mask_excludes_corners():
    cfg = make_config(1001, 8.0)
    n = C.EVAL_SIZE
    mask = support_mask(cfg, n)
    assert mask[0, 0]  # DC
    assert not mask[n // 2, n // 2]  # Nyquist corner, |f|=√2 f_c


def test_rh_illconditioned_flag():
    n = 32
    window = np.ones((n, n))
    mtf = np.ones((n, n))
    mask = np.zeros((n, n), dtype=bool)
    mask[n // 4, n // 4] = True
    smooth = np.ones((n, n))
    rh = high_band_truth_fraction(smooth, window, mtf, mask)
    assert rh < C.RH_ILLCONDITIONED


@pytest.mark.slow
def test_e1_e2a0_on_development_file():
    from pathlib import Path

    from planetrecon.evaluate import load_crop
    from planetrecon.estimators import frame_noise_variance

    path = Path("out/gate1_Dr0-8_seed-01001.h5")
    if not path.exists():
        pytest.skip("development HDF5 is not present")
    cfg, crop, extras = load_crop(path, "feature")
    sigma2 = frame_noise_variance(crop.expected, extras["read_noise_e"])
    idx = np.arange(8)
    lam = lambda_field(cfg, cfg.eval_size, C.E1_LAMBDA_REL)
    o1 = e1(crop.otf[idx], crop.observed[idx], sigma2[idx], lam)
    o0, info = e2a0(
        crop.otf[idx], crop.observed[idx], sigma2[idx], lam, x0=o1, maxiter=80
    )
    eh1 = eh_metric(o1, crop.truth_e, crop.window, crop.mtf, crop.hmask)
    eh0 = eh_metric(o0, crop.truth_e, crop.window, crop.mtf, crop.hmask)
    rel = abs(eh0 - eh1) / max(abs(eh1), 1e-12)
    assert rel < C.E1_E2A0_EH_REL_TOL
    assert info["cg_info"] == 0
