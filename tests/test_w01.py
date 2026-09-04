import numpy as np
import pytest

from planetrecon import constants as C
from planetrecon.config import make_config
from planetrecon.estimators import (
    e1,
    e2a,
    e2a0,
    e2b,
    out_of_support_fraction,
    positivity_violation,
    project_positivity_support,
    wiener_num_den,
)
from planetrecon.operators import (
    adjoint_relative_error,
    bin_box_adjoint,
    circular_convolve,
    circular_psf_convolve,
    crop_xy,
    crop_xy_adjoint,
    fourier_shift_adjoint,
    fourier_shift_image,
    interior_mask,
    linear_convolve_same,
    linear_convolve_same_adjoint,
    padded_detector_crop,
    padded_detector_crop_adjoint,
    quadratic_gradient,
    relative_l2,
    scalar_noise_mismatch,
    spatial_convolve_same,
)
from planetrecon.optics import bin_box, otf_from_centered_psf
from planetrecon.provenance import (
    certificate_is_current,
    current_method_fingerprint,
    physics_compatible,
)
from planetrecon.validate import check_lowfreq_ranking, ranking_jaccard


def _legacy_clip_project(image, support):
    o = np.maximum(np.asarray(image, dtype=np.float64), 0.0)
    o = np.fft.ifft2(np.fft.fft2(o) * support).real
    return np.maximum(o, 0.0)


def _impulse_support(n=32, rho=0.12):
    image = np.zeros((n, n), dtype=np.float64)
    image[n // 2, n // 2] = 1.0
    fy = np.fft.fftfreq(n)
    fx = np.fft.fftfreq(n)
    FX, FY = np.meshgrid(fx, fy, indexing="xy")
    support = (np.hypot(FX, FY) <= rho).astype(np.float64)
    return image, support


def test_legacy_clip_projection_leaks_support():
    image, support = _impulse_support()
    leaked = _legacy_clip_project(image, support)
    assert leaked.min() >= 0.0
    frac = out_of_support_fraction(leaked, support)
    assert frac == pytest.approx(0.187657, abs=5e-4)


def test_dykstra_projects_impulse_onto_intersection():
    image, support = _impulse_support()
    projected, info = project_positivity_support(image, support, maxiter=2500, tol=1e-14)
    assert info["method"] == "dykstra"
    # The documented defect is support leakage after a final positivity clip
    # (~0.1877). Dykstra removes that leakage; tiny Gibbs negatives remain until
    # the dual gap is driven down further.
    assert out_of_support_fraction(projected, support) <= 1e-12
    assert positivity_violation(projected) < 2e-5
    assert projected.min() > -2e-5


def test_e2a_reports_feasibility_and_kkt():
    n = 32
    rng = np.random.default_rng(11)
    yy, xx = np.indices((n, n))
    obj = np.exp(-0.5 * ((xx - n / 2.0) ** 2 + (yy - n / 2.0) ** 2) / 6.0**2)
    psf = np.exp(-0.5 * ((xx - n / 2.0) ** 2 + (yy - n / 2.0) ** 2) / 1.6**2)
    psf /= psf.sum()
    otf = otf_from_centered_psf(psf)[None, ...]
    img = circular_convolve(obj, otf[0])[None, ...]
    img = img + rng.normal(scale=0.02, size=img.shape)
    fy = np.fft.fftfreq(n)
    fx = np.fft.fftfreq(n)
    FX, FY = np.meshgrid(fx, fy, indexing="xy")
    support = (np.hypot(FX, FY) <= 0.45).astype(np.float64)
    recon, info = e2a(otf, img, np.array([0.1]), np.full((n, n), 0.03), support)
    assert recon.min() >= -1e-7
    assert info["out_of_support"] <= C.FEASIBLE_SUPPORT_TOL
    assert info["positivity_violation"] <= 1e-7
    assert np.isfinite(info["kkt_residual"])
    assert info["projection_method"] == "dykstra"
    assert info["estimator_operator_version"] == C.ESTIMATOR_OPERATOR_VERSION


def test_e1_e2a0_identical_objective_survives_projection_change():
    n = 32
    rng = np.random.default_rng(12)
    yy, xx = np.indices((n, n))
    obj = np.exp(-0.5 * ((xx - 16) ** 2 / 18.0**2 + (yy - 16) ** 2 / 14.0**2))
    k = 4
    otfs = np.empty((k, n, n), dtype=np.complex128)
    images = np.empty((k, n, n), dtype=np.float64)
    for i in range(k):
        psf = np.exp(
            -0.5
            * ((xx - 16 - 0.4 * i) ** 2 + (yy - 16 + 0.3 * i) ** 2)
            / (1.5 + 0.2 * i) ** 2
        )
        psf /= psf.sum()
        otfs[i] = otf_from_centered_psf(psf)
        images[i] = circular_convolve(obj, otfs[i]) + rng.normal(scale=0.01, size=obj.shape)
    lam = np.full((n, n), C.E1_LAMBDA_REL)
    sigma2 = np.full(k, 0.1)
    o1 = e1(otfs, images, sigma2, lam)
    o0, info = e2a0(otfs, images, sigma2, lam)
    assert info["cg_info"] == 0
    rel = np.linalg.norm(o0 - o1) / np.linalg.norm(o1)
    assert rel < C.E1_E2A0_IMAGE_REL_TOL


def test_e2b_uses_the_same_intersection():
    n = 24
    rng = np.random.default_rng(13)
    yy, xx = np.indices((n, n))
    obj = np.clip(np.exp(-0.5 * ((xx - 12) ** 2 + (yy - 12) ** 2) / 4.0**2), 0, None)
    psf = np.exp(-0.5 * ((xx - 12) ** 2 + (yy - 12) ** 2) / 1.4**2)
    psf /= psf.sum()
    otf = otf_from_centered_psf(psf)[None, ...]
    img = circular_convolve(obj, otf[0])[None, ...] + rng.normal(scale=0.03, size=(1, n, n))
    support = np.ones((n, n))
    recon, info = e2b(otf, img, np.array([0.15]), np.full((n, n), 0.03), support, mu=1e-2)
    assert info["feasible"]
    assert recon.min() >= -1e-10


def test_linear_convolution_matches_spatial_oracle():
    rng = np.random.default_rng(14)
    image = rng.normal(size=(12, 11))
    psf = rng.normal(size=(5, 5))
    psf -= psf.mean()
    spatial = spatial_convolve_same(image, psf, circular=False)
    linear = linear_convolve_same(image, psf)
    assert relative_l2(linear, spatial) < 1e-10


def test_circular_convolution_matches_spatial_oracle():
    rng = np.random.default_rng(15)
    image = rng.normal(size=(10, 10))
    psf = rng.normal(size=(10, 10))
    spatial = spatial_convolve_same(image, psf, circular=True)
    circular = circular_psf_convolve(image, psf)
    assert relative_l2(circular, spatial) < 1e-10


def test_linear_and_circular_adjoints():
    rng = np.random.default_rng(16)
    x = rng.normal(size=(14, 13))
    y = rng.normal(size=(14, 13))
    psf = rng.normal(size=(6, 5))
    err = adjoint_relative_error(
        lambda u: linear_convolve_same(u, psf),
        lambda v: linear_convolve_same_adjoint(v, psf),
        x,
        y,
    )
    assert err < 1e-12
    otf = otf_from_centered_psf(np.abs(rng.normal(size=(14, 13))))
    err_c = adjoint_relative_error(
        lambda u: circular_convolve(u, otf),
        lambda v: circular_convolve(v, np.conj(otf)),
        x,
        y,
    )
    assert err_c < 1e-12


def test_crop_bin_and_shift_adjoints():
    rng = np.random.default_rng(17)
    full = rng.normal(size=(20, 24))
    crop = rng.normal(size=(8, 8))
    origin = (5, 3)
    err_crop = adjoint_relative_error(
        lambda u: crop_xy(u, origin, 8),
        lambda v: crop_xy_adjoint(v, origin, full.shape),
        full,
        crop,
    )
    assert err_crop < 1e-15
    det = rng.normal(size=(6, 7))
    optical = rng.normal(size=(24, 28))
    err_bin = adjoint_relative_error(
        lambda u: bin_box(u, 4),
        lambda v: bin_box_adjoint(v, 4),
        optical,
        det,
    )
    assert err_bin < 1e-15
    img = rng.normal(size=(16, 16))
    shift = (1.25, -0.75)
    err_shift = adjoint_relative_error(
        lambda u: fourier_shift_image(u, shift),
        lambda v: fourier_shift_adjoint(v, shift),
        img,
        img,
    )
    assert err_shift < 1e-12


def test_padded_detector_crop_adjoint():
    rng = np.random.default_rng(18)
    scene = rng.normal(size=(32, 32))
    psf = rng.normal(size=(9, 9))
    residual = rng.normal(size=(8, 8))
    origin = (0, 0)
    flux = 3.5
    err = adjoint_relative_error(
        lambda u: padded_detector_crop(u, psf, origin, 8, 4, flux),
        lambda v: padded_detector_crop_adjoint(v, psf, origin, scene.shape, 4, flux),
        scene,
        residual,
    )
    assert err < 1e-11


def test_quadratic_gradient_matches_finite_difference():
    rng = np.random.default_rng(19)
    n = 16
    k = 3
    obj = rng.normal(size=(n, n))
    otfs = np.empty((k, n, n), dtype=np.complex128)
    images = np.empty((k, n, n), dtype=np.float64)
    for i in range(k):
        psf = np.abs(rng.normal(size=(n, n)))
        psf /= psf.sum()
        otfs[i] = otf_from_centered_psf(psf)
        images[i] = circular_convolve(obj, otfs[i]) + rng.normal(scale=0.02, size=(n, n))
    lam = np.full((n, n), 0.03)
    sigma2 = np.full(k, 0.2)
    grad = quadratic_gradient(obj, otfs, images, sigma2, lam)
    direction = rng.normal(size=obj.shape)
    direction /= np.linalg.norm(direction)

    def objective(x):
        num, den = wiener_num_den(otfs, images, sigma2, lam)
        xf = np.fft.fft2(x)
        # 0.5 <o, A o> - <b, o> with A o = ifft(den * fft(o)), b = ifft(num)
        return 0.5 * float(np.vdot(x, np.fft.ifft2(den * xf).real)) - float(
            np.vdot(x, np.fft.ifft2(num).real)
        )

    eps = 1e-6
    fd = (objective(obj + eps * direction) - objective(obj - eps * direction)) / (2 * eps)
    directional = float(np.vdot(grad, direction).real)
    assert abs(fd - directional) / max(abs(fd), abs(directional), 1e-12) < 2e-4


def test_interior_circular_approximation_vs_boundary():
    n = 48
    k = 11
    yy, xx = np.indices((n, n))
    ky, kx = np.indices((k, k))
    interior_obj = np.exp(-0.5 * ((xx - n / 2.0) ** 2 + (yy - n / 2.0) ** 2) / 4.0**2)
    psf_small = np.exp(-0.5 * ((kx - k // 2) ** 2 + (ky - k // 2) ** 2) / 1.3**2)
    psf_small /= psf_small.sum()
    psf_full = np.zeros((n, n), dtype=np.float64)
    y0 = n // 2 - k // 2
    x0 = n // 2 - k // 2
    psf_full[y0 : y0 + k, x0 : x0 + k] = psf_small
    linear = linear_convolve_same(interior_obj, psf_small)
    circular = circular_psf_convolve(interior_obj, psf_full)
    mask = interior_mask(n, 8)
    interior_rel = relative_l2(circular, linear, mask)
    assert interior_rel < 0.02
    edge_obj = np.exp(-0.5 * ((xx - 3.0) ** 2 + (yy - 3.0) ** 2) / 3.0**2)
    linear_e = linear_convolve_same(edge_obj, psf_small)
    circular_e = circular_psf_convolve(edge_obj, psf_full)
    edge_rel = relative_l2(circular_e, linear_e)
    assert edge_rel > interior_rel


def test_bin_box_conserves_flux_and_phase_of_block_sum():
    optical = np.arange(64, dtype=np.float64).reshape(8, 8)
    det = bin_box(optical, 4)
    assert det.shape == (2, 2)
    assert det[0, 0] == optical[:4, :4].sum()
    assert det.sum() == optical.sum()
    back = bin_box_adjoint(det, 4)
    assert back.shape == optical.shape
    assert back[:4, :4].sum() == 16 * det[0, 0]


def test_scalar_noise_is_an_approximation():
    yy, xx = np.indices((32, 32))
    expected = 50.0 + 400.0 * np.exp(-0.5 * ((xx - 16) ** 2 + (yy - 16) ** 2) / 6.0**2)
    stats = scalar_noise_mismatch(expected, read_rms=2.0)
    assert stats["scalar_variance"] == pytest.approx(float(expected.mean() + 4.0))
    assert stats["variance_rel_rms"] > 0.05
    assert stats["true_var_dynamic_range"] > 2.0


def test_lowfreq_ranking_overlap_is_bounded():
    checks = check_lowfreq_ranking()
    assert checks and checks[0].name == "lowfreq_ranking_subharmonics"
    assert checks[0].passed, checks[0].message
    assert ranking_jaccard([1, 2, 3], [2, 3, 4]) == pytest.approx(0.5)


def test_stale_and_mismatched_certificates_are_rejected():
    current = current_method_fingerprint(make_config(1001, 8.0, exposure_samples_j=8))
    stale = dict(current)
    stale["simulator_operator_version"] = "not-a-version"
    assert not certificate_is_current(stale, current)
    other = current_method_fingerprint(make_config(1001, 8.0, padding_detector_px=32))
    assert not physics_compatible(current, other)
