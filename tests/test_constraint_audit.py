import numpy as np

from planetrecon.constraint_audit import convolution_matrix, operator_cases, quadratic_oracle
from planetrecon.estimators import e2a
from planetrecon.optics import otf_from_centered_psf


def test_composed_operator_and_adjoint_match_direct_matrix_oracle():
    rows = operator_cases()
    assert len(rows) == 16
    assert all(r['forward_relative_error'] < 1e-11 for r in rows)
    assert all(r['adjoint_relative_error'] < 1e-11 for r in rows)


def test_dense_oracle_matches_exact_identity_with_active_positivity():
    kernel = np.zeros((1, 4, 4))
    kernel[0, 2, 2] = 1.
    images = np.random.default_rng(5).normal(size=(1, 4, 4))
    result, info, objective = quadratic_oracle(kernel, images, np.ones(1), .3, np.ones((4, 4)))
    exact = np.maximum(images[0], 0.) / 1.3
    assert info['success']
    np.testing.assert_allclose(result, exact, atol=1e-7)
    assert abs(objective(result)-objective(exact)) < 1e-10


def test_dense_oracle_respects_dc_only_support():
    kernel = np.zeros((1, 4, 4))
    kernel[0, 2, 2] = 1.
    images = np.random.default_rng(7).normal(1., 2., size=(1, 4, 4))
    support = np.zeros((4, 4))
    support[0, 0] = 1.
    result, info, _ = quadratic_oracle(kernel, images, np.ones(1), .3, support)
    assert info['success']
    np.testing.assert_allclose(result, max(images.mean(), 0.) / 1.3, atol=1e-8)


def test_warm_stationarity_certifies_active_constraints_against_oracle():
    # Audit v2's cold stationarity projection failed on this converged solution.
    y, x = np.indices((8, 8))
    kernels = []
    for width, dx, dy in ((.8, .2, -.3), (1.3, -.4, .1)):
        p = np.exp(-((x-4-dx)**2+(y-4-dy)**2)/(2*width**2))
        kernels.append(p/p.sum())
    truth = 3*np.exp(-((x-3.1)**2+(y-4.2)**2)/2.)
    variance = np.array([.04, .09])
    images = np.array([convolution_matrix((8, 8), p, circular=True) @ truth.ravel()
                       for p in kernels]).reshape(2, 8, 8)
    images += np.random.default_rng(1002).normal(size=images.shape)*np.sqrt(variance)[:, None, None]
    f = np.fft.fftfreq(8)
    support = (np.hypot(f[:, None], f[None, :]) <= .26).astype(float)
    oracle, oracle_info, _ = quadratic_oracle(kernels, images, variance, .03, support)
    fitted, info = e2a(np.array([otf_from_centered_psf(p) for p in kernels]), images,
                       variance, np.full((8, 8), .03), support, maxiter=128, x0=np.zeros((8, 8)))
    assert oracle_info['success']
    assert info['converged'] and info['kkt_projection_converged']
    assert info['kkt_residual'] < 1e-6
    np.testing.assert_allclose(fitted, oracle, atol=1e-7)
