import numpy as np

from planetrecon.constraint_audit import convolution_matrix
from planetrecon.estimators import e1
from planetrecon.optics import otf_from_centered_psf
from tools.audit_noise_weights import weighted_solve


def problem():
    y, x = np.indices((6, 6))
    kernels = [np.exp(-((x-3-dx)**2+(y-3)**2)/width) for dx, width in ((.2, 1.), (-.5, 2.))]
    kernels = np.array([p/p.sum() for p in kernels])
    h = np.array([otf_from_centered_psf(p) for p in kernels])
    images = np.random.default_rng(14).normal(2., 1., h.shape)
    variance = np.random.default_rng(8).uniform(.1, 3., h.shape)
    return kernels, h, images, variance, np.full((6, 6), .03)


def test_heteroscedastic_normal_equation_matches_independent_dense_solution():
    kernels, h, images, variance, lam = problem()
    matrices = [convolution_matrix(lam.shape, p, circular=True) for p in kernels]
    normal = sum(a.T @ ((1/v.ravel())[:, None]*a) for a, v in zip(matrices, variance)) + .03*np.eye(36)
    rhs = sum(a.T @ (y/v).ravel() for a, y, v in zip(matrices, images, variance))
    oracle = np.linalg.solve(normal, rhs).reshape(lam.shape)
    fitted, info = weighted_solve(h, images, variance, lam, batch=1)
    assert info['converged'] and info['normal_relative_residual'] <= 1e-9
    np.testing.assert_allclose(fitted, oracle, atol=1e-7, rtol=1e-7)


def test_constant_pixel_variance_reduces_to_scalar_fourier_estimator():
    _, h, images, _, lam = problem()
    scalar = np.array([.4, 1.3])
    variance = np.broadcast_to(scalar[:, None, None], images.shape)
    fitted, info = weighted_solve(h, images, variance, lam)
    assert info['converged'] and info['n_iter'] == 1
    np.testing.assert_allclose(fitted, e1(h, images, scalar, lam), atol=1e-12)


def test_budget_exhaustion_cannot_be_reported_as_convergence():
    _, h, images, variance, lam = problem()
    _, info = weighted_solve(h, images, variance, lam, maxiter=1)
    assert not info['converged'] and info['cg_info'] == 1
