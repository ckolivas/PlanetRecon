import numpy as np
import pytest

from planetrecon.constraint_audit import convolution_matrix, quadratic_oracle
from planetrecon.estimators import e1, shift_otf
from planetrecon.optics import otf_from_centered_psf
from tools.quadratic_admm import solve


@pytest.mark.parametrize('start', [None, 'zero', 'pattern'])
@pytest.mark.parametrize('bandlimited', [False, True])
@pytest.mark.parametrize('seed', [1001, 1002, 1003])
def test_admm_matches_independent_active_constraint_oracle(start, bandlimited, seed):
    y, x = np.indices((8, 8))
    kernels = []
    for width, dx, dy in ((.8, .2, -.3), (1.3, -.4, .1)):
        p = np.exp(-((x-4-dx)**2+(y-4-dy)**2)/(2*width**2))
        kernels.append(p/p.sum())
    variance = np.array([.04, .09])
    truth = 3*np.exp(-((x-3.1)**2+(y-4.2)**2)/2.)
    matrices = np.array([convolution_matrix((8, 8), p, circular=True) for p in kernels])
    images = (matrices @ truth.ravel()).reshape(2, 8, 8)
    images += np.random.default_rng(seed).normal(size=images.shape)*np.sqrt(variance)[:, None, None]
    f = np.fft.fftfreq(8)
    support = (np.hypot(f[:, None], f[None, :]) <= (.26 if bandlimited else 1.)).astype(float)
    oracle, oi, _ = quadratic_oracle(kernels, images, variance, .03, support)
    initial = None if start is None else np.zeros((8, 8)) if start == 'zero' else 2+np.cos(x)
    image, info = solve(np.array([otf_from_centered_psf(p) for p in kernels]), images, variance,
                        np.full((8, 8), .03), support, x0=initial)
    assert oi['success'] and info['converged'] and info['feasible']
    error = np.linalg.norm(image-oracle)/np.linalg.norm(image)
    # The SLSQP oracle itself has finite optimizer precision.
    assert error <= info['relative_solution_error_bound'] + 1e-6
    assert error < 1e-4


def test_distance_certificate_bounds_error_to_exact_unconstrained_solution():
    h = np.ones((2, 8, 8), dtype=complex)
    images = np.random.default_rng(4).uniform(1., 3., h.shape)
    variance = np.array([.4, 1.])
    lam = np.full((8, 8), .03)
    exact = e1(h, images, variance, lam)
    image, info = solve(h, images, variance, lam, np.ones((8, 8)), x0=np.zeros((8, 8)), maxiter=1)
    error = np.linalg.norm(image-exact)/np.linalg.norm(image)
    assert error <= info['relative_solution_error_bound']*(1+1e-12)
    assert not info['converged']


def test_active_positivity_certificate_against_exact_identity_solution():
    h = np.ones((1, 8, 8), dtype=complex)
    images = np.random.default_rng(4).normal(size=h.shape)
    exact = np.maximum(images[0], 0.)/1.03
    image, info = solve(h, images, np.ones(1), np.full((8, 8), .03), np.ones((8, 8)))
    assert info['converged'] and info['feasible']
    error = np.linalg.norm(image-exact)/np.linalg.norm(image)
    assert error <= info['relative_solution_error_bound']+1e-12


def test_zero_data_and_missing_dc():
    args = (np.ones((1, 4, 4), complex), np.zeros((1, 4, 4)), np.ones(1), np.ones((4, 4)))
    image, info = solve(*args, np.ones((4, 4)))
    assert info['converged'] and info['relative_solution_error_bound'] == 0
    assert not np.any(image)
    support = np.ones((4, 4)); support[0, 0] = 0
    with pytest.raises(ValueError, match='DC'):
        solve(*args, support)


def test_fractional_registration_uses_the_real_quadratic_normal_operator():
    # Complex Nyquist coefficients make the real-space normal matrix differ
    # from taking .real after division by the unsymmetrized Fourier diagonal.
    n = 6
    h = (shift_otf(np.ones((n, n), complex), (.3, -.2))
         + shift_otf(np.ones((n, n), complex), (-.6, .4)))/2
    kernel = np.fft.fftshift(np.fft.ifft2(h))
    matrix = convolution_matrix((n, n), kernel, circular=True)
    observed = np.random.default_rng(4).normal(10., .1, (n, n))
    normal = (matrix.conj().T @ matrix).real + .05*np.eye(n*n)
    rhs = (matrix.conj().T @ observed.ravel()).real
    exact = np.linalg.solve(normal, rhs).reshape((n, n))
    assert exact.min() > 0
    image, info = solve(h[None], observed[None], np.ones(1), np.full((n, n), .05), np.ones((n, n)))
    assert info['converged'] and info['feasible']
    np.testing.assert_allclose(image, exact, atol=1e-9)
