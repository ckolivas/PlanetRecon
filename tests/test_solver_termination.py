"""Convergence must certify the current point, not acceleration stagnation."""
import numpy as np
import pytest

from planetrecon.estimators import e1, e2a, e2a0
from planetrecon.optics import otf_from_centered_psf


def problem():
    y, x = np.indices((8, 8))
    psf = np.exp(-((x - 4)**2 + (y - 4)**2) / 3)
    psf /= psf.sum()
    return (otf_from_centered_psf(psf)[None],
            np.random.default_rng(1).normal(size=(1, 8, 8)),
            np.ones(1), np.full((8, 8), .001))


def test_acceleration_stagnation_does_not_end_remaining_budget():
    args = problem()
    _, early = e2a(*args, np.ones((8, 8)), x0=np.zeros((8, 8)), tol=1e-3, maxiter=40)
    assert early['rel_delta'] < 1e-3 < early['kkt_residual']
    assert not early['converged']
    assert early['termination_reason'] == 'iteration_limit'
    _, complete = e2a(*args, np.ones((8, 8)), x0=np.zeros((8, 8)), tol=1e-3, maxiter=300)
    assert complete['n_iter'] > 40
    assert complete['converged']
    assert complete['kkt_residual'] < 1e-3
    assert complete['termination_reason'] == 'converged'


def test_cg_distinguishes_budget_exhaustion_and_measured_convergence():
    args = problem()
    _, early = e2a0(*args, maxiter=1)
    assert early['n_iter'] == 1
    assert not early['converged']
    assert early['normal_relative_residual'] > 1e-10
    assert early['termination_reason'] == 'iteration_limit'
    result, complete = e2a0(*args)
    assert complete['converged']
    assert 1 < complete['n_iter'] < complete['n_iter_cap']
    assert complete['normal_relative_residual'] < 1e-10
    np.testing.assert_allclose(result, e1(*args), atol=1e-8)


def test_cg_exact_zero_solution_requires_no_iterations():
    h, images, variance, lam = problem()
    result, info = e2a0(h, np.zeros_like(images), variance, lam)
    assert info['converged'] and info['n_iter'] == 0
    assert info['normal_residual'] == 0
    assert not np.any(result)


@pytest.mark.parametrize('kwargs', [{'maxiter': 0}, {'tol': 0}, {'tol': float('nan')}])
def test_invalid_constraint_solver_budget_is_rejected(kwargs):
    with pytest.raises(ValueError):
        e2a(*problem(), np.ones((8, 8)), **kwargs)
