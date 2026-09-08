import time
import numpy as np
import pytest
from tools.scene_conditioning import correction_bound, cone_residual, probe_cg
from test_scene_quadratic import dense_oracle
from test_scene_lbfgsb import problem


@pytest.mark.parametrize('seed', [31, 32, 33])
@pytest.mark.parametrize('rgb', [False, True])
def test_correction_bound_covers_dense_constrained_gap_and_distance(seed, rgb):
    p = problem(seed, rgb)
    optimum, matrix = dense_oracle(p); h = matrix.T@matrix
    rng = np.random.default_rng(seed+7)
    x = np.maximum(rng.normal(.5, 1., p.shape), 0.)
    gradient = p.gradient(x); residual = cone_residual(x, gradient)
    exact = np.linalg.solve(h, residual.ravel()).reshape(p.shape)
    corrections = [np.zeros_like(x), exact, -exact, rng.normal(size=p.shape)*10]
    correction, _ = probe_cg(p.normal, residual, p.majorizer, steps=3)
    corrections.append(correction)
    for y in corrections:
        bound = correction_bound(x, gradient, y, p.normal(y), p.ridge)
        assert p.objective(x)-p.objective(optimum) <= bound['objective_gap_upper_bound']+1e-10
        assert np.linalg.norm(x-optimum) <= bound['absolute_solution_error_bound']+1e-10
    refined = correction_bound(x, gradient, exact, p.normal(exact), p.ridge)
    assert refined['objective_gap_upper_bound'] < p.certificate(x)['objective_gap_upper_bound']


def test_diagonal_preconditioning_solves_diagonal_hessian_and_retains_trace():
    diagonal = np.geomspace(.001, 10., 64).reshape(8, 8)
    rhs = np.ones((8, 8)); events = []
    x, info = probe_cg(lambda x: diagonal*x, rhs, diagonal, steps=8, callback=lambda row, y: events.append(row))
    np.testing.assert_allclose(diagonal*x, rhs, rtol=1e-12)
    assert info['iterations'] == 1 and len(events) == 1 and not info['scene_updated']
    _, plain = probe_cg(lambda x: diagonal*x, rhs, np.ones_like(rhs), steps=8)
    assert plain['trace'][-1]['recursive_relative_residual'] > 1e-3


def test_invalid_curvature_and_deadline_are_visible():
    x = np.ones((2, 2))
    with pytest.raises(ValueError, match='curvature'):
        probe_cg(lambda p: -p, x, x)
    y, info = probe_cg(lambda p: p, x, x, deadline=time.monotonic()-1)
    assert info['reason'] == 'wall_budget' and info['iterations'] == 0 and np.all(y == 0)
    with pytest.raises(ValueError, match='feasible'):
        correction_bound(-x, x, x, x, 1.)
