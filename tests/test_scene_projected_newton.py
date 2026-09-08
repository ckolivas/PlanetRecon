import time
import numpy as np
import pytest
from test_scene_lbfgsb import problem
from test_scene_quadratic import dense_oracle
from tools.scene_projected_newton import solve, direction


@pytest.mark.parametrize('seed', [31, 32, 33])
@pytest.mark.parametrize('rgb', [False, True])
@pytest.mark.parametrize('positive_start', [False, True])
def test_dense_optimum_feasibility_descent_and_wrong_active_set(seed, rgb, positive_start):
    p = problem(seed, rgb, True)
    oracle, _ = dense_oracle(p)
    initial = np.full(p.shape, 5.) if positive_start else np.zeros(p.shape)
    states = [initial]
    x, info = solve(p, x0=initial, callback=lambda row, x: states.append(x))
    assert info['converged'] and info['feasible']
    assert np.linalg.norm(x-oracle) <= info['absolute_solution_error_bound']+1e-10
    assert p.objective(x)-p.objective(oracle) <= info['objective_gap_upper_bound']+1e-10
    values = [p.objective(x) for x in states]
    assert np.all(np.diff(values) <= 1e-12)
    assert all(np.min(x) >= 0 for x in states)
    assert all(row['quadratic_objective_change'] < 0 for row in info['trace'])


def test_reduced_direction_matches_principal_dense_solve():
    h = np.array([[4., 1., 2.], [1., 3., .2], [2., .2, 5.]])
    free = np.array([True, False, True]); g = np.array([-2., 7., 1.])
    d, count = direction(lambda x: h@x, g, free, np.diag(h), steps=3, relative_tolerance=1e-12)
    np.testing.assert_allclose(d[free], np.linalg.solve(h[np.ix_(free, free)], -g[free]))
    assert d[1] == 0 and count == 2


def test_bad_newton_direction_uses_valid_projected_safeguard(monkeypatch):
    import tools.scene_projected_newton as module
    monkeypatch.setattr(module, 'direction', lambda normal, g, free, diagonal, **kw: (g.copy(), 0))
    p = problem(72)
    x, info = solve(p, max_products=3)
    assert all(row['step_kind'] == 'projected_majorizer' for row in info['trace'])
    assert p.objective(x) < p.objective(np.zeros(p.shape))


def test_deadline_and_product_budget_preserve_feasible_incomplete_outcome():
    p = problem(72, True, True, ridge=1e-5)
    for budget in [1, 5, 35]:
        x, info = solve(p, max_products=budget, tolerance=1e-12)
        assert not info['converged'] and info['reason'] == 'product_budget'
        assert info['hessian_products'] == budget and np.min(x) >= 0
    x, info = solve(p, deadline=time.monotonic()-1)
    assert info['reason'] == 'wall_budget' and info['n_iter'] == 0 and np.all(x == 0)


def test_invalid_initialization_and_curvature_rejected():
    p = problem(72)
    with pytest.raises(ValueError, match='feasible'): solve(p, x0=-np.ones(p.shape))
    with pytest.raises(ValueError, match='curvature'):
        direction(lambda x: -x, np.ones(3), np.ones(3, bool), np.ones(3))


@pytest.mark.hardware
@pytest.mark.parametrize('rgb', [False, True])
def test_cuda_solver_matches_independent_cpu_certificate(rgb):
    import torch
    from tools.scene_fft import SceneFFTBatch
    from tools.scene_quadratic import SceneQuadratic
    if not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    cpu = problem(314, rgb)
    gpu = SceneQuadratic(cpu.operators, cpu.images, [1/w for w in cpu.weights],
                         ridge=cpu.ridge, smoothness=cpu.smoothness, prior=cpu.prior,
                         batch=SceneFFTBatch(cpu.operators, device='cuda'))
    a, ia = solve(cpu)
    b, ib = solve(gpu)
    assert ia['converged'] and ib['converged']
    cert = cpu.certificate(b)
    assert cert['relative_solution_error_bound'] <= 1e-5
    assert np.linalg.norm(a-b) <= ia['absolute_solution_error_bound']+cert['absolute_solution_error_bound']
