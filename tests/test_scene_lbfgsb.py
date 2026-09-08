import time
import numpy as np
import pytest
from planetrecon.operators import SceneDetectorOperator
from tools.scene_fft import SceneFFTBatch
from tools.scene_quadratic import SceneQuadratic
from tools.scene_lbfgsb import solve
from tools.scene_residual import objective_gradient
from test_scene_quadratic import dense_oracle


def problem(seed, rgb=False, fast=False, ridge=.01):
    rng = np.random.default_rng(seed)
    shape = (6, 6)+( (3,) if rgb else () )
    psf = rng.uniform(size=(2, 3)); psf /= psf.sum()
    ops = [SceneDetectorOperator(shape, (psf,), 2, (0, 0), (3, 3),
                                cfa_pattern='RGGB' if rgb else None)]
    return SceneQuadratic(ops, [rng.normal(.1, 1., ops[0].output_shape)],
                          [rng.uniform(.5, 2., ops[0].output_shape)], ridge=ridge,
                          smoothness=.025, prior=rng.uniform(size=shape),
                          batch=SceneFFTBatch(ops) if fast else None)


@pytest.mark.parametrize('rgb', [False, True])
@pytest.mark.parametrize('fast', [False, True])
def test_residual_fusion_matches_objective_gradient(rgb, fast):
    p = problem(111, rgb, fast)
    x = np.random.default_rng(112).uniform(size=p.shape)
    value, g = objective_gradient(p, x)
    assert value == pytest.approx(p.objective(x), rel=1e-12)
    np.testing.assert_allclose(g, p.gradient(x), atol=1e-12)


@pytest.mark.parametrize('seed', [31, 32, 33])
@pytest.mark.parametrize('rgb', [False, True])
@pytest.mark.parametrize('positive_start', [False, True])
def test_scaled_lbfgsb_independent_dense_nonnegative_oracle(seed, rgb, positive_start):
    p = problem(seed, rgb, True)
    oracle, _ = dense_oracle(p)
    x, info = solve(p, x0=np.ones(p.shape)*5 if positive_start else None, tolerance=1e-5)
    assert info['converged'] and info['feasible']
    assert np.linalg.norm(x-oracle) <= info['absolute_solution_error_bound']+1e-10
    assert p.objective(x)-p.objective(oracle) <= info['objective_gap_upper_bound']+1e-10


def test_budget_and_expired_deadline_remain_incomplete():
    p = problem(72, True, True, ridge=.00001)
    _, info = solve(p, maxiter=1, tolerance=1e-12)
    assert not info['converged'] and info['reason'] == 'iteration_budget'
    x, info = solve(p, deadline=time.monotonic()-1)
    assert not info['converged'] and info['reason'] == 'wall_budget'
    assert info['n_iter'] == 0 and np.all(x == 0)


def test_optimizer_success_cannot_replace_distance_certificate(monkeypatch):
    import tools.scene_lbfgsb as solver
    from types import SimpleNamespace
    monkeypatch.setattr(solver, 'minimize', lambda *a, **k: SimpleNamespace(success=True, message='success'))
    _, info = solver.solve(problem(91))
    assert not info['converged'] and info['reason'] == 'optimizer_stop'


@pytest.mark.hardware
@pytest.mark.parametrize('factor', [1, 2])
def test_cuda_fused_cell_residual_and_solver_match_cpu(factor):
    import torch
    from tools.scene_study import CellBasisOperator, CellFFTBatch
    if not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    rng = np.random.default_rng(912)
    op = SceneDetectorOperator((8, 8, 3), (np.ones((3, 4))/12,), 2, (0, 0), (4, 4),
                               cfa_pattern='GBRG', cfa_offset_xy=(1, 1),
                               valid_mask=rng.uniform(size=(4, 4)) > .2)
    ops = [CellBasisOperator(op, factor)]
    images = [rng.normal(.5, 1., op.output_shape)]
    images[0][~op.valid_mask] = np.nan
    cpu = SceneQuadratic(ops, images, [1.], ridge=.1, smoothness=.01)
    gpu = SceneQuadratic(ops, images, [1.], ridge=.1, smoothness=.01,
                         batch=CellFFTBatch(ops, device='cuda'))
    x = rng.uniform(size=cpu.shape)
    value, gradient = objective_gradient(gpu, x)
    assert value == pytest.approx(cpu.objective(x), rel=1e-12)
    np.testing.assert_allclose(gradient, cpu.gradient(x), atol=1e-12)
    a, ia = solve(cpu)
    b, ib = solve(gpu)
    assert ia['converged'] and ib['converged']
    assert cpu.certificate(b)['relative_solution_error_bound'] <= 1e-5
    assert np.linalg.norm(a-b) <= ia['absolute_solution_error_bound']+ib['absolute_solution_error_bound']
