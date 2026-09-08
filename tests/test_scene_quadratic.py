import numpy as np
import pytest
from scipy.optimize import nnls
from planetrecon.operators import SceneDetectorOperator
from tools.scene_quadratic import SceneQuadratic, solve


def dense_oracle(problem):
    n = int(np.prod(problem.shape))
    basis = np.eye(n).reshape((n,)+problem.shape)
    matrices, targets = [], []
    for op, image, weight in zip(problem.operators, problem.images, problem.weights):
        matrix = np.stack([op.forward(v).ravel() for v in basis], axis=1)
        matrices.append(np.sqrt(weight.ravel())[:, None]*matrix)
        targets.append(np.sqrt(weight.ravel())*image.ravel())
    matrices.append(np.sqrt(problem.ridge)*np.eye(n))
    targets.append(np.sqrt(problem.ridge)*problem.prior.ravel())
    for axis in (0, 1):
        matrix = np.stack([np.diff(v, axis=axis).ravel() for v in basis], axis=1)
        matrices.append(np.sqrt(problem.smoothness)*matrix)
        targets.append(np.zeros(matrix.shape[0]))
    matrix, target = np.vstack(matrices), np.concatenate(targets)
    result, _ = nnls(matrix, target, maxiter=1000)
    gradient = matrix.T@(matrix@result-target)
    residual = np.where(result > 0, gradient, np.minimum(gradient, 0.))
    assert np.linalg.norm(residual) < 1e-11
    return result.reshape(problem.shape), matrix


@pytest.mark.parametrize('seed', [11, 12, 13])
@pytest.mark.parametrize('rgb', [False, True])
@pytest.mark.parametrize('start', ['zero', 'positive', 'negative'])
def test_dense_oracle_active_constraints_multiple_starts(seed, rgb, start):
    rng = np.random.default_rng(seed)
    shape = (6, 6)+( (3,) if rgb else () )
    psf = rng.uniform(size=(2, 3)); psf /= psf.sum()
    ops = [SceneDetectorOperator(shape, (psf,), 2, (0, 0), (3, 3),
            shifts_xy=((s, -.25),), cfa_pattern='RGGB' if rgb else None) for s in (0., .4)]
    images = [rng.normal(.1, 1., size=op.output_shape) for op in ops]
    variances = [rng.uniform(.5, 2., size=op.output_shape) for op in ops]
    problem = SceneQuadratic(ops, images, variances, ridge=.1, smoothness=.025)
    oracle, matrix = dense_oracle(problem)
    eigen = np.linalg.eigvalsh(matrix.T@matrix)
    assert eigen.min() >= problem.ridge-1e-12
    assert eigen.max() <= problem.lipschitz+1e-12
    assert np.linalg.eigvalsh(np.diag(problem.majorizer.ravel())-matrix.T@matrix).min() >= -1e-12
    x0 = {'zero': np.zeros(shape), 'positive': np.ones(shape)*5, 'negative': -np.ones(shape)}[start]
    result, info = solve(problem, x0=x0, tolerance=1e-7)
    assert info['converged'] and info['feasible']
    error = np.linalg.norm(result-oracle)
    assert error <= info['absolute_solution_error_bound']+1e-10
    assert problem.objective(result)-problem.objective(oracle) <= info['objective_gap_upper_bound']+1e-12
    np.testing.assert_allclose(result, oracle, atol=2e-7)
    assert (oracle < 1e-12).any()
    # The entire extended scene is solved, including detector-unobserved margins.
    assert result.shape == problem.shape


def test_independent_gradient_and_budget_failure():
    rng = np.random.default_rng(4)
    op = SceneDetectorOperator((8, 8), (np.ones((3, 3))/9,), 1, (2, 2), (4, 4))
    p = SceneQuadratic([op], [rng.normal(size=(4, 4))], [2.], ridge=.001, smoothness=.01)
    x, d = rng.uniform(size=(8, 8)), rng.normal(size=(8, 8))
    eps = 1e-5
    np.testing.assert_allclose((p.objective(x+eps*d)-p.objective(x-eps*d))/(2*eps), np.vdot(p.gradient(x), d), rtol=1e-8)
    result, info = solve(p, maxiter=1, tolerance=1e-12)
    assert info['status'] == 'incomplete' and not info['converged']
    assert info['reason'] == 'iteration_budget'
    assert info['relative_solution_error_bound'] > 1e-12


def test_masked_invalid_measurements_and_expired_deadline():
    import time
    mask = np.ones((4, 4), bool); mask[0] = False
    op = SceneDetectorOperator((4, 4), (np.ones((1, 1)),), 1, (0, 0), (4, 4), valid_mask=mask)
    image = np.ones((4, 4)); image[0] = np.nan
    var = np.ones((4, 4)); var[0] = np.nan
    p = SceneQuadratic([op], [image], [var], ridge=.1)
    _, info = solve(p, deadline=time.monotonic()-1)
    assert not info['converged'] and info['reason'] == 'wall_budget' and info['n_iter'] == 0
    result, info = solve(p)
    np.testing.assert_allclose(result[mask], 1/1.1)
    np.testing.assert_allclose(result[~mask], 0., atol=1e-14)
    assert info['converged']


@pytest.mark.parametrize('ridge', [0., -1., np.nan])
def test_certificate_requires_positive_ridge(ridge):
    op = SceneDetectorOperator((2, 2), (np.ones((1, 1)),), 1, (0, 0), (2, 2))
    with pytest.raises(ValueError):
        SceneQuadratic([op], [np.zeros((2, 2))], [1.], ridge=ridge)


def test_diagonal_scaling_preserves_objective_and_certificate():
    rng = np.random.default_rng(91)
    op = SceneDetectorOperator((8, 8), (np.ones((3, 3))/9,), 1, (0, 0), (8, 8))
    variances = np.ones((8, 8)); variances[:, :4] = 100.
    p = SceneQuadratic([op], [rng.normal(.5, 1., (8, 8))], [variances], ridge=.01)
    diagonal, a = solve(p, scaling='diagonal', tolerance=1e-7)
    global_image, b = solve(p, scaling='global', tolerance=1e-7)
    assert a['converged'] and b['converged']
    assert np.linalg.norm(diagonal-global_image) <= a['absolute_solution_error_bound']+b['absolute_solution_error_bound']
