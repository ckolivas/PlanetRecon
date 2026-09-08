import numpy as np
import pytest
from scipy import fft
from tools.scene_parallel_reference import ParallelSceneBatch
from tools.scene_quadratic import SceneQuadratic, solve
from test_scene_fft import operators


@pytest.mark.parametrize('rgb', [False, True])
@pytest.mark.parametrize('workers', [1, 2, 8])
def test_parallel_reference_preserves_ordered_cpu_results(rgb, workers):
    ops = operators(rgb, True)*3
    batch = ParallelSceneBatch(ops, workers=workers)
    rng = np.random.default_rng(83)
    x = rng.normal(size=ops[0].scene_shape)
    images = [rng.normal(size=o.output_shape) for o in ops]
    weights = [rng.uniform(size=o.output_shape) for o in ops]
    for actual, op in zip(batch.forward(x), ops):
        np.testing.assert_array_equal(actual, op.forward(x))
    expected = np.zeros_like(x)
    for op, y in zip(ops, images): expected += op.adjoint(y)
    np.testing.assert_array_equal(batch.adjoint(images), expected)
    expected = np.zeros_like(x)
    for op, w in zip(ops, weights): expected += op.adjoint(w*op.forward(x))
    np.testing.assert_array_equal(batch.normal(x, weights), expected)
    assert batch.peak_pending <= workers


def test_worker_failure_propagates_and_fft_workers_stay_bounded():
    batch = ParallelSceneBatch(operators(False, True), workers=2)
    def fail(value):
        assert fft.get_workers() == 1
        if value == 2: raise RuntimeError('worker failed')
        return value
    with pytest.raises(RuntimeError, match='worker failed'):
        list(batch._ordered(fail, range(8)))
    assert list(batch._ordered(lambda x: x, range(8))) == list(range(8))


def test_parallel_certificate_and_solver_match_reference():
    ops = operators(False, True)
    rng = np.random.default_rng(9)
    images = [rng.normal(size=op.output_shape) for op in ops]
    cpu = SceneQuadratic(ops, images, [1.]*3, ridge=.1)
    parallel = SceneQuadratic(ops, images, [1.]*3, ridge=.1, batch=ParallelSceneBatch(ops))
    x, a = solve(cpu)
    y, b = solve(parallel)
    assert a['converged'] and b['converged']
    assert cpu.certificate(y)['relative_solution_error_bound'] <= 1e-5
    assert np.linalg.norm(x-y) <= a['absolute_solution_error_bound']+b['absolute_solution_error_bound']
