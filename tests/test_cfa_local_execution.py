"""Execution controls for the qualified local-map interpolator."""
import numpy as np
import pytest
from scipy.spatial import cKDTree

import tools.cfa_local_transport_probe as module
from tools.cfa_local_transport_probe import LocalQuadraticAccumulator, inverse_pull_map


def inputs(shape=(14, 18)):
    y, x = np.indices(shape, dtype=float)
    shift = (.27+.12*np.sin(y/3), -.19+.1*np.cos(x/4))
    return 30+np.sin(x/2)+np.cos(y/3), shift


@pytest.mark.parametrize('color', ['RGGB', 'GRBG', 'GBRG', 'BGGR'])
@pytest.mark.parametrize('chunk', [1, 17, 256])
def test_batched_moments_match_original_scalar_order(color, chunk):
    raw, shift = inputs()
    model = LocalQuadraticAccumulator(raw.shape, color, chunk_rows=chunk, max_neighbour_entries=1000)
    model.add(raw, shift, .8)
    positions, valid = inverse_pull_map(shift, raw.shape)
    points, values, labels = positions[valid], raw[valid], model.labels[valid]
    tree = cKDTree(points)
    expected = [np.zeros_like(a) for a in (model.gram, model.noise, model.rhs)]
    for n, target in enumerate(model.targets):
        ids = np.array(sorted(tree.query_ball_point(target, 4., p=np.inf)), dtype=int)
        u, v = ((points[ids]-target)/4).T
        phi = np.stack([np.ones_like(u), u, v, u*u, u*v, v*v], axis=-1)
        kernel = .8*np.maximum(1-u*u, 0)**3*np.maximum(1-v*v, 0)**3
        for c, name in enumerate('RGB'):
            weight = kernel*(labels[ids] == name)
            expected[0][n, c] = (phi.T*weight)@phi
            expected[1][n, c] = (phi.T*weight**2)@phi
            expected[2][n, c] = (phi.T*weight)@values[ids]
    for actual, reference in zip((model.gram, model.noise, model.rhs), expected):
        np.testing.assert_allclose(actual, reference, rtol=1e-14, atol=1e-12)
    reference_model = LocalQuadraticAccumulator(raw.shape, color, chunk_rows=chunk)
    reference_model.gram, reference_model.noise, reference_model.rhs = expected
    reference_model.weights = model.weights.copy()
    reference_model.sums = model.sums.copy()
    reference_model.variance_sum = model.variance_sum.copy()
    reference_model.n_used = model.n_used
    actual, reference = model.finish(), reference_model.finish()
    np.testing.assert_allclose(actual[0], reference[0], rtol=0, atol=1e-10)
    np.testing.assert_allclose(actual[1], reference[1], rtol=0, atol=1e-12)
    np.testing.assert_array_equal(actual[2], reference[2])


def state(model):
    return [v.copy() for v in (model.gram, model.noise, model.rhs, model.sums, model.weights, model.variance_sum)]


@pytest.mark.parametrize('phase', ['inverse', 'moments', 'fit'])
def test_cancel_during_work_preserves_previous_frame_and_retry(monkeypatch, phase):
    raw, shift = inputs()
    model = LocalQuadraticAccumulator(raw.shape, 'RGGB', chunk_rows=7)
    model.add(raw, shift, .8)
    previous = state(model)
    calls = 0
    def cancelled():
        return calls >= 2
    name = {'inverse': 'map_coordinates', 'moments': 'cKDTree', 'fit': 'eigvalsh'}[phase]
    if phase == 'moments':
        original = module.cKDTree
        class Tree:
            def __init__(self, points):
                self.tree = original(points)
            def query_ball_point(self, *args, **kwargs):
                nonlocal calls
                if not kwargs.get('return_length'):
                    calls += 1
                return self.tree.query_ball_point(*args, **kwargs)
        monkeypatch.setattr(module, name, Tree)
    else:
        owner = module.np.linalg if phase == 'fit' else module
        original = getattr(owner, name)
        def counted(*args, **kwargs):
            nonlocal calls
            result = original(*args, **kwargs)
            calls += 1
            return result
        monkeypatch.setattr(owner, name, counted)
    model.should_cancel = cancelled
    with pytest.raises(InterruptedError):
        model.finish() if phase == 'fit' else model.add(raw, shift, 1.1)
    assert model.n_used == 1
    for saved, actual in zip(previous, state(model)):
        np.testing.assert_array_equal(saved, actual)
    model.should_cancel = None
    model.add(raw, shift, 1.1)
    expected = LocalQuadraticAccumulator(raw.shape, 'RGGB', chunk_rows=7)
    expected.add(raw, shift, .8); expected.add(raw, shift, 1.1)
    for a, b in zip(model.finish()[:3], expected.finish()[:3]):
        np.testing.assert_array_equal(a, b)


def test_array_budget_refuses_before_large_allocations(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('array allocation preceded budget check')
    monkeypatch.setattr(module.np, 'indices', forbidden)
    with pytest.raises(MemoryError, match='array bytes'):
        LocalQuadraticAccumulator((424, 656), 'RGGB', max_array_bytes=1024)


def test_neighbour_budget_failure_preserves_previous_frame():
    raw, shift = inputs()
    model = LocalQuadraticAccumulator(raw.shape, 'RGGB')
    model.add(raw, shift, .8)
    previous = state(model)
    model.max_neighbour_entries = 1
    with pytest.raises(MemoryError, match='neighbour'):
        model.add(raw, shift, 1.1)
    for saved, actual in zip(previous, state(model)):
        np.testing.assert_array_equal(saved, actual)
    assert model.n_used == 1
