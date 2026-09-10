"""Local CUDA moment parity and complete-frame retry/cancellation controls."""
import numpy as np
import pytest
from tools.cfa_local_transport_probe import LocalQuadraticAccumulator
from tools.cfa_local_gpu import CudaLocalAccumulator

NAMES = ('gram', 'noise', 'rhs', 'sums', 'weights', 'variance_sum')


def frames(shape):
    y, x = np.indices(shape, dtype=float)
    raw = [30+np.sin(x/2+i*.2)+np.cos(y/3-i*.1) for i in range(3)]
    maps = [(.2*i+.12*np.sin(y/3), -.15*i+.1*np.cos(x/4)) for i in range(3)]
    return raw, maps


def test_unavailable_cuda_retries_whole_frame_once(monkeypatch):
    raw, maps = frames((10, 12))
    gpu = CudaLocalAccumulator(raw[0].shape, 'RGGB')
    cpu = LocalQuadraticAccumulator(raw[0].shape, 'RGGB')
    calls = []
    def fail():
        calls.append(1)
        raise RuntimeError('CUDA unavailable')
    monkeypatch.setattr(gpu, '_device', fail)
    for a, b in zip(raw, maps):
        gpu.add(a, b, 1.); cpu.add(a, b, 1.)
    assert calls == [1]
    assert gpu.execution_history == ['cpu']*3
    assert gpu.fallback_frame == 0 and not gpu.gpu_enabled
    for name in NAMES: np.testing.assert_array_equal(getattr(gpu, name), getattr(cpu, name))


def test_publication_refuses_an_invalid_late_array_before_any_change():
    raw, maps = frames((10, 12))
    model = LocalQuadraticAccumulator(raw[0].shape, 'RGGB')
    model.add(raw[0], maps[0], 1.)
    saved = {name: getattr(model, name).copy() for name in NAMES}
    staged = model.stage(raw[1], maps[1], 1.)
    staged['variance_sum'][0, 0, 0] = np.nan
    with pytest.raises(ValueError, match='variance_sum'): model.publish(staged)
    assert model.n_used == 1
    for name in NAMES: np.testing.assert_array_equal(saved[name], getattr(model, name))


@pytest.mark.hardware
@pytest.mark.parametrize('pattern', ['RGGB', 'GRBG', 'GBRG', 'BGGR'])
@pytest.mark.parametrize('shape', [(18, 22), (2, 8), (8, 2), (3, 3)])
def test_cuda_parity_repeatability_and_support(pattern, shape):
    import torch
    if not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    raw, maps = frames(shape)
    cpu = LocalQuadraticAccumulator(shape, pattern)
    gpu = CudaLocalAccumulator(shape, pattern, gpu_chunk_rows=17, max_neighbour_entries=1000)
    repeat = CudaLocalAccumulator(shape, pattern, gpu_chunk_rows=17, max_neighbour_entries=1000)
    for i, (a, b) in enumerate(zip(raw, maps)):
        for model in (cpu, gpu, repeat): model.add(a, b, .8+i*.1)
    assert gpu.gpu_enabled and gpu.execution_history == ['cuda']*3
    assert gpu.maximum_batch_entries <= 1000
    for name in NAMES:
        np.testing.assert_allclose(getattr(gpu, name), getattr(cpu, name), rtol=1e-13, atol=1e-10)
        np.testing.assert_array_equal(getattr(gpu, name), getattr(repeat, name))
    a, b = gpu.finish(), cpu.finish()
    np.testing.assert_allclose(a[0], b[0], rtol=0, atol=1e-10)
    np.testing.assert_allclose(a[1], b[1], rtol=0, atol=1e-12)
    np.testing.assert_array_equal(a[2], b[2])
    for name in ('image', 'coverage', 'validity'):
        np.testing.assert_array_equal(getattr(a[3], name), getattr(b[3], name))


@pytest.mark.hardware
@pytest.mark.parametrize('failure', ['runtime', 'cancel'])
def test_third_gpu_batch_failure_discards_partial_frame(monkeypatch, failure):
    import torch
    if not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    shape = (18, 22)
    raw, maps = frames(shape)
    model = CudaLocalAccumulator(shape, 'RGGB', gpu_chunk_rows=17)
    model.add(raw[0], maps[0], .8)
    saved = {name: getattr(model, name).copy() for name in NAMES}
    original = model._cuda_products
    calls = 0
    def fail(*args):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError('injected third batch failure') if failure == 'runtime' else InterruptedError('cancel')
        return original(*args)
    monkeypatch.setattr(model, '_cuda_products', fail)
    if failure == 'cancel':
        with pytest.raises(InterruptedError): model.add(raw[1], maps[1], .9)
        assert model.n_used == 1 and model.execution_history == ['cuda'] and model.gpu_enabled
        for name in NAMES: np.testing.assert_array_equal(saved[name], getattr(model, name))
        monkeypatch.setattr(model, '_cuda_products', original)
        model.add(raw[1], maps[1], .9)
    else:
        model.add(raw[1], maps[1], .9)
        assert not model.gpu_enabled and model.fallback_frame == 1
        assert model.execution_history == ['cuda', 'cpu']
    expected = CudaLocalAccumulator(shape, 'RGGB', gpu_chunk_rows=17)
    expected.add(raw[0], maps[0], .8)
    if failure == 'runtime': expected.gpu_enabled = False
    expected.add(raw[1], maps[1], .9)
    for name in NAMES: np.testing.assert_array_equal(getattr(model, name), getattr(expected, name))


@pytest.mark.hardware
def test_gpu_budget_failure_retries_on_cpu_without_counting_twice():
    import torch
    if not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    raw, maps = frames((10, 12))
    gpu = CudaLocalAccumulator(raw[0].shape, 'RGGB', max_gpu_array_bytes=1)
    cpu = LocalQuadraticAccumulator(raw[0].shape, 'RGGB')
    gpu.add(raw[0], maps[0], .8); cpu.add(raw[0], maps[0], .8)
    assert gpu.n_used == 1 and gpu.execution_history == ['cpu']
    assert 'budget' in gpu.fallback_reason
    for name in NAMES: np.testing.assert_array_equal(getattr(gpu, name), getattr(cpu, name))


def test_cpu_only_torch_is_treated_as_unavailable_without_device_initialization(monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(torch.cuda, 'get_device_properties', lambda *_: pytest.fail('initialized unavailable CUDA'))
    raw, maps = frames((10, 12))
    model = CudaLocalAccumulator(raw[0].shape, 'RGGB')
    model.add(raw[0], maps[0], 1.)
    assert model.execution_history == ['cpu'] and not model.gpu_enabled


@pytest.mark.parametrize('kind', ['translation', 'shear', 'compression', 'expansion', 'crossed'])
def test_detector_window_finds_every_reference_space_neighbour(kind):
    from scipy.spatial import cKDTree
    from tools.cfa_local_gpu import detector_window
    from tools.cfa_local_transport_probe import inverse_pull_map
    shape = (24, 28)
    y, x = np.indices(shape, dtype=float)
    if kind == 'translation': shift = (np.full(shape, 1.3), np.full(shape, -2.2))
    elif kind == 'shear': shift = (.7*np.sin(y/3), .1*np.cos(x/4))
    elif kind == 'compression': shift = (-.6*(x-14), -.5*(y-12))
    elif kind == 'expansion': shift = (.6*(x-14), .5*(y-12))
    else: shift = (.45*(y-12), -.45*(x-14))
    positions, valid = inverse_pull_map(shift, shape)
    observed_ids = np.flatnonzero(valid)
    tree = cKDTree(positions[valid])
    dx, dy, ox, oy = detector_window(shift, shape)
    for ty in range(shape[0]):
        for tx in range(shape[1]):
            ids = observed_ids[tree.query_ball_point([tx, ty], 4., p=np.inf)]
            # Strict window: points at exactly radius four have zero kernel weight.
            if len(ids):
                uv = positions.reshape(-1, 2)[ids]-[tx, ty]
                ids = ids[(np.abs(uv) < 4).all(axis=-1)]
            sx = int(np.floor(tx+dx[ty, tx]))+ox
            sy = int(np.floor(ty+dy[ty, tx]))+oy
            mask = (sx >= 0) & (sx < shape[1]) & (sy >= 0) & (sy < shape[0])
            sx, sy = sx[mask], sy[mask]
            uv = positions[sy, sx]-[tx, ty]
            mask = valid[sy, sx] & (np.abs(uv) < 4).all(axis=-1)
            np.testing.assert_array_equal(np.sort(ids), np.sort(sy[mask]*shape[1]+sx[mask]))
