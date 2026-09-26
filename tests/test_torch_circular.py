"""GPU alignment keeps the CPU algorithm, brightness and support semantics."""
from dataclasses import replace

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter, map_coordinates

from planetrecon.pipeline.circular_align import CircularMultiscaleRegistration, _peak
from planetrecon.pipeline.local_align import LocalRegistration
from test_local_alignment import capture, config


@pytest.fixture
def torch_runtime():
    torch = pytest.importorskip('torch')
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield torch
    torch.set_num_threads(previous)


@pytest.mark.parametrize('nearest', [False, True])
def test_sampling_matches_detector_support(torch_runtime, nearest):
    from planetrecon.backends.torch_circular import sample
    rng = np.random.default_rng(28)
    image = rng.normal(size=(2, 17, 21))
    y, x = rng.uniform(-2, 19, size=(3, 11)), rng.uniform(-2, 23, size=(3, 11))
    expected = np.stack([map_coordinates(a, [y, x], order=1, prefilter=False,
                         mode='nearest' if nearest else 'grid-constant') for a in image])
    actual = sample(torch_runtime.tensor(image), torch_runtime.tensor(y), torch_runtime.tensor(x), nearest=nearest)
    np.testing.assert_allclose(actual.numpy(), expected, atol=2e-14, rtol=2e-14)


def test_batched_peak_gates_match_cpu(torch_runtime):
    from planetrecon.backends.torch_circular import peaks
    y, x = np.indices((7, 7))-3
    costs = [.95-.015*((x-dx)**2+(y-dy)**2) for dx, dy in ((0, 0), (.7, -.3), (-1.9, 1.5), (3, 3))]
    costs += [np.ones((7, 7)), .95-.02*(x+y)**2, np.full((7, 7), np.nan)]
    offsets, valid, _ = peaks(torch_runtime.tensor(np.stack(costs)))
    for i, cost in enumerate(costs):
        expected = _peak(cost)
        assert bool(valid[i]) == (expected is not None)
        if expected is not None:
            np.testing.assert_allclose(offsets[i].numpy(), expected, atol=1e-13)


@pytest.mark.parametrize('kind', ['motion', 'brightness', 'blur', 'noise', 'ambiguous', 'small'])
def test_resident_algorithm_matches_cpu(torch_runtime, kind):
    from planetrecon.backends.torch_circular import TorchCircularRegistration
    rng = np.random.default_rng(58)
    shape = (20, 24) if kind == 'small' else (80, 96)
    ref = gaussian_filter(rng.normal(size=shape), 1.5)*100+300
    y, x = np.indices(shape)
    frame = map_coordinates(ref, [y+.3, x-.8*np.sin(y/15)], order=3, mode='reflect')
    if kind == 'brightness':
        frame = ref*.8+25
    elif kind == 'blur':
        frame = gaussian_filter(ref, 1.)
    elif kind == 'noise':
        frame = gaussian_filter(rng.normal(size=shape), 1.5)*100+300
    elif kind == 'ambiguous':
        ref, frame = 300+100*np.sin((x+y)/4), 300+100*np.sin((x+y-1)/4)
    matcher = CircularMultiscaleRegistration(ref)
    shift = (.2, -.1)
    expected = matcher.displacement(frame, shift)
    engine = TorchCircularRegistration(matcher, device='cpu', chunk_bytes=2*1024**2)
    actual = engine.displacement(LocalRegistration.proxy(frame), shift, lambda: None).numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)


def test_resident_cancellation(torch_runtime):
    from planetrecon.backends.torch_circular import TorchCircularRegistration
    ref = gaussian_filter(np.random.default_rng(23).normal(size=(80, 96)), 1.)
    matcher = CircularMultiscaleRegistration(ref)
    engine = TorchCircularRegistration(matcher, device='cpu')
    def cancel():
        raise InterruptedError('cancelled')
    with pytest.raises(InterruptedError):
        engine.displacement(LocalRegistration.proxy(ref), (0., 0.), cancel)


@pytest.mark.hardware
@pytest.mark.parametrize('fused', [False, True])
def test_cuda_resident_and_fused_match_cpu(torch_runtime, fused):
    from planetrecon.backends.torch_circular import TorchCircularRegistration
    if not torch_runtime.cuda.is_available():
        pytest.skip('CUDA unavailable')
    y, x = np.indices((160, 192))
    ref = gaussian_filter(np.random.default_rng(204).normal(size=y.shape), 1.5)*100+300
    frame = map_coordinates(ref, [y+.25, x-.8*np.sin(y/25)], order=3, mode='reflect')
    matcher = CircularMultiscaleRegistration(ref)
    engine = TorchCircularRegistration(matcher)
    if fused:
        if engine.fused_scores is None:
            pytest.skip('Triton unavailable')
    else:
        for layer in engine.layers:
            layer['fused'] = False
            layer['chunk'] = 4
    expected = matcher.displacement(frame, (0., 0.))
    actual = engine.displacement(LocalRegistration.proxy(frame), (0., 0.), lambda: None).cpu().numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-9, rtol=1e-9)


@pytest.mark.hardware
@pytest.mark.parametrize('window', [23, 77, 127])
def test_fused_correlations_across_window_sizes(torch_runtime, window):
    if not torch_runtime.cuda.is_available():
        pytest.skip('CUDA unavailable')
    pytest.importorskip('triton')
    from planetrecon.backends.triton_correlation import scores
    from planetrecon.pipeline.circular_align import _costs
    ref = gaussian_filter(np.random.default_rng(94).normal(size=(168, 180)), 1.5)
    layer = LocalRegistration(ref, window=window, step=window//2, circular=True)
    image = np.roll(layer.ref, 1, axis=1)
    y, x = np.repeat(layer.ys, len(layer.xs)), np.tile(layer.xs, len(layer.ys))
    tensor = lambda a: torch_runtime.as_tensor(a, device='cuda')
    actual = scores(tensor(image), tensor(y), tensor(x), tensor(layer.templates),
                    tensor(layer.strength), tensor(layer.weight)).cpu().numpy()
    expected = np.stack([_costs(image, cy, cx, layer.templates[k], layer.weight, layer.strength[k])
                         for k, (cy, cx) in enumerate(zip(y, x))])
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-10)


@pytest.mark.hardware
@pytest.mark.parametrize('color', [0, 8, 100])
@pytest.mark.parametrize('failure_stage', ['matching', 'backprojection'])
def test_gpu_stack_and_midrun_fallback_preserve_output(torch_runtime, tmp_path, monkeypatch, color, failure_stage):
    if not torch_runtime.cuda.is_available():
        pytest.skip('CUDA unavailable')
    from planetrecon.io.ser import SERSource
    from planetrecon.pipeline.baseline import stack_source
    from planetrecon.pipeline.preprocess_cache import preprocess_source
    from planetrecon.backends.torch_accel import TorchBackend
    from planetrecon.backends.torch_circular import TorchCircularRegistration
    with SERSource(capture(tmp_path, color)) as source:
        cpu_cfg = config(alignment_method='circular_multiscale')
        preprocess_source(source, cpu_cfg)
        cpu = stack_source(source, cpu_cfg)
        gpu = stack_source(source, replace(cpu_cfg, device='gpu'))
        assert gpu.backend == 'cuda'
        np.testing.assert_allclose(gpu.image, cpu.image, rtol=1e-9, atol=1e-8)
        np.testing.assert_allclose(gpu.coverage, cpu.coverage, rtol=1e-9, atol=1e-8)
        target, name = ((TorchBackend, 'backproject') if failure_stage == 'backprojection'
                        else (TorchCircularRegistration, 'displacement'))
        original = getattr(target, name)
        calls = 0
        def failure(self, *args):
            nonlocal calls
            calls += 1
            if calls == 3:
                raise RuntimeError('injected CUDA failure')
            return original(self, *args)
        monkeypatch.setattr(target, name, failure)
        fallback = stack_source(source, replace(cpu_cfg, device='gpu'))
        assert fallback.backend == 'cpu' and any('injected CUDA' in w for w in fallback.warnings)
        np.testing.assert_allclose(fallback.image, cpu.image, rtol=1e-9, atol=1e-8)
        np.testing.assert_allclose(fallback.coverage, cpu.coverage, rtol=1e-9, atol=1e-8)
