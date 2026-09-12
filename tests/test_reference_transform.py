"""Prepared reference transforms cannot become stale or alter registration."""
from dataclasses import replace
import numpy as np
import pytest
from scipy.ndimage import gaussian_filter, shift
from planetrecon.backends.base import Backend, DeviceReport
from planetrecon.backends.cpu import CPUBackend
from planetrecon.pipeline.align import phase_correlation_shift


@pytest.fixture(params=['cpu', pytest.param('cuda', marks=pytest.mark.hardware)])
def backend(request):
    if request.param == 'cpu':
        return CPUBackend(threads=2)
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    from planetrecon.backends.torch_accel import TorchBackend
    return TorchBackend()


def scene(shape=(65, 81)):
    return gaussian_filter(np.random.default_rng(314).normal(size=shape), 1.5)


def test_prepared_reference_is_owned_and_read_only(backend):
    original = scene()
    reference = backend.prepare_reference(original)
    frame = shift(original, (.35, -1.2), order=1, mode='wrap')
    expected = backend.phase_correlation(original.copy(), frame)
    original.fill(0.)
    assert not reference.flags.writeable
    assert not np.shares_memory(reference, original)
    np.testing.assert_array_equal(backend.phase_correlation(reference, frame), expected)
    with pytest.raises(ValueError):
        reference[0, 0] = 99.


def test_prepared_reference_replacement_and_unprepared_mutation(backend):
    first = backend.prepare_reference(scene())
    frame = shift(first, (-.4, .7), order=1, mode='wrap')
    second = backend.prepare_reference(scene((48, 72)))
    np.testing.assert_array_equal(backend.phase_correlation(second, second), (0., 0.))
    # An old snapshot or a different mutable array must not use the new spectrum.
    np.testing.assert_allclose(backend.phase_correlation(first, frame),
                               phase_correlation_shift(first, frame), atol=1e-10, rtol=0)
    changed = first.copy()
    changed[:] = np.roll(changed, 4, axis=1)
    np.testing.assert_allclose(backend.phase_correlation(changed, frame),
                               phase_correlation_shift(changed, frame), atol=1e-10, rtol=0)


def test_cpu_prepared_reference_removes_repeated_forward_transforms(monkeypatch):
    calls = []
    transform = np.fft.fft2
    def record(array, *args, **kwargs):
        calls.append(array.shape)
        return transform(array, *args, **kwargs)
    monkeypatch.setattr(np.fft, 'fft2', record)
    backend = Backend()
    reference = backend.prepare_reference(scene())
    for i in range(5):
        np.testing.assert_array_equal(backend.phase_correlation(reference, np.roll(reference, i, 1)), (i, 0.))
    assert len(calls) == 6  # One fixed reference, then one forward transform per frame.


@pytest.mark.parametrize('fail_at', [1, 2])
def test_reference_preparation_failure_keeps_cpu_stack_and_template(monkeypatch, tmp_path, fail_at):
    import planetrecon.pipeline.baseline as baseline
    from planetrecon.pipeline.preprocess_cache import preprocess_source
    from planetrecon.io.ser import SERSource
    from test_local_alignment import capture, config
    cfg = config()
    with SERSource(capture(tmp_path)) as source:
        preprocess_source(source, cfg)
        expected = baseline.stack_source(source, cfg)
        class FailingCUDA(CPUBackend):
            name = 'cuda'
            calls = 0
            def prepare_reference(self, reference):
                self.calls += 1
                if self.calls == fail_at:
                    raise RuntimeError('reference allocation failed')
                return super().prepare_reference(reference)
        backend = FailingCUDA(threads=2)
        monkeypatch.setattr(baseline, 'select_backend', lambda *args, **kwargs:
                            (backend, DeviceReport('gpu', 'cuda', ['cpu', 'gpu'], False)))
        actual = baseline.stack_source(source, replace(cfg, device='gpu'))
    assert actual.backend == 'cpu'
    assert actual.provenance['device_report']['execution_history'] == ['cuda', 'cpu']
    for key in ('image', 'coverage', 'validity'):
        np.testing.assert_array_equal(getattr(actual, key), getattr(expected, key))
