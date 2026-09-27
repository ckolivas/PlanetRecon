"""CUDA screening preserves detector calibration and CPU selection semantics."""
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from planetrecon.calibration import Calibration
from planetrecon.detector import cfa_labels
from planetrecon.io.source import ArraySource
from planetrecon.pipeline import preprocess as module
from planetrecon.reconstruction import ReconstructionConfig
from test_preprocessing import planet


@pytest.fixture
def cuda():
    torch = pytest.importorskip('torch')
    from planetrecon.backends.base import probe_torch_cuda
    if probe_torch_cuda().selected != 'gpu':
        pytest.skip('supported CUDA device required')
    return torch


@pytest.mark.parametrize('color', ['mono', 'RGB', 'BGR', 'RGGB', 'BGGR', 'GRBG', 'GBRG'])
@pytest.mark.parametrize('calibrated', [False, True])
def test_cuda_measurements_and_selections_match_cpu(cuda, color, calibrated):
    frames = np.stack([np.pad(planet(blur=.5+i*.07), ((0, 1), (0, 1)), mode='edge') for i in range(24)])
    if color in ('RGB', 'BGR'):
        frames = frames[..., None] * np.array([.8, 1., 1.2])
    elif color != 'mono':
        labels = cfa_labels(*frames.shape[1:], color)
        frames *= np.where(labels == 'R', .8, np.where(labels == 'B', 1.2, 1.))
    frames[0] = 0
    frames[1] = np.roll(frames[1], 45, axis=1)
    frames[2, 0, 0] = np.nan
    frames[3, :10] = 65535
    frames[4, 0, 0] = 60000
    frames[5, 0, 0] = np.inf
    shape = frames.shape[1:]
    cal = (Calibration(bias=np.full(shape, 2.), dark=np.full(shape, 1.),
                       flat=np.linspace(.8, 1.1, int(np.prod(shape))).reshape(shape),
                       gain_e_per_adu=1.7, saturate_adu=50000.) if calibrated else None)
    source = ArraySource(frames, color_mode=color)
    cfg = ReconstructionConfig(device='cpu', threads=4, batch_frames=8)
    expected = module.screen_source(source, cfg, cal)
    events = []
    actual = module.screen_source(source, replace(cfg, device='gpu'), cal, on_device=events.append)
    assert actual.summary['execution']['cuda_frames'] == len(frames)
    assert events[0]['backend'] == 'cuda'
    np.testing.assert_allclose(actual.measurements, expected.measurements, rtol=2e-12, atol=1e-10)
    np.testing.assert_array_equal(actual.accepted, expected.accepted)
    assert actual.summary['rejected_indices_by_reason'] == expected.summary['rejected_indices_by_reason']
    assert actual.best_reference_index == expected.best_reference_index
    for mode in ('frame_count', 'quality_range'):
        for percent in (1, 25, 50, 75, 100):
            np.testing.assert_array_equal(module.best_frame_mask(actual, percent, mode),
                                          module.best_frame_mask(expected, percent, mode))


@pytest.mark.parametrize('dtype', ['u1', '>u2', 'f4'])
def test_cuda_raw_types_and_saturation_disabled(cuda, dtype):
    frames = np.stack([planet(blur=.7+i*.1) for i in range(12)]).astype(dtype)
    frames[0] = 255
    source = ArraySource(frames, bit_depth=8)
    cfg = ReconstructionConfig(device='cpu', threads=2, reject_saturated=False)
    expected = module.screen_source(source, cfg)
    actual = module.screen_source(source, replace(cfg, device='gpu'))
    assert actual.summary['execution']['cuda_frames'] == 12
    np.testing.assert_allclose(actual.measurements, expected.measurements, rtol=2e-12, atol=1e-10)
    np.testing.assert_array_equal(actual.accepted, expected.accepted)


def test_cuda_saturation_boundary_and_float_detector_threshold(cuda):
    frames = np.zeros((3, 80, 100), dtype='u2')
    for frame, count in zip(frames, (399, 400, 401)):
        frame.ravel()[:count] = 65535
    source = ArraySource(frames)
    cfg = ReconstructionConfig(device='gpu', threads=2)
    result = module.screen_source(source, cfg)
    assert result.summary['rejected_indices_by_reason']['saturated'] == [2]
    # NumPy compares a float32 detector to this scalar in float32. Preserve
    # that detector-space threshold before doing float64 CUDA calibration.
    source = ArraySource(np.full((2, 8, 12), .1, dtype='f4'), bit_depth=32)
    cal = Calibration(saturate_adu=.100000002)
    cpu = module.screen_source(source, replace(cfg, device='cpu'), cal)
    gpu = module.screen_source(source, cfg, cal)
    assert cpu.summary['rejected_indices_by_reason']['saturated'] == [0, 1]
    assert gpu.summary['rejected_indices_by_reason'] == cpu.summary['rejected_indices_by_reason']
    source = ArraySource(np.zeros((2, 8, 12), dtype='u2'))
    result = module.screen_source(source, cfg, Calibration(saturate_adu=70000))
    assert result.summary['execution']['cuda_frames'] == 2
    assert result.summary['rejected_indices_by_reason']['saturated'] == []


def test_cuda_cancel_keeps_prefix_and_allows_rerun(cuda):
    source = ArraySource(np.stack([planet(blur=.7+i*.05) for i in range(24)]))
    cfg = ReconstructionConfig(device='auto', threads=4, batch_frames=8)
    stop = False
    def progress(done, total):
        nonlocal stop
        stop = True
    partial = module.screen_source(source, cfg, should_cancel=lambda: stop, on_progress=progress)
    assert partial.cancelled and partial.summary['n_measured'] == 8
    assert partial.summary['execution']['cuda_frames'] == 8
    assert np.isnan(partial.measurements[8:]).all()
    again = module.screen_source(source, cfg)
    reference = module.screen_source(source, replace(cfg, device='cpu'))
    np.testing.assert_allclose(again.measurements, reference.measurements, rtol=2e-12, atol=1e-10)


def test_cuda_budget_failure_uses_cpu_and_restores_allocator(cuda):
    source = ArraySource(np.stack([planet()]*4))
    previous = cuda.cuda.get_per_process_memory_fraction(0)
    cfg = ReconstructionConfig(device='gpu', max_vram_bytes=1024, threads=2)
    result = module.screen_source(source, cfg)
    assert result.summary['execution']['backend'] == 'cpu'
    assert result.summary['execution']['cuda_frames'] == 0
    assert 'using CPU' in result.summary['execution']['reason']
    assert cuda.cuda.get_per_process_memory_fraction(0) == previous


def test_unavailable_cuda_has_truthful_cpu_fallback(monkeypatch):
    monkeypatch.setattr('planetrecon.backends.base.probe_torch_cuda',
                        lambda: SimpleNamespace(selected='cpu', reason='test unavailable'))
    source = ArraySource(np.stack([planet()]*8))
    events = []
    result = module.screen_source(source, ReconstructionConfig(device='gpu', threads=2), on_device=events.append)
    assert result.summary['execution']['backend'] == 'cpu'
    assert 'test unavailable' in events[0]['reason']
    assert result.summary['n_measured'] == 8


@pytest.mark.parametrize('successful_batches', [0, 1])
def test_failed_cuda_batch_is_remeasured_on_cpu(monkeypatch, successful_batches):
    source = ArraySource(np.stack([planet(blur=.7+i*.03) for i in range(12)]))
    cfg = ReconstructionConfig(device='cpu', threads=2, batch_frames=4)
    expected = module.screen_source(source, cfg)
    class Failing:
        closed = False
        calls = 0
        def prepare(self, *args):
            self.calls += 1
            if self.calls <= successful_batches:
                return args[0]
            raise RuntimeError('injected CUDA allocation failure')
        @staticmethod
        def measure(raw):
            return module._measure_observation(raw, color='mono', bit_depth=16,
                                                reject_saturated=True, calibration=None)
        def close(self):
            self.closed = True
    accelerator = Failing()
    @contextmanager
    def backend(*args):
        yield accelerator, {'backend': 'cuda', 'reason': 'test'}
    monkeypatch.setattr(module, '_screening_backend', backend)
    events = []
    actual = module.screen_source(source, cfg, on_device=events.append)
    np.testing.assert_array_equal(actual.measurements, expected.measurements)
    np.testing.assert_array_equal(actual.accepted, expected.accepted)
    assert accelerator.closed
    assert actual.summary['execution']['cuda_frames'] == 4*successful_batches
    assert [event['backend'] for event in events] == ['cuda', 'cpu']
    assert 'injected CUDA allocation failure' in actual.summary['execution']['reason']


def test_next_cuda_batch_overlaps_cpu_measurements_and_keeps_io_on_owner(monkeypatch):
    import threading
    from test_parallel_screening import OwnedSource
    next_started, measured = threading.Event(), threading.Event()
    class Overlapping:
        calls = 0
        def prepare(self, batch, *args):
            self.calls += 1
            if self.calls == 2:
                next_started.set()
                assert measured.wait(5), 'CPU measurements did not overlap the next GPU batch'
            return batch
        @staticmethod
        def measure(raw):
            assert next_started.wait(5)
            measured.set()
            return module._measure_observation(raw, color='mono', bit_depth=16,
                                                reject_saturated=True, calibration=None)
    @contextmanager
    def backend(*args):
        yield Overlapping(), {'backend': 'cuda', 'reason': 'test'}
    monkeypatch.setattr(module, '_screening_backend', backend)
    source = OwnedSource(count=8)
    result = module.screen_source(source, ReconstructionConfig(device='cpu', threads=2, batch_frames=4))
    assert source.reads == list(range(8)) and result.summary['n_measured'] == 8
    assert not any(t.name.startswith('planetrecon-screen-cuda') for t in threading.enumerate())


def test_cuda_subdivides_batches_and_cancels_between_chunks(cuda, monkeypatch):
    from planetrecon.backends.screening import CUDAScreening
    frames = np.stack([planet()]*8)
    source = ArraySource(frames)
    cfg = ReconstructionConfig(device='gpu', threads=2, batch_frames=8)
    # Enough reported headroom for exactly one frame at a time; no actual
    # allocator limits are changed by this test.
    monkeypatch.setattr(cuda.cuda, 'mem_get_info',
                        lambda _: (128*int(np.prod(frames.shape[1:])), 12*1024**3))
    calls = []
    original = CUDAScreening._filter
    def filtering(self, plane, axis):
        calls.append(plane.shape[0])
        return original(self, plane, axis)
    monkeypatch.setattr(CUDAScreening, '_filter', filtering)
    actual = module.screen_source(source, cfg)
    assert calls == [1]*16 and actual.summary['execution']['cuda_frames'] == 8
    calls.clear()
    cancelled = module.screen_source(source, cfg, should_cancel=lambda: len(calls) >= 2)
    assert cancelled.cancelled and cancelled.summary['n_measured'] == 0
    assert np.isnan(cancelled.measurements).all()


def test_cuda_cache_reload_and_cli_device_selection(cuda, tmp_path, capsys):
    import json
    from planetrecon.cli import main
    from planetrecon.io.ser import SERSource, write_ser
    from planetrecon.pipeline.preprocess_cache import load_cache
    path = write_ser(tmp_path/'capture.ser', np.stack([planet(blur=.7+i*.02) for i in range(16)]).astype('u2'))
    cache = tmp_path/'preprocessing.npz'
    assert main(['--threads', '2', 'preprocess', '--path', str(path), '--cache', str(cache),
                 '--device', 'gpu']) == 0
    report = json.loads(capsys.readouterr().out)
    assert report['execution']['cuda_frames'] == 16
    with SERSource(path) as source:
        cached, report = load_cache(source, ReconstructionConfig(device='cpu', threads=2), path=cache)
        reference = module.screen_source(source, ReconstructionConfig(device='cpu', threads=2))
    assert report['status'] == 'ready'
    np.testing.assert_array_equal(cached.accepted, reference.accepted)
    np.testing.assert_allclose(cached.measurements, reference.measurements, rtol=2e-12, atol=1e-10)


def test_preprocessing_worker_reports_cuda_then_cpu_geometry(cuda, tmp_path):
    import time
    from planetrecon.io.ser import write_ser
    from planetrecon.jobs import start_stack_job
    path = write_ser(tmp_path/'capture.ser', np.stack([planet()]*16).astype('u2'))
    job = start_stack_job(path, ReconstructionConfig(device='gpu', threads=2), preprocess_only=True)
    events = []
    try:
        deadline = time.monotonic()+30
        while job.state not in ('completed', 'failed') and time.monotonic() < deadline:
            events.extend(job.poll(.05))
        assert job.state == 'completed', [(event.kind, event.payload) for event in events]
        progress = [event.payload for event in events if event.kind == 'progress']
        assert any(item.get('backend') == 'cuda' for item in progress)
        assert any(item.get('backend') == 'cpu' and 'geometry' in item.get('stage', '') for item in progress)
        final = next(event.payload for event in events if event.kind == 'completed')
        assert final['preprocessing_cache']['execution']['cuda_frames'] == 16
    finally:
        job.close()
