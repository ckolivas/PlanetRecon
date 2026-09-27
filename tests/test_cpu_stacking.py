"""Threaded stacking must preserve detector values and ordered checkpoints."""
from dataclasses import replace
import threading
import time

import numpy as np
import pytest

from planetrecon.io.ser import SERSource
from planetrecon.pipeline.baseline import stack_source
from planetrecon.pipeline.preprocess_cache import preprocess_source
from planetrecon.pipeline.cpu_pool import CPUFramePool
from planetrecon.reconstruction import ReconstructionConfig
from test_local_alignment import capture, config


@pytest.mark.parametrize('color_id', [0, 8, 100])
@pytest.mark.parametrize('method', ['global', 'square', 'circular_multiscale'])
def test_workers_preserve_exact_image_and_coverage(tmp_path, color_id, method):
    cfg = replace(config(), threads=1, batch_frames=8, local_patch_size=33,
                  local_alignment=method != 'global',
                  alignment_method='circular_multiscale' if method == 'circular_multiscale' else 'square')
    with SERSource(capture(tmp_path, color_id)) as source:
        preprocess_source(source, cfg)
        serial = stack_source(source, cfg)
        parallel = stack_source(source, replace(cfg, threads=4))
    assert parallel.provenance['cpu_frame_workers']['workers'] == 4
    assert parallel.provenance['device_report']['frame_workers'] == 4
    assert serial.n_used == parallel.n_used and serial.n_rejected == parallel.n_rejected
    np.testing.assert_array_equal(parallel.image, serial.image)
    np.testing.assert_array_equal(parallel.coverage, serial.coverage)
    assert not any(t.name.startswith('planetrecon-stack') for t in threading.enumerate())


def test_uncached_reference_is_established_before_parallel_work(tmp_path):
    cfg = ReconstructionConfig(device='cpu', threads=1, frame_preselection=False, batch_frames=8)
    with SERSource(capture(tmp_path)) as source:
        serial = stack_source(source, cfg)
        parallel = stack_source(source, replace(cfg, threads=4))
    np.testing.assert_array_equal(parallel.image, serial.image)
    np.testing.assert_array_equal(parallel.coverage, serial.coverage)
    assert parallel.provenance['reference_index'] == serial.provenance['reference_index'] == 0
    assert parallel.provenance['cpu_frame_workers']['workers'] == 4


def test_frame_matching_really_overlaps(tmp_path, monkeypatch):
    from planetrecon.pipeline.circular_align import CircularMultiscaleRegistration
    method = CircularMultiscaleRegistration.displacement
    lock = threading.Lock()
    barrier = threading.Barrier(4, timeout=10)
    active = 0
    maximum = 0
    seen = set()
    def measured(self, frame, shift):
        nonlocal active, maximum
        ident = threading.get_ident()
        with lock:
            first = ident not in seen
            seen.add(ident)
            active += 1
            maximum = max(maximum, active)
        try:
            if first:
                barrier.wait()
            return method(self, frame, shift)
        finally:
            with lock:
                active -= 1
    monkeypatch.setattr(CircularMultiscaleRegistration, 'displacement', measured)
    cfg = replace(config(alignment_method='circular_multiscale'), threads=4, batch_frames=4,
                  stack_percent=25, frame_selection_mode='frame_count')
    with SERSource(capture(tmp_path)) as source:
        preprocess_source(source, cfg)
        result = stack_source(source, cfg)
    assert result.n_used == 4 and result.n_rejected == 12 and maximum == 4 and len(seen) == 4


def test_sparse_selection_and_trailing_rejections_remain_exact(tmp_path):
    cfg = replace(config(alignment_method='circular_multiscale'), threads=1, batch_frames=4,
                  stack_percent=25, frame_selection_mode='frame_count')
    with SERSource(capture(tmp_path)) as source:
        preprocess_source(source, cfg)
        serial = stack_source(source, cfg)
        parallel = stack_source(source, replace(cfg, threads=4), state_checkpoint=tmp_path/'sparse.npz')
        resumed = stack_source(source, replace(cfg, threads=4), resume_from=tmp_path/'sparse.npz')
    for result in (parallel, resumed):
        assert result.n_used == serial.n_used == 4 and result.n_rejected == serial.n_rejected == 12
        np.testing.assert_array_equal(result.image, serial.image)
        np.testing.assert_array_equal(result.coverage, serial.coverage)


def test_cancel_during_active_frames_resumes_exactly(tmp_path, monkeypatch):
    from planetrecon.pipeline.circular_align import CircularMultiscaleRegistration
    cfg = replace(config(alignment_method='circular_multiscale'), threads=4, batch_frames=8)
    with SERSource(capture(tmp_path)) as source:
        preprocess_source(source, cfg)
        whole = stack_source(source, cfg)
        cancel = threading.Event()
        calls = 0
        lock = threading.Lock()
        method = CircularMultiscaleRegistration.displacement
        def cancelling(self, frame, shift):
            nonlocal calls
            with lock:
                calls += 1
                if calls == 5:
                    cancel.set()
            return method(self, frame, shift)
        with monkeypatch.context() as patch:
            patch.setattr(CircularMultiscaleRegistration, 'displacement', cancelling)
            partial = stack_source(source, cfg, should_cancel=cancel.is_set,
                                   state_checkpoint=tmp_path/'state.npz')
        assert partial.incomplete and partial.n_used < whole.n_used
        assert not any(t.name.startswith('planetrecon-stack') for t in threading.enumerate())
        resumed = stack_source(source, cfg, resume_from=tmp_path/'state.npz')
        np.testing.assert_array_equal(resumed.image, whole.image)
        np.testing.assert_array_equal(resumed.coverage, whole.coverage)


def test_pool_bounds_reordering_errors_and_cleanup(monkeypatch):
    monkeypatch.setattr('planetrecon.memory.total_ram_bytes', lambda: 4*1024**3)
    cfg = ReconstructionConfig(threads=4, batch_frames=8)
    gate = threading.Event()
    started = []
    lock = threading.Lock()
    def work(i):
        with lock:
            started.append(i)
        if i == 0:
            gate.wait(5)
        return i
    with CPUFramePool(cfg, (96, 112), 20, enabled=True) as pool:
        iterator = pool.map(work, range(20))
        result = []
        owner = threading.Thread(target=lambda: result.extend(iterator))
        owner.start()
        deadline = time.monotonic()+3
        while len(started) < 4 and time.monotonic() < deadline:
            time.sleep(.01)
        assert len(started) == 4
        gate.set()
        owner.join(10)
        assert not owner.is_alive() and result == list(range(20))
    def failure(i):
        raise ValueError('injected worker failure')
    with pytest.raises(ValueError, match='injected'):
        with CPUFramePool(cfg, (96, 112), 20, enabled=True) as pool:
            list(pool.map(failure, range(20)))
    assert not any(t.name.startswith('planetrecon-stack') for t in threading.enumerate())
    assert CPUFramePool(cfg, (4000, 4000, 3), 20, enabled=True).workers == 1
    assert CPUFramePool(replace(cfg, batch_frames=2), (96, 112), 20, enabled=True).workers == 2


@pytest.mark.parametrize('total_gib,cap_gib,expected', [
    (96, None, 32),  # A large-memory machine can use every requested worker.
    (4, None, 2),    # Low total RAM still limits concurrent scratch.
    (96, 16, 5),    # The explicit cap uses remaining address space, not RAM size.
    (4, 128, 2),    # A high explicit cap cannot override physical capacity.
    (96, 8, 1),     # No remaining address space cannot enable parallel workers.
    (0, None, 1),
    (None, None, 1),
])
def test_pool_sizes_large_frames_from_total_memory(monkeypatch, total_gib, cap_gib, expected):
    gib = 1024**3
    total = None if total_gib is None else total_gib*gib
    monkeypatch.setattr('planetrecon.memory.total_ram_bytes', lambda: total)
    monkeypatch.setattr('planetrecon.memory.virtual_bytes', lambda: 8*gib)
    cfg = ReconstructionConfig(device='cpu', threads=32, batch_frames=32,
                               max_ram_bytes=None if cap_gib is None else cap_gib*gib)
    # Size the pool without allocating these large frames.
    pool = CPUFramePool(cfg, (2048, 2048), 64, enabled=True)
    assert pool.workers == expected
    assert CPUFramePool(replace(cfg, threads=1), (2048, 2048), 64, enabled=True).workers == 1
    assert CPUFramePool(cfg, (2048, 2048), 64, enabled=False).workers == 1
    assert CPUFramePool(cfg, (2048, 2048), 3, enabled=True).workers == min(expected, 3)
    assert CPUFramePool(replace(cfg, batch_frames=2), (2048, 2048), 64, enabled=True).workers == min(expected, 2)
    if total_gib == 96 and cap_gib is None:
        assert pool.scratch_budget_bytes == 48*gib
        assert pool.estimated_frame_bytes == 776*1024**2


def test_completion_racing_poll_timeout_is_not_a_worker_failure():
    from types import SimpleNamespace
    attempts = 0
    def result(timeout=None):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise TimeoutError
        return 42
    future = SimpleNamespace(result=result, done=lambda: True, cancel=lambda: None)
    pool = CPUFramePool(ReconstructionConfig(threads=2), (16, 16), 2, enabled=True)
    pool.executor = SimpleNamespace(submit=lambda *args: future)
    assert list(pool.map(lambda _: 42, [0])) == [42]
    def failure(_):
        raise TimeoutError('worker timeout')
    with pytest.raises(TimeoutError, match='worker timeout'):
        with CPUFramePool(ReconstructionConfig(threads=2), (16, 16), 2, enabled=True) as pool:
            list(pool.map(failure, [0]))


def test_detected_default_honours_affinity_and_gui_preferences(tmp_path, monkeypatch):
    monkeypatch.setenv('QT_QPA_PLATFORM', 'offscreen')
    from planetrecon import runtime
    from planetrecon.gui.app import MainWindow, create_app
    monkeypatch.setattr(runtime.os, 'sched_getaffinity', lambda _: set(range(6)), raising=False)
    monkeypatch.setattr(runtime.os, 'cpu_count', lambda: 64)
    monkeypatch.setenv('PLANETRECON_THREADS', '2')
    assert runtime.detected_thread_count() == 6
    assert runtime.default_thread_count() == 2
    app = create_app([])
    prefs = tmp_path/'settings.json'
    win = MainWindow(settings_path=prefs)
    try:
        assert win.controls.fields['threads'].value() == 6
        assert win.path is None
        win.controls.fields['threads'].setValue(3)
    finally:
        win.window.close()
    restored = MainWindow(settings_path=prefs)
    try:
        assert restored.controls.fields['threads'].value() == 3
    finally:
        restored.window.close()
    monkeypatch.delattr(runtime.os, 'sched_getaffinity')
    monkeypatch.setattr(runtime.os, 'cpu_count', lambda: 10)
    assert runtime.detected_thread_count() == 10
    monkeypatch.setattr(runtime.os, 'cpu_count', lambda: None)
    assert runtime.detected_thread_count() == 1
