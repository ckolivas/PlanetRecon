"""Concurrent measurement keeps serial I/O, ordering, cancellation and cleanup."""
from dataclasses import replace
import sys
import threading

import numpy as np
import pytest

from planetrecon.calibration import Calibration
from planetrecon.io.source import ArraySource
from planetrecon.pipeline import preprocess as module
from planetrecon.reconstruction import ReconstructionConfig
from test_preprocessing import planet


class OwnedSource(ArraySource):
    def __init__(self):
        super().__init__(np.stack([planet(blur=.7+i*.03)+i for i in range(16)]))
        self.owner = threading.get_ident()
        self.reads = []

    def read_raw(self, index):
        assert threading.get_ident() == self.owner
        self.reads.append(index)
        return super().read_raw(index)

    def metadata(self):
        assert threading.get_ident() == self.owner
        return super().metadata()


def same(first, second):
    np.testing.assert_array_equal(first.measurements, second.measurements)
    np.testing.assert_array_equal(first.accepted, second.accepted)
    assert first.summary == second.summary and first.cancelled == second.cancelled


def test_parallel_measurement_preserves_calibrated_results_and_thread_limits(monkeypatch):
    ctl = pytest.importorskip('threadpoolctl')
    cfg = ReconstructionConfig(device='cpu', threads=4, batch_frames=8)
    cal = Calibration(bias=np.full((96,112), 2.), flat=np.linspace(.8,1.1,96*112).reshape(96,112), gain_e_per_adu=1.7)
    expected = module.screen_source(OwnedSource(), replace(cfg, threads=1), cal)
    original = module.measure_frame
    barrier = threading.Barrier(2, timeout=5)
    workers = set()
    def measure(*args):
        workers.add(threading.current_thread())
        barrier.wait()
        return original(*args)
    monkeypatch.setattr(module, 'measure_frame', measure)
    before = {i['filepath']: i['num_threads'] for i in ctl.threadpool_info()}
    source = OwnedSource()
    progress = []
    def report(done, total):
        assert threading.get_ident() == source.owner
        progress.append((done,total))
    actual = module.screen_source(source, cfg, cal, on_progress=report)
    same(actual, expected)
    assert source.reads == list(range(16)) and progress == [(8,16),(16,16)]
    assert 2 <= len(workers) <= 4 and all(not t.is_alive() for t in workers)
    after = {i['filepath']: i['num_threads'] for i in ctl.threadpool_info()}
    assert before == after


def test_cancel_keeps_a_measured_prefix_and_allows_immediate_rerun():
    cfg = ReconstructionConfig(device='cpu', threads=4, batch_frames=8)
    source = OwnedSource()
    stop = False
    def progress(done, total):
        nonlocal stop
        stop = True
    partial = module.screen_source(source, cfg, should_cancel=lambda: stop, on_progress=progress)
    assert partial.cancelled and partial.summary['n_measured'] == 8
    assert source.reads == list(range(8))
    assert np.isnan(partial.measurements[8:]).all() and not partial.accepted[8:].any()
    whole = module.screen_source(source, cfg)
    same(whole, module.screen_source(OwnedSource(), replace(cfg, threads=1)))


def test_missing_pool_control_uses_serial_measurement(monkeypatch):
    cfg = ReconstructionConfig(device='cpu', threads=4, batch_frames=8)
    expected = module.screen_source(OwnedSource(), replace(cfg, threads=1))
    monkeypatch.setitem(sys.modules, 'threadpoolctl', None)
    owner = threading.get_ident()
    original = module.measure_frame
    def measure(*args):
        assert threading.get_ident() == owner
        return original(*args)
    monkeypatch.setattr(module, 'measure_frame', measure)
    same(module.screen_source(OwnedSource(), cfg), expected)


def test_worker_exception_joins_threads_before_propagation(monkeypatch):
    pytest.importorskip('threadpoolctl')
    workers = set()
    def fail(*args):
        workers.add(threading.current_thread())
        raise ValueError('measurement failed')
    monkeypatch.setattr(module, 'measure_frame', fail)
    with pytest.raises(ValueError, match='measurement failed'):
        module.screen_source(OwnedSource(), ReconstructionConfig(device='cpu', threads=4, batch_frames=8))
    assert workers and all(not t.is_alive() for t in workers)
