"""Session validation crosses workers without re-reading unchanged captures."""
from dataclasses import replace
import os

import numpy as np
import pytest

from planetrecon.io.ser import SERSource
from planetrecon.io.source import ArraySource
from planetrecon.pipeline import preprocess_cache as cache
from planetrecon.pipeline.baseline import stack_source
from test_preprocess_cache import capture, config
from test_w14 import gui, pump


def test_preprocess_proof_reused_across_workers_and_sessions(tmp_path, monkeypatch):
    path, cfg = capture(tmp_path), config()
    receipt = cache.CacheValidation()
    with SERSource(path) as source:
        expected = cache.preprocess_source(source, cfg)
        hashes = []
        identity = cache.identity
        def counted(*args, **kwargs):
            hashes.append(True)
            return identity(*args, **kwargs)
        monkeypatch.setattr(cache, 'identity', counted)
        selected, report = cache.load_cache(source, cfg, validation=receipt)
        assert report['status'] == 'ready' and not hashes
    # Reopened adapters model separate desktop workers; output settings can change.
    with SERSource(path) as source:
        selected, report = cache.load_cache(source, replace(cfg, threads=1, stack_percent=50), validation=receipt)
        assert report['status'] == 'ready'
        result = stack_source(source, cfg, cache_validation=receipt)
        assert result.n_used == int(expected.accepted.sum())
        assert not hashes
        assert 'cache_validation' not in result.provenance
        cache.load_cache(source, cfg)  # A new session also uses the persisted proof.
        assert not hashes
        cache.load_cache(source, cfg, force_full_validation=True)
        assert len(hashes) == 1


def test_preprocess_hands_validation_directly_to_stack(tmp_path, monkeypatch):
    path, cfg = capture(tmp_path), config()
    receipt = cache.CacheValidation()
    with SERSource(path) as source:
        cache.preprocess_source(source, cfg, validation=receipt)
    monkeypatch.setattr(cache, 'identity', lambda *a, **k: pytest.fail('duplicate full validation'))
    with SERSource(path) as source:
        assert stack_source(source, cfg, cache_validation=receipt).n_used == 32


@pytest.mark.parametrize('change', ['pixels', 'calibration', 'colour', 'saturation', 'cache', 'replacement'])
def test_receipt_never_hides_input_or_cache_changes(tmp_path, change):
    path, cfg = capture(tmp_path), config()
    receipt = cache.CacheValidation()
    with SERSource(path) as source:
        cache.preprocess_source(source, cfg, validation=receipt)
        cache_path = cache.default_cache_path(source)
    kwargs = {}
    if change == 'pixels':
        stamp = path.stat()
        with path.open('r+b') as stream:
            stream.seek(stamp.st_size//2)
            value = stream.read(1)
            stream.seek(-1, 1)
            stream.write(bytes([value[0] ^ 1]))
        os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    elif change == 'calibration':
        cfg = replace(cfg, gain_e_per_adu=2)
    elif change == 'colour':
        kwargs['bayer_override'] = 'RGGB'
    elif change == 'saturation':
        cfg = replace(cfg, reject_saturated=not cfg.reject_saturated)
    elif change == 'cache':
        cache_path.write_bytes(b'broken')
    else:
        stamp = path.stat()
        data = bytearray(path.read_bytes())
        data[len(data)//2] ^= 1
        replacement = path.with_suffix('.tmp')
        replacement.write_bytes(data)
        os.utime(replacement, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        replacement.replace(path)
    with SERSource(path, **kwargs) as source:
        selected, report = cache.load_cache(source, cfg, validation=receipt)
    assert selected is None
    assert report['status'] == ('invalid' if change == 'cache' else 'stale')
    assert receipt.digest is None


def test_reuse_rechecks_geometry_applicability_and_cancellation(tmp_path, monkeypatch):
    path, cfg = capture(tmp_path), config()
    receipt = cache.CacheValidation()
    with SERSource(path) as source:
        cache.preprocess_source(source, cfg, validation=receipt)
        monkeypatch.setattr(cache, 'identity', lambda *a, **k: pytest.fail('duplicate full validation'))
        _, report = cache.load_cache(source, replace(cfg, reference_epoch_s=10), validation=receipt)
        assert report['status'] == 'ready'
        assert report['geometry_estimate']['applicable'] is False
        with pytest.raises(InterruptedError):
            cache.load_cache(source, cfg, validation=receipt, should_cancel=lambda: True)
        assert receipt.digest is None


def test_file_path_on_array_source_does_not_allow_reuse(tmp_path):
    path, cfg = capture(tmp_path), config()
    receipt = cache.CacheValidation()
    with SERSource(path) as source:
        frames = np.stack([source.read_raw(i) for i in range(source.n_frames())])
    source = ArraySource(frames, path=str(path))
    cache.preprocess_source(source, cfg, validation=receipt)
    assert receipt.source_key is None
    frames[0, 0, 0] += 1
    assert cache.load_cache(source, cfg, validation=receipt)[1]['status'] == 'stale'


def test_desktop_load_then_stack_passes_validation_between_processes(gui, tmp_path, monkeypatch):
    from planetrecon.gui import app as module
    app, win = gui
    win.path = capture(tmp_path)
    with SERSource(win.path) as source:
        cache.preprocess_source(source, win.controls.configuration())
    stages, receipts = [], []
    start = module.start_stack_job
    def tracked(*args, **kwargs):
        receipts.append(kwargs.get('cache_validation'))
        handle = start(*args, **kwargs)
        poll = handle.poll
        def tracked_poll(*args, **kwargs):
            events = poll(*args, **kwargs)
            stages.extend(event.payload.get('stage', '') for event in events if event.kind == 'progress')
            return events
        handle.poll = tracked_poll
        return handle
    monkeypatch.setattr(module, 'start_stack_job', tracked)
    win._capture_ready()
    pump(app, lambda: win.job is None, timeout=30)
    assert not win.error.text()
    assert not any('Validating cached preprocessing' in stage for stage in stages)
    assert win.cache_validation['digest']
    stages.clear()
    win._stack()
    pump(app, lambda: win.job is None, timeout=30)
    assert not win.error.text()
    assert win.last_result is not None and win.last_result.n_used > 0
    assert len(receipts) == 3 and receipts[0] is None
    assert all(receipt['digest'] for receipt in receipts[1:])
    assert not any('Validating cached preprocessing' in stage for stage in stages)


def test_changed_capture_during_validation_does_not_issue_receipt(tmp_path, monkeypatch):
    path, cfg = capture(tmp_path), config()
    receipt = cache.CacheValidation()
    with SERSource(path) as source:
        cache.preprocess_source(source, cfg)
        identity = cache.identity
        def changing(*args, **kwargs):
            result = identity(*args, **kwargs)
            stamp = path.stat()
            os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns + 1_000_000))
            return result
        monkeypatch.setattr(cache, 'identity', changing)
        selected, report = cache.load_cache(source, cfg, validation=receipt, force_full_validation=True)
    assert selected is None and report['status'] == 'invalid'
    assert receipt.digest is None


def test_identical_cache_copy_retains_verified_source_proof(tmp_path, monkeypatch):
    path, cfg = capture(tmp_path), config()
    receipt = cache.CacheValidation()
    with SERSource(path) as source:
        cache.preprocess_source(source, cfg, validation=receipt)
        destination = cache.default_cache_path(source)
        replacement = destination.with_suffix('.tmp')
        replacement.write_bytes(destination.read_bytes())
        replacement.replace(destination)
        progress = []
        _, report = cache.load_cache(source, cfg, validation=receipt,
                                     on_progress=lambda done, total: progress.append(done))
        assert report['status'] == 'ready' and not progress
