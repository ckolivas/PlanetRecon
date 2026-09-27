"""Parallel verification must retain full-content checks and legacy selections."""
from dataclasses import replace
import hashlib
import json
import os
import threading

import numpy as np
import pytest

from planetrecon.io.ser import SERSource, SER_HEADER_SIZE
from planetrecon.io.source import ArraySource
from planetrecon.pipeline import capture_hash as hashes, preprocess_cache as cache
from planetrecon.reconstruction import ReconstructionConfig
from test_ser_export import capture


@pytest.mark.parametrize('dtype', ['u1', '<u2', '>u2', 'f8'])
def test_hash_independent_of_workers_batches_and_frame_order_sensitive(dtype):
    frames = np.arange(37*13*16).reshape(37, 13, 16).astype(dtype)
    source = ArraySource(frames)
    cfg = ReconstructionConfig(device='cpu', threads=1, batch_frames=1)
    expected = cache.identity(source, cfg, None)
    for workers, batch in ((2, 7), (4, 1), (32, 64)):
        progress = []
        actual = cache.identity(source, replace(cfg, threads=workers, batch_frames=batch), None,
                                on_progress=lambda done, total: progress.append((done, total)))
        assert actual == expected
        assert progress[0] == (0, 37) and progress[-1] == (37, 37)
        assert all(a[0] < b[0] for a, b in zip(progress, progress[1:]))
    assert cache.identity(ArraySource(frames[::-1]), cfg, None) != expected
    frames[18, 6, 8] += 1
    assert cache.identity(source, cfg, None) != expected


@pytest.mark.parametrize('workers', [1, 4, 8])
def test_honours_worker_count_but_never_reads_source_in_worker(monkeypatch, workers):
    frames = np.arange(24*64*64).reshape(24, 64, 64).astype('u2')
    source = ArraySource(frames)
    owner, seen = threading.get_ident(), set()
    barrier = threading.Barrier(workers, timeout=5)
    original_hash, original_read = hashes._hash_batch, source.read_raw
    def read(index):
        assert threading.get_ident() == owner
        return original_read(index)
    def hash_batch(*args):
        seen.add(threading.get_ident())
        barrier.wait()
        return original_hash(*args)
    monkeypatch.setattr(source, 'read_raw', read)
    monkeypatch.setattr(hashes, '_hash_batch', hash_batch)
    cache.identity(source, ReconstructionConfig(device='cpu', threads=workers, batch_frames=1), None)
    assert len(seen) == workers
    assert (owner in seen) == (workers == 1)
    assert not any(t.name.startswith('planetrecon-hash') for t in threading.enumerate())


def test_adapters_reusing_mutable_buffer_are_safe():
    frames = np.arange(37*13*16).reshape(37, 13, 16).astype('u2')
    source = ArraySource(frames)
    cfg = ReconstructionConfig(device='cpu', threads=4, batch_frames=3)
    expected = cache.identity(source, cfg, None)
    buffer = np.empty_like(frames[0])
    def read(index):
        buffer[:] = frames[index]
        return buffer
    source.read_raw = read
    assert cache.identity(source, cfg, None) == expected


@pytest.mark.parametrize('color', [0, 8, 100, 101])
@pytest.mark.parametrize('depth,endian', [(8, 0), (12, 1), (16, 0)])
def test_ser_hash_matches_decoded_pixels(tmp_path, color, depth, endian):
    path, cfg, _ = capture(tmp_path, color=color, depth=depth, endian=endian)
    with SERSource(path) as source:
        decoded = ArraySource(np.stack([source.read_raw(i) for i in range(source.n_frames())]))
        expected = hashes.pixel_hash(decoded, replace(cfg, threads=1))[0]
        for workers in (1, 4):
            assert hashes.pixel_hash(source, replace(cfg, threads=workers, batch_frames=1))[0] == expected


@pytest.mark.parametrize('mode', ['cancel', 'error'])
def test_shutdown_joins_workers_on_cancel_and_failure(monkeypatch, mode):
    source = ArraySource(np.zeros((64, 64, 64), dtype='u2'))
    cancel = threading.Event()
    original = hashes._hash_batch
    def run(*args):
        if mode == 'error':
            raise OSError('hash worker failed')
        cancel.set()
        return original(*args)
    monkeypatch.setattr(hashes, '_hash_batch', run)
    with pytest.raises(InterruptedError if mode == 'cancel' else OSError):
        cache.identity(source, ReconstructionConfig(device='cpu', threads=8, batch_frames=1), None,
                       should_cancel=cancel.is_set)
    assert not any(t.name.startswith('planetrecon-hash') for t in threading.enumerate())


def test_memory_budget_reduces_batches_before_workers(monkeypatch):
    source = ArraySource(np.zeros((16, 64, 64), dtype='u2'))
    frame_bytes = source.read_raw(0).nbytes
    monkeypatch.setattr(hashes, 'total_ram_bytes', lambda: frame_bytes*2*4)
    monkeypatch.setattr(hashes, 'virtual_bytes', lambda: 100)
    cfg = ReconstructionConfig(device='cpu', threads=4, batch_frames=32,
                               max_ram_bytes=100+2*4*frame_bytes)
    barrier = threading.Barrier(4, timeout=5)
    original = hashes._hash_batch
    def run(frames, cancel):
        assert len(frames) == 1
        barrier.wait()
        return original(frames, cancel)
    monkeypatch.setattr(hashes, '_hash_batch', run)
    cache.identity(source, cfg, None)


def legacy_capture(tmp_path):
    path, cfg, selected = capture(tmp_path)
    with SERSource(path) as source:
        digest = hashlib.sha256()
        for i in range(source.n_frames()):
            frame = np.ascontiguousarray(source.read_raw(i))
            digest.update(frame.dtype.str.encode())
            digest.update(frame.tobytes())
        selected.identity.pop('pixels_hash_method')
        selected.identity['pixels_sha256'] = digest.hexdigest()
        selected.digest = cache.selection_digest(selected)
        cache.save_cache(cache.default_cache_path(source), selected, source, cfg)
    return path, cfg, selected


def test_legacy_upgrade_single_read_preserves_selection_and_checkpoint_identity(tmp_path, monkeypatch):
    path, cfg, expected = legacy_capture(tmp_path)
    receipt = cache.CacheValidation()
    checkpoint_digest = cache.reconstruction_digest(expected)
    with SERSource(path) as source:
        reads, original = [], source.read_frame_bytes
        def read(i):
            reads.append(i)
            return original(i)
        monkeypatch.setattr(source, 'read_frame_bytes', read)
        actual, report = cache.load_cache(source, cfg, validation=receipt)
        assert report['status'] == 'ready'
        assert reads == list(range(source.n_frames()))
        assert actual.validation_identity['pixels_hash_method'] == hashes.HASH_METHOD
        assert actual.identity == expected.identity and actual.digest == expected.digest
        assert actual.summary == expected.summary
        np.testing.assert_array_equal(actual.measurements, expected.measurements)
        np.testing.assert_array_equal(actual.accepted, expected.accepted)
        assert cache.reconstruction_digest(actual) == checkpoint_digest
        with np.load(cache.default_cache_path(source)) as data:
            metadata = json.loads(str(data['metadata']))
            assert metadata['validation_identity'] == actual.validation_identity
        reads.clear()
        assert cache.load_cache(source, cfg, validation=receipt)[1]['status'] == 'ready'
        assert not reads
    # Touching the file forces a parallel check, without repeating the stream hash.
    stamp = path.stat()
    os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns + 1))
    identity = cache.identity
    def current_only(*args, **kwargs):
        assert kwargs.get('legacy_identity') is None
        assert kwargs.get('hash_method') == hashes.HASH_METHOD
        return identity(*args, **kwargs)
    monkeypatch.setattr(cache, 'identity', current_only)
    with SERSource(path) as source:
        assert cache.load_cache(source, replace(cfg, threads=4))[1]['status'] == 'ready'
    stamp = path.stat()
    with path.open('r+b') as stream:
        stream.seek(SER_HEADER_SIZE + 180)
        stream.write(b'\xff')
    os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    with SERSource(path) as source:
        assert cache.load_cache(source, cfg, validation=receipt)[1]['status'] == 'stale'


def test_legacy_cancel_does_not_replace_cache(tmp_path):
    path, cfg, _ = legacy_capture(tmp_path)
    with SERSource(path) as source:
        cache_path = cache.default_cache_path(source)
        before = cache_path.read_bytes()
        cancel = threading.Event()
        def progress(done, total):
            if done == total:
                cancel.set()
        with pytest.raises(InterruptedError):
            cache.load_cache(source, cfg, should_cancel=cancel.is_set, on_progress=progress)
        assert cache_path.read_bytes() == before
        assert not list(tmp_path.glob('.*.tmp'))


def test_read_only_legacy_cache_still_usable(tmp_path, monkeypatch):
    path, cfg, selected = legacy_capture(tmp_path)
    def denied(*args, **kwargs):
        raise PermissionError('read-only cache')
    monkeypatch.setattr(cache, 'save_cache', denied)
    with SERSource(path) as source:
        loaded, report = cache.load_cache(source, cfg)
        assert report['status'] == 'ready' and loaded.digest == selected.digest


def test_corrupt_migrated_verification_is_rejected(tmp_path):
    path, cfg, _ = legacy_capture(tmp_path)
    with SERSource(path) as source:
        assert cache.load_cache(source, cfg)[1]['status'] == 'ready'
        cache_path = cache.default_cache_path(source)
        with np.load(cache_path) as data:
            payload = {key: data[key].copy() for key in data.files}
        metadata = json.loads(str(payload['metadata']))
        metadata['validation_identity']['pixels_sha256'] = '0'*64
        payload['metadata'] = json.dumps(metadata)
        np.savez_compressed(cache_path, **payload)
        assert cache.load_cache(source, cfg)[1]['status'] == 'invalid'


def test_metadata_only_change_requires_one_full_check_then_reuses_across_sessions(tmp_path, monkeypatch):
    path, cfg, expected = capture(tmp_path)
    with SERSource(path) as source:
        assert cache.load_cache(source, cfg)[1]['status'] == 'ready'
    stamp = path.stat()
    os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns + 1))
    calls, identity = [], cache.identity
    def counted(*args, **kwargs):
        calls.append(True)
        return identity(*args, **kwargs)
    monkeypatch.setattr(cache, 'identity', counted)
    for _ in range(2):
        with SERSource(path) as source:
            selected, report = cache.load_cache(source, cfg)
            assert report['status'] == 'ready' and selected.digest == expected.digest
    assert len(calls) == 1


def test_persisted_proof_cannot_hide_corrupt_measurements(tmp_path):
    path, cfg, _ = capture(tmp_path)
    with SERSource(path) as source:
        assert cache.load_cache(source, cfg)[1]['status'] == 'ready'
        cache_path = cache.default_cache_path(source)
        with np.load(cache_path) as data:
            payload = {key: data[key].copy() for key in data.files}
            assert 'source_validation_digest' in json.loads(str(data['metadata']))
        payload['measurements'][3, 0] += 1
        np.savez_compressed(cache_path, **payload)
        assert cache.load_cache(source, cfg)[1]['status'] == 'invalid'
