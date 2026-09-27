"""Ordered frame hashes with bounded, CPU-parallel checksum work.

Source adapters are read only by the calling thread. SHA-256 releases the GIL
for image-sized buffers; workers never share file handles or mutable frames.
"""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import hashlib
import struct

import numpy as np

from planetrecon.memory import total_ram_bytes, virtual_bytes

HASH_METHOD = 'sha256-frames-v1'
LEGACY_HASH_METHOD = 'sha256-stream-v1'


def _cancel(should_cancel):
    if should_cancel and should_cancel():
        raise InterruptedError('preprocessing cache verification cancelled')


def _update(digest, frame):
    digest.update(frame.dtype.str.encode())
    digest.update(memoryview(frame).cast('B'))


def _hash_batch(frames, should_cancel):
    hashes = bytearray()
    for frame in frames:
        _cancel(should_cancel)
        digest = hashlib.sha256()
        _update(digest, frame)
        hashes.extend(digest.digest())
    return bytes(hashes)


def pixel_hash(source, config, should_cancel=None, on_progress=None, *,
               method=HASH_METHOD, include_legacy=False):
    """Return (pixel digest, optional legacy digest), independent of concurrency.

    Leaves hash dtype plus observed pixels; the root hashes a domain marker,
    frame count and every leaf in capture order. Batch size is scheduling only.
    A legacy digest can be computed during the same read pass for cache migration.
    """
    if method not in (HASH_METHOD, LEGACY_HASH_METHOD):
        raise ValueError(f'unknown capture checksum method: {method}')
    total = source.n_frames()
    stride = max(1, total // 100)
    done = reported = 0
    if on_progress:
        on_progress(0, total)
    _cancel(should_cancel)
    legacy = hashlib.sha256() if include_legacy or method == LEGACY_HASH_METHOD else None
    if method == LEGACY_HASH_METHOD:
        for i in range(total):
            _cancel(should_cancel)
            _update(legacy, np.ascontiguousarray(source.read_raw(i)))
            if on_progress and ((i + 1) % stride == 0 or i + 1 == total):
                on_progress(i + 1, total)
        _cancel(should_cancel)
        return legacy.hexdigest(), legacy.hexdigest()

    root = hashlib.sha256(HASH_METHOD.encode() + b'\0' + struct.pack('<Q', total))
    pending = deque()
    pool = None
    workers = max(1, min(config.threads, total))
    batch_frames = max(1, config.batch_frames)
    window = workers

    def finish(entry):
        nonlocal done, reported
        count, future = entry
        root.update(future.result())
        done += count
        _cancel(should_cancel)
        if on_progress and (done - reported >= stride or done == total):
            on_progress(done, total)
            reported = done

    try:
        frames = []
        for i in range(total):
            _cancel(should_cancel)
            frame = np.ascontiguousarray(source.read_raw(i))
            if i == 0:
                frame_bytes = frame.nbytes
                # Keep all requested workers, reducing batch size before
                # concurrency when a memory ceiling constrains queued work.
                budget = (total_ram_bytes() or frame_bytes * workers * 2) // 2
                if config.max_ram_bytes is not None:
                    budget = min(budget, max(0, config.max_ram_bytes - virtual_bytes()) // 2)
                capacity = max(1, budget // max(1, frame_bytes))
                window = min(workers, capacity)
                batch_frames = max(1, min(batch_frames, (4 * 1024**2) // max(1, frame_bytes),
                                          capacity // window))
                if window > 1:
                    pool = ThreadPoolExecutor(max_workers=window, thread_name_prefix='planetrecon-hash')
                else:
                    batch_frames = 1
            if pool is not None:
                # A custom adapter may reuse its read buffer. Own worker input;
                # serial hashing finishes before the next read and needs no copy.
                frame = frame.copy()
            if legacy is not None:
                _update(legacy, frame)
            frames.append(frame)
            if len(frames) == batch_frames or i + 1 == total:
                if pool is None:
                    root.update(_hash_batch(frames, should_cancel))
                    done += len(frames)
                    if on_progress and (done - reported >= stride or done == total):
                        on_progress(done, total)
                        reported = done
                else:
                    pending.append((len(frames), pool.submit(_hash_batch, frames, should_cancel)))
                frames = []
                if len(pending) >= window:
                    finish(pending.popleft())
        while pending:
            finish(pending.popleft())
        _cancel(should_cancel)
    finally:
        for _, future in pending:
            future.cancel()
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
    return root.hexdigest(), None if legacy is None else legacy.hexdigest()
