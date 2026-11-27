"""Frame reads ahead of a known order on a background thread, overlapping I/O with compute."""

from __future__ import annotations

import queue
import threading

import numpy as np

_DONE = object()


class _Worker:
    """Reads ``order`` on a thread, at most ``depth`` frames ahead of the consumer."""

    def __init__(self, read, order, depth):
        self.position = {}
        for p, index in enumerate(order):
            self.position.setdefault(index, p)
        self.next = 0
        self.done = False
        self.queue = queue.Queue(maxsize=depth)
        self.stopping = threading.Event()
        self.thread = threading.Thread(target=self._run, args=(read, order), daemon=True,
                                       name='planetrecon-readahead')
        self.thread.start()

    def _put(self, item):
        while not self.stopping.is_set():
            try:
                self.queue.put(item, timeout=.1)
                return True
            except queue.Full:
                continue
        return False

    def _run(self, read, order):
        for p, index in enumerate(order):
            if self.stopping.is_set():
                return
            try:
                item = (p, read(index))
            except Exception:  # The consumer's own read raises it where it belongs.
                break
            if not self._put(item):
                return
        self._put(_DONE)

    def take(self, index):
        """The prefetched frame ``index`` when it is still ahead in the order, else None."""
        p = self.position.get(index)
        if self.done or p is None or p < self.next:
            return None
        while True:
            item = self.queue.get()
            if item is _DONE:
                self.done = True
                return None
            got, frame = item
            self.next = got + 1
            if got == p:
                return frame

    def close(self):
        self.stopping.set()
        while self.thread.is_alive():
            try:
                self.queue.get(timeout=.05)
            except queue.Empty:
                pass
        self.thread.join()


class Readahead:
    """Serialised frame reads with background read-ahead of an announced order.

    ``expect(indices)`` starts reading those frames, in that order, on a
    background thread. ``read_raw`` returns a prefetched frame when it is still
    ahead in the announced order (announced frames passed over are dropped)
    and otherwise reads directly. Every source read holds one lock, so the
    source need not be thread-safe. Frames are returned exactly as the source
    reads them.
    """

    BYTES_AHEAD = 256 * 1024 * 1024

    def __init__(self, source, *, frames_ahead=32):
        self.source = source
        self._lock = threading.Lock()
        self._worker = None
        shape = tuple(source.frame_shape())
        frame_bytes = 2 * int(np.prod(shape))  # Raw samples are read as 16-bit integers.
        self.frames_ahead = int(max(2, min(frames_ahead, self.BYTES_AHEAD // max(frame_bytes, 1))))

    def _read(self, index):
        with self._lock:
            return self.source.read_raw(int(index))

    def expect(self, indices):
        """Announce the next reads; replaces any earlier announcement."""
        self.stop()
        order = [int(i) for i in indices]
        if order:
            self._worker = _Worker(self._read, order, self.frames_ahead)

    def read_raw(self, index):
        worker = self._worker
        if worker is not None:
            frame = worker.take(int(index))
            if frame is not None:
                return frame
        return self._read(index)

    def batches(self, indices, size, *, should_cancel=None):
        """(indices, frames) batches of ``indices`` in order, read ahead; stops early on cancellation."""
        indices = np.asarray(indices, dtype=np.int64)
        self.expect(indices)
        try:
            for start in range(0, len(indices), max(1, int(size))):
                chunk = indices[start:start + max(1, int(size))]
                frames = []
                for index in chunk:
                    if should_cancel is not None and should_cancel():
                        return
                    frames.append(self.read_raw(index))
                yield chunk, frames
        finally:
            self.stop()

    def stop(self):
        if self._worker is not None:
            self._worker.close()
            self._worker = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.stop()
