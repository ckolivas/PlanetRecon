"""Bounded CPU frame workers with ordered reduction and owned cancellation."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from contextlib import ExitStack
import threading

from planetrecon.runtime import MAX_CPU_THREADS


def selected_batches(source, size, *, start, accepted, should_cancel):
    """Fill work batches with selected frames, without retaining raw batches.

    A 20% quality selection must not leave 80% of CPU workers idle. Detector
    reads remain serial; copies own only retained observations across batches.
    """
    import numpy as np
    indices, frames = [], []
    for raw_indices, raw_batch in source.iter_batches(size, start=start, should_cancel=should_cancel):
        for index, raw in zip(raw_indices, raw_batch):
            if should_cancel is not None and should_cancel():
                return
            if not accepted[index]:
                continue
            indices.append(int(index))
            frames.append(np.array(raw, copy=True))
            if len(frames) == size:
                yield indices, frames
                indices, frames = [], []
    if frames:
        yield indices, frames


class CPUFramePool:
    def __init__(self, config, shape, n_frames, *, enabled, should_cancel=None):
        import numpy as np
        pixels = int(np.prod(shape))
        # Includes working proxies, fields and queued projected colour planes.
        # Allow half the total RAM for scratch, leaving the rest for source batches
        # and other uses. This sizes concurrency; it does not reserve memory.
        from planetrecon.memory import total_ram_bytes, virtual_bytes
        total = total_ram_bytes()
        budget = 0 if total is None else total//2
        if config.max_ram_bytes is not None:
            budget = min(budget, max(0, config.max_ram_bytes-virtual_bytes())//2)
        per_frame = 192*pixels + 8*1024**2
        self.scratch_budget_bytes = budget
        self.estimated_frame_bytes = per_frame
        self.workers = max(1, min(config.threads, MAX_CPU_THREADS, config.batch_frames,
                                  n_frames, budget//max(1, per_frame))) if enabled else 1
        self.should_cancel = should_cancel
        self.stop = threading.Event()
        self.resources = ExitStack()
        self.executor = None
        self.reason = ('bounded by threads, batch size, remaining frames and half total RAM; '
                       'also half remaining process allowance when configured')

    def __enter__(self):
        if self.workers > 1:
            try:
                from threadpoolctl import threadpool_limits
            except ImportError:
                self.workers = 1
                self.reason = 'threadpoolctl unavailable; serial to prevent nested native pools'
            else:
                self.resources.enter_context(threadpool_limits(limits=1))
                try:
                    self.executor = ThreadPoolExecutor(self.workers, thread_name_prefix='planetrecon-stack')
                except BaseException:
                    self.resources.close()
                    raise
        return self

    def cancelled(self):
        if self.should_cancel is not None and self.should_cancel():
            self.stop.set()
        return self.stop.is_set()

    def map(self, function, items):
        """At most workers outstanding results; emit in original frame order."""
        items = iter(items)
        if self.executor is None:
            for item in items:
                if self.cancelled():
                    return
                yield function(item)
            return
        pending = deque()
        def submit():
            try:
                item = next(items)
            except StopIteration:
                return
            pending.append(self.executor.submit(function, item))
        for _ in range(self.workers):
            submit()
        try:
            while pending:
                if self.cancelled():
                    return
                future = pending[0]
                try:
                    value = future.result(timeout=.05)
                except TimeoutError:
                    if not future.done():
                        continue
                    # Completion can race the timeout. Retrieve that result;
                    # a TimeoutError raised by the worker itself still escapes.
                    value = future.result()
                pending.popleft()
                yield value
                if not self.cancelled():
                    submit()
        finally:
            for future in pending:
                future.cancel()

    def __exit__(self, *exc):
        self.stop.set()
        try:
            if self.executor is not None:
                self.executor.shutdown(wait=True, cancel_futures=True)
        finally:
            self.resources.close()
