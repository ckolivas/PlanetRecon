"""Bounded parallel frame evaluation using the independent spatial CPU operators.

The frame sum remains in input order; each FFT uses one worker. Caller owns BLAS
thread limits. At most `workers` full-scene results are pending, not all frames.
"""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from scipy import fft


class ParallelSceneBatch:
    def __init__(self, operators, *, workers=8):
        self.operators = tuple(operators)
        if not self.operators or int(workers) != workers or not 1 <= workers <= 16:
            raise ValueError('operators and one to sixteen workers required')
        self.workers = int(workers)
        self.shape = self.operators[0].scene_shape
        if any(op.scene_shape != self.shape for op in self.operators):
            raise ValueError('common scene shape required')
        self.peak_pending = 0

    def _ordered(self, fn, jobs):
        def work(job):
            with fft.set_workers(1):
                return fn(job)
        jobs = iter(jobs)
        pending = deque()
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            try:
                for _ in range(self.workers):
                    try: job = next(jobs)
                    except StopIteration: break
                    pending.append(pool.submit(work, job))
                self.peak_pending = max(self.peak_pending, len(pending))
                while pending:
                    # Consume in input order, regardless of completion order.
                    yield pending.popleft().result()
                    try: job = next(jobs)
                    except StopIteration: continue
                    pending.append(pool.submit(work, job))
            finally:
                for future in pending: future.cancel()

    def forward(self, scene):
        if np.shape(scene) != self.shape: raise ValueError('wrong scene shape')
        return list(self._ordered(lambda op: op.forward(scene), self.operators))

    def adjoint(self, residuals):
        if len(residuals) != len(self.operators): raise ValueError('one residual per frame required')
        result = np.zeros(self.shape)
        for value in self._ordered(lambda pair: pair[0].adjoint(pair[1]), zip(self.operators, residuals)):
            result += value
        return result

    def normal(self, scene, weights):
        if np.shape(scene) != self.shape or len(weights) != len(self.operators):
            raise ValueError('scene shape and one weight per frame required')
        result = np.zeros(self.shape)
        def normal(pair):
            op, weight = pair
            return op.adjoint(weight*op.forward(scene))
        for value in self._ordered(normal, zip(self.operators, weights)):
            result += value
        return result

    def cache_info(self):
        return {'workers': self.workers, 'peak_pending_frames': self.peak_pending,
                'scope': 'At most workers pending per-frame results plus accumulator and worker workspaces; no retained PSF spectra.'}
