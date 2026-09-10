"""Bounded float64 CUDA moments with complete-frame CPU retry.

Geometry, ordinary CFA projection and final fitting retain the qualified CPU
implementation. Only successful complete frames enter authoritative sums.
"""
import numpy as np

from tools.cfa_local_transport_probe import LocalQuadraticAccumulator, inverse_pull_map


def detector_window(shift, shape):
    """Enclose exactly the fixed reference-space search; extra slots get zero weight.

    For |r-r0|_infinity <= 4, the mapped detector x changes by at most
    4*(1 + sup|d(dx)/dx| + sup|d(dx)/dy|), and similarly for y. Bilinear
    interpolation is bounded by the adjacent grid differences, including knots.
    A 1e-8 detector-coordinate margin covers the inverse residual tolerance.
    """
    dx, dy = (np.broadcast_to(np.asarray(v, dtype=float), shape) for v in shift)
    extents = [4*(1+sum(float(np.abs(np.diff(v, axis=axis)).max()) for axis in (0, 1)))+1e-8
               for v in (dx, dy)]
    offsets = [np.arange(-int(np.floor(v)), int(np.ceil(v))+1) for v in extents]
    ys, xs = np.meshgrid(offsets[1], offsets[0], indexing='ij')
    return dx, dy, xs.ravel(), ys.ravel()


class CudaLocalAccumulator(LocalQuadraticAccumulator):
    def __init__(self, *args, gpu_chunk_rows=512, max_gpu_array_bytes=128*1024**2, **kwargs):
        super().__init__(*args, **kwargs)
        for name, value in [('gpu_chunk_rows', gpu_chunk_rows), ('max_gpu_array_bytes', max_gpu_array_bytes)]:
            if type(value) is not int or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        self.gpu_chunk_rows = gpu_chunk_rows
        self.max_gpu_array_bytes = max_gpu_array_bytes
        # GPU batches may exceed CPU chunk_rows; include their CPU host arrays
        # in the existing neighbour budget rather than widening that budget.
        self.gpu_enabled = True
        self.execution_history = []
        self.fallback_reason = None
        self.fallback_frame = None
        self.device_identity = None
        self.expected_device_identity = None
        self.maximum_batch_entries = 0

    def _device(self):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable in this runtime')
        properties = torch.cuda.get_device_properties(0)
        identity = {'name': properties.name, 'capability': [properties.major, properties.minor],
                    'multiprocessors': properties.multi_processor_count, 'torch': str(torch.__version__),
                    'cuda': torch.version.cuda}
        if self.expected_device_identity is not None and identity != self.expected_device_identity:
            raise ValueError('CUDA device/runtime differs from checkpoint execution')
        self.device_identity = identity
        return torch

    def _cuda_products(self, phi, weights, values):
        """One bounded batch; return all three moment arrays together."""
        import torch
        entries = int(np.prod(phi.shape[:2]))
        planned = entries*512+len(phi)*3*78*8+32*1024**2
        if planned > self.max_gpu_array_bytes:
            raise RuntimeError('CUDA moment batch exceeds the configured array planning budget')
        self.maximum_batch_entries = max(self.maximum_batch_entries, entries)
        p = torch.as_tensor(phi, dtype=torch.float64, device='cuda:0')
        w = torch.as_tensor(weights, dtype=torch.float64, device='cuda:0')
        v = torch.as_tensor(values, dtype=torch.float64, device='cuda:0')
        transposed = p.transpose(1, 2)[:, None]
        left = transposed*w[:, :, None]
        gram = left@p[:, None]
        noise = (transposed*w[:, :, None].square())@p[:, None]
        rhs = (left@v[:, None, :, None])[..., 0]
        combined = torch.cat([gram.flatten(2), noise.flatten(2), rhs], dim=2).cpu().numpy()
        if not np.isfinite(combined).all():
            raise RuntimeError('CUDA moment batch produced nonfinite contributions')
        return combined[..., :36].reshape(-1, 3, 6, 6), combined[..., 36:72].reshape(-1, 3, 6, 6), combined[..., 72:]

    def stage_moments(self, raw, shift, quality):
        if not self.gpu_enabled:
            return super().stage_moments(raw, shift, quality)
        self._check_cancel()
        self._device()
        positions, observed = inverse_pull_map(shift, self.shape, check_cancel=self._check_cancel)
        dx, dy, ox, oy = detector_window(shift, self.shape)
        rows_per_batch = min(self.gpu_chunk_rows, self.max_neighbour_entries//len(ox))
        if rows_per_batch < 1:
            raise MemoryError('one local detector window exceeds the neighbour entry budget')
        gram, noise, rhs = (np.zeros_like(v) for v in (self.gram, self.noise, self.rhs))
        h, w = self.shape
        for start in range(0, len(self.targets), rows_per_batch):
            self._check_cancel()
            stop = min(start+rows_per_batch, len(self.targets))
            targets = self.targets[start:stop]
            tx, ty = targets.T
            centres_x = np.floor(tx+dx[ty, tx]).astype(int)
            centres_y = np.floor(ty+dy[ty, tx]).astype(int)
            sx, sy = centres_x[:, None]+ox, centres_y[:, None]+oy
            valid = (sx >= 0) & (sx < w) & (sy >= 0) & (sy < h)
            cx, cy = sx.clip(0, w-1), sy.clip(0, h-1)
            valid &= observed[cy, cx]
            uv = (positions[cy, cx]-targets[:, None])/4
            u, v = uv[..., 0], uv[..., 1]
            phi = np.stack([np.ones_like(u), u, v, u*u, u*v, v*v], axis=-1)
            kernel = quality*np.maximum(1-u*u, 0)**3*np.maximum(1-v*v, 0)**3*valid
            labels = self.labels[cy, cx]
            weights = np.stack([kernel*(labels == name) for name in 'RGB'], axis=1)
            g, hessian, b = self._cuda_products(phi, weights, raw[cy, cx])
            gram[start:stop], noise[start:stop], rhs[start:stop] = g, hessian, b
        self._check_cancel()
        return gram, noise, rhs

    def add(self, raw, shift, quality):
        if self.gpu_enabled:
            try:
                staged = self.stage(raw, shift, quality)
            except (RuntimeError, ImportError) as exc:
                # Exit the exception scope before retry to release failed GPU
                # temporaries and any partial host contribution held by traceback.
                self.gpu_enabled = False
                self.fallback_frame = self.n_used
                self.fallback_reason = f'{type(exc).__name__}: {exc}; complete frame retried on CPU'
            else:
                history = self.execution_history+['cuda']
                self.publish(staged)
                self.execution_history = history
                return
        staged = self.stage(raw, shift, quality)
        history = self.execution_history+['cpu']
        self.publish(staged)
        self.execution_history = history
