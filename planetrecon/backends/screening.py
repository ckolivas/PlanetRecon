"""Batched CUDA measurement filters; raw stacking observations are untouched."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from planetrecon.detector import is_bayer


class CUDAScreening:
    def __init__(self, shape, color, calibration):
        self.shape = tuple(shape)
        self.scale = 2 if min(shape[:2]) >= 10 or is_bayer(color) else 1
        self.calibration = calibration
        self.tables = {}
        if calibration is not None:
            for name in ('bias', 'dark', 'flat'):
                value = getattr(calibration, name)
                if value is None:
                    continue
                value = np.asarray(value, dtype=np.float64)
                if name == 'flat':
                    if value.shape != self.shape or not np.isfinite(value).all() or np.any(value <= 0):
                        raise ValueError('flat must be finite, positive and match the raw detector frame shape')
                    relative = value / value.max()
                    value = float(np.median(relative)) / relative
                self.tables[name] = torch.tensor(value, device='cuda:0', dtype=torch.float64)
        # Same radius, coefficients and nearest-edge extension as SciPy sigma=1.
        x = np.arange(-4, 5, dtype=np.float64)
        kernel = np.exp(-.5*x*x)
        self.kernel = kernel/kernel.sum()

    def close(self):
        self.tables.clear()

    def _filter(self, plane, axis):
        h, w = plane.shape[-2:]
        padded = F.pad(plane[:, None], (0, 0, 4, 4) if axis == 1 else (4, 4, 0, 0), mode='replicate')[:, 0]
        result = plane * float(self.kernel[4])
        for distance in (4, 3, 2, 1):
            if axis == 1:
                pair = padded[:, 4-distance:4-distance+h] + padded[:, 4+distance:4+distance+h]
            else:
                pair = padded[:, :, 4-distance:4-distance+w] + padded[:, :, 4+distance:4+distance+w]
            result = result + pair * float(self.kernel[4-distance])
        return result

    @torch.no_grad()
    def prepare(self, batch, bit_depth, reject_saturated, should_cancel=None):
        free, total = torch.cuda.mem_get_info(0)
        cap = int(total * torch.cuda.get_per_process_memory_fraction(0))
        room = max(0, min(free, cap-torch.cuda.memory_allocated(0)))
        # Conservative simultaneous raw/calibrated/filter scratch estimate.
        chunk = max(1, min(len(batch), room//2//max(1, 64*int(np.prod(self.shape)))))
        items = []
        for start in range(0, len(batch), chunk):
            if should_cancel is not None and should_cancel():
                raise InterruptedError('CUDA preprocessing cancelled')
            raw = np.asarray(batch[start:start+chunk])
            raw = np.ascontiguousarray(raw, dtype=raw.dtype.newbyteorder('='))
            work = torch.as_tensor(raw, device='cuda:0').to(dtype=torch.float64)
            finite = torch.isfinite(work).flatten(1).all(1)
            saturated = torch.zeros_like(finite)
            if reject_saturated and bit_depth is not None and bit_depth <= 16:
                peak = (1 << min(int(bit_depth), 16))-1
                # Integer comparison keeps the exact 5% boundary independent
                # of a GPU mean reduction's rounding order.
                saturated = (work >= peak).flatten(1).sum(1)*20 > int(np.prod(self.shape))
            cal_saturated = torch.zeros_like(finite)
            cal = self.calibration
            if reject_saturated and cal is not None and cal.saturate_adu is not None:
                threshold = cal.saturate_adu
                if raw.dtype.kind == 'f':
                    threshold = np.asarray(threshold, dtype=np.result_type(raw, threshold)).item()
                cal_saturated = (work >= threshold).flatten(1).any(1)
            for name in ('bias', 'dark'):
                if name in self.tables:
                    work = work - self.tables[name]
            if 'flat' in self.tables:
                work = work * self.tables['flat']
            if cal is not None and cal.gain_e_per_adu is not None:
                work = work * float(cal.gain_e_per_adu)
            calibrated_finite = torch.isfinite(work).flatten(1).all(1)
            # Preserve CPU rejection precedence: invalid raw, raw saturation,
            # invalid calibration, then the configured detector saturation.
            codes = torch.zeros_like(finite, dtype=torch.int8)
            codes[cal_saturated] = 1
            codes[~calibrated_finite] = 2
            codes[saturated] = 1
            codes[~finite] = 2
            if work.ndim == 4:
                work = .25*work[..., 0] + .5*work[..., 1] + .25*work[..., 2]
            if self.scale == 2:
                n, h, w = work.shape
                work = work[:, :h//2*2, :w//2*2].reshape(n, h//2, 2, w//2, 2).mean((2, 4))
            smooth = self._filter(self._filter(work, 1), 2)
            padded = F.pad(smooth[:, None], (1, 1, 1, 1), mode='replicate')[:, 0]
            vertical = padded[:, :-2, 1:-1] - 2*smooth + padded[:, 2:, 1:-1]
            horizontal = padded[:, 1:-1, :-2] - 2*smooth + padded[:, 1:-1, 2:]
            planes = torch.stack((smooth, vertical+horizontal), dim=1).cpu().numpy()
            flags = codes.cpu().numpy()
            items.extend((plane[0], plane[1], int(flag), self.scale) for plane, flag in zip(planes, flags))
        return items

    @staticmethod
    def measure(item):
        from planetrecon.pipeline.preprocess import measure_filtered
        smooth, laplacian, code, scale = item
        values, status = (np.nan,)*4, 'invalid' if code == 2 else 'saturated'
        if code == 0:
            *values, status = measure_filtered(smooth, scale, laplacian)
            if status == 'ok' and not np.isfinite(values).all():
                status = 'invalid'
        return values, status
