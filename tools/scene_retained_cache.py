"""Experimental fixed-admission PSF cache for repeated full-sequence sweeps.

When the complete sequence does not fit, retain the first fitting spectra and
evaluate other frames transiently. Unlike LRU, a sequential sweep cannot evict
all entries before their next use. This changes retention only, not arithmetic.
"""
import numpy as np
from scipy import fft
from tools.scene_fft import SceneFFTBatch
from tools.scene_study import CellFFTBatch


def spectrum_storage(scene_shape, kernel_shape, frames):
    if len(scene_shape) not in (2, 3) or len(kernel_shape) != 2 or frames < 1 or int(frames) != frames:
        raise ValueError('scene, kernel and positive integer frame count required')
    if any(int(v) != v or v < 1 for v in (*scene_shape, *kernel_shape)):
        raise ValueError('positive integer dimensions required')
    shape = tuple(fft.next_fast_len(n+k-1, real=True) for n,k in zip(scene_shape[:2], kernel_shape))
    per_frame = shape[0]*(shape[1]//2+1)*16
    return {'fft_shape': list(shape), 'per_spectrum_bytes': per_frame,
            'all_spectra_bytes': per_frame*frames,
            'scope': 'Shared spatial PSF spectra in complex128; excludes workspaces, scene arrays and CUDA runtime.'}


class RetainedSceneFFTBatch(SceneFFTBatch):
    def _spectrum(self, index):
        if index in self.cache:
            self.hits += 1
            return self.cache[index]
        self.misses += 1
        op = self.operators[index]
        psf = sum(w*p for w, p in zip(op.exposure_weights, op.psfs))
        spectrum = self._fft(self._array(psf))
        size = int(np.prod(spectrum.shape))*16
        if self.resident_bytes+size <= self.cache_bytes:
            self.cache[index] = spectrum
            self.resident_bytes += size
            self.peak_cache_bytes = max(self.peak_cache_bytes, self.resident_bytes)
        return spectrum

    def cache_info(self):
        return super().cache_info() | {'policy': 'retain-first', 'retained_frames': len(self.cache)}


class RetainedCellFFTBatch(CellFFTBatch):
    def __init__(self, operators, **kwargs):
        self.operators = tuple(operators)
        if not self.operators or len({op.factor for op in self.operators}) != 1:
            raise ValueError('common cell factor required')
        self.factor = self.operators[0].factor
        self.native = RetainedSceneFFTBatch([op.base for op in self.operators], **kwargs)
