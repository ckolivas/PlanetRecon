"""Shared, bounded-cache Fourier evaluation of fixed-PSF scene operators.

Exact fast path for zero reference translations and common native scene/kernel
shapes. CPU uses float64/complex128, as does the optional Torch CUDA backend.
The cache limit covers retained PSF spectra only, not total RAM/VRAM.
"""
from collections import OrderedDict
import numpy as np
from scipy import fft


class SceneFFTBatch:
    def __init__(self, operators, *, cache_bytes=256*1024**2, device='cpu'):
        self.operators = tuple(operators)
        if not self.operators:
            raise ValueError('operators required')
        if int(cache_bytes) != cache_bytes or cache_bytes < 0:
            raise ValueError('cache_bytes must be a nonnegative integer')
        self.shape = self.operators[0].scene_shape
        self.kernel_shape = self.operators[0].psfs[0].shape
        for op in self.operators:
            if op.scene_shape != self.shape or any(p.shape != self.kernel_shape for p in op.psfs):
                raise ValueError('common scene and kernel shapes required')
            if any(any(v != 0 for v in s) for s in op.shifts_xy):
                raise ValueError('translated exposure samples require the reference operator')
        self.fft_shape = tuple(fft.next_fast_len(n+k-1, real=True) for n, k in zip(self.shape[:2], self.kernel_shape))
        self.cache_bytes = int(cache_bytes)
        self.cache = OrderedDict()
        self.resident_bytes = self.peak_cache_bytes = self.hits = self.misses = 0
        self.device = device
        self.torch = None
        if device != 'cpu':
            if not device.startswith('cuda'):
                raise ValueError('device must be cpu or cuda')
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA unavailable; request CPU explicitly')
            self.torch = torch
            # Execute a real float64 FFT, not only a CUDA availability query.
            self._fft(self._array(np.ones((2, 2))))
            self.synchronize()

    def _array(self, value):
        if self.torch is None:
            return np.asarray(value, dtype=np.float64)
        return self.torch.as_tensor(np.array(value, dtype=np.float64, copy=True), device=self.device)

    def _numpy(self, value):
        return np.asarray(value) if self.torch is None else value.detach().cpu().numpy()

    def _zeros(self, shape, *, complex=False):
        if self.torch is None:
            return np.zeros(shape, dtype=np.complex128 if complex else np.float64)
        return self.torch.zeros(shape, dtype=self.torch.complex128 if complex else self.torch.float64, device=self.device)

    def _fft(self, image):
        if self.torch is None:
            return fft.rfft2(image, s=self.fft_shape, axes=(0, 1), workers=1)
        return self.torch.fft.rfft2(image, s=self.fft_shape, dim=(0, 1))

    def _inverse(self, spectrum):
        if self.torch is None:
            return fft.irfft2(spectrum, s=self.fft_shape, axes=(0, 1), workers=1)
        return self.torch.fft.irfft2(spectrum, s=self.fft_shape, dim=(0, 1))

    def synchronize(self):
        if self.torch is not None:
            self.torch.cuda.synchronize(self.device)

    def clear_cache(self):
        self.cache.clear()
        self.resident_bytes = 0

    def cache_info(self):
        return {'limit_bytes': self.cache_bytes, 'resident_bytes': self.resident_bytes,
                'peak_bytes': self.peak_cache_bytes, 'hits': self.hits, 'misses': self.misses,
                'scope': 'retained PSF spectra only; excludes operators, workspaces, outputs and driver allocations'}

    def _spectrum(self, index):
        if index in self.cache:
            self.hits += 1
            self.cache.move_to_end(index)
            return self.cache[index]
        self.misses += 1
        op = self.operators[index]
        psf = sum(w*p for w, p in zip(op.exposure_weights, op.psfs))
        spectrum = self._fft(self._array(psf))
        size = int(np.prod(spectrum.shape))*16
        if size <= self.cache_bytes:
            while self.resident_bytes+size > self.cache_bytes:
                _, removed = self.cache.popitem(last=False)
                self.resident_bytes -= int(np.prod(removed.shape))*16
            self.cache[index] = spectrum
            self.resident_bytes += size
            self.peak_cache_bytes = max(self.peak_cache_bytes, self.resident_bytes)
        return spectrum

    def _crop_slices(self, op):
        x, y = op.origin_xy
        b = op.bin_factor
        ky, kx = self.kernel_shape
        sy, sx = (ky-1)//2+b*y, (kx-1)//2+b*x
        return (slice(sy, sy+b*op.detector_shape[0]), slice(sx, sx+b*op.detector_shape[1]))

    def _mask(self, op, image):
        mask = self._array(op.valid_mask)
        return image*(mask[..., None] if image.ndim == 3 else mask)

    def _detector(self, optical, op):
        cut = optical[self._crop_slices(op)]
        h, w = op.detector_shape
        b = op.bin_factor
        shape = (h, b, w, b)+self.shape[2:]
        cut = cut.reshape(shape)
        det = cut.sum(axis=(1, 3)) if self.torch is None else cut.sum(dim=(1, 3))
        if op.cfa_pattern is not None:
            idx = op._cfa_indices()
            if self.torch is None:
                det = np.take_along_axis(det, idx[..., None], axis=2)[..., 0]
            else:
                ix = self.torch.as_tensor(idx.copy(), device=self.device, dtype=self.torch.int64)
                det = self.torch.gather(det, 2, ix[..., None])[..., 0]
        return op.flux*self._mask(op, det)

    def _embed(self, residual, op):
        residual = self._mask(op, residual)*op.flux
        if op.cfa_pattern is not None:
            rgb = self._zeros(op.detector_shape+(3,))
            idx = op._cfa_indices()
            if self.torch is None:
                np.put_along_axis(rgb, idx[..., None], residual[..., None], axis=2)
            else:
                ix = self.torch.as_tensor(idx.copy(), device=self.device, dtype=self.torch.int64)
                rgb.scatter_(2, ix[..., None], residual[..., None])
            residual = rgb
        b = op.bin_factor
        if self.torch is None:
            expanded = np.repeat(np.repeat(residual, b, axis=0), b, axis=1)
        else:
            expanded = residual.repeat_interleave(b, dim=0).repeat_interleave(b, dim=1)
        padded = self._zeros(self.fft_shape+self.shape[2:])
        padded[self._crop_slices(op)] = expanded
        return padded

    def _check_image(self, x):
        if np.shape(x) != self.shape:
            raise ValueError('wrong scene shape')

    def forward(self, scene):
        self._check_image(scene)
        sf = self._fft(self._array(scene))
        output = []
        for i, op in enumerate(self.operators):
            h = self._spectrum(i)
            if len(self.shape) == 3:
                h = h[..., None]
            output.append(self._numpy(self._detector(self._inverse(sf*h), op)))
        return output

    def adjoint(self, residuals):
        if len(residuals) != len(self.operators):
            raise ValueError('one residual per operator required')
        acc = self._zeros((self.fft_shape[0], self.fft_shape[1]//2+1)+self.shape[2:], complex=True)
        for i, (op, residual) in enumerate(zip(self.operators, residuals)):
            if np.shape(residual) != op.output_shape:
                raise ValueError('wrong detector shape')
            h = self._spectrum(i)
            if len(self.shape) == 3:
                h = h[..., None]
            acc += h.conj()*self._fft(self._embed(self._array(residual), op))
        return self._numpy(self._inverse(acc)[:self.shape[0], :self.shape[1]])

    def normal(self, scene, weights):
        self._check_image(scene)
        if len(weights) != len(self.operators):
            raise ValueError('one weight per operator required')
        sf = self._fft(self._array(scene))
        acc = self._zeros(tuple(sf.shape), complex=True)
        for i, (op, weight) in enumerate(zip(self.operators, weights)):
            h = self._spectrum(i)
            if len(self.shape) == 3:
                h = h[..., None]
            predicted = self._detector(self._inverse(sf*h), op)
            weighted = predicted*self._array(weight)
            acc += h.conj()*self._fft(self._embed(weighted, op))
        return self._numpy(self._inverse(acc)[:self.shape[0], :self.shape[1]])
