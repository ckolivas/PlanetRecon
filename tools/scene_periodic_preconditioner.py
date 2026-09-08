"""Positive periodic inverse used only to precondition the exact scene operator.

Replace finite detector sampling by uniform sampling density for this approximate
inverse. The physical forward model, gradient and certificate stay unchanged.
"""
import numpy as np
from scipy import fft
from scipy.signal import convolve
from tools.scene_parallel_reference import ParallelSceneBatch


def _spectrum_term(operator, weight):
    if hasattr(operator, 'base'):
        if operator.factor != 1: raise ValueError('native cells required')
        op = operator.base
    else:
        op = operator
    if any(any(v != 0 for v in shift) for shift in op.shifts_xy):
        raise ValueError('zero exposure translations required')
    if len({p.shape for p in op.psfs}) != 1: raise ValueError('common PSF shape required')
    weight = np.asarray(weight, dtype=float)
    if weight.shape != op.output_shape or not np.isfinite(weight).all() or np.any(weight < 0):
        raise ValueError('finite nonnegative detector weights required')
    b = op.bin_factor
    psf = sum(w*p for w,p in zip(op.exposure_weights, op.psfs))
    integrated = convolve(psf, np.ones((b,b)), mode='full', method='direct')
    shape = op.scene_shape[:2]
    folded = np.zeros(shape)
    if all(k <= n for k,n in zip(integrated.shape,shape)):
        folded[:integrated.shape[0], :integrated.shape[1]] = integrated
    else:
        yy,xx = np.indices(integrated.shape)
        np.add.at(folded, (yy % shape[0], xx % shape[1]), integrated)
    spectrum = np.abs(fft.rfft2(folded, workers=1))**2
    mask = op.valid_mask[...,None] if weight.ndim == 3 else op.valid_mask
    weight = np.where(mask,weight,0.)
    if len(op.scene_shape) == 3:
        if op.cfa_pattern is not None:
            indices = op._cfa_indices()
            mean = np.array([np.mean(np.where(indices == c, weight, 0.)) for c in range(3)])
        else:
            mean = np.mean(weight, axis=(0,1))
        spectrum = spectrum[...,None]*mean
    else:
        spectrum *= float(np.mean(weight))
    return spectrum*(op.flux**2/b**2)


class PeriodicScenePreconditioner:
    def __init__(self, problem, *, workers=8):
        self.shape = tuple(problem.shape)
        if not np.isfinite(problem.ridge) or problem.ridge <= 0:
            raise ValueError('positive finite ridge required')
        shape = (self.shape[0], self.shape[1]//2+1)+self.shape[2:]
        self.symbol = np.full(shape, problem.ridge)
        fy = np.fft.fftfreq(self.shape[0]); fx = np.fft.rfftfreq(self.shape[1])
        laplacian = 4*np.sin(np.pi*fy[:,None])**2+4*np.sin(np.pi*fx[None,:])**2
        if len(self.shape) == 3: laplacian = laplacian[...,None]
        self.symbol += problem.smoothness*laplacian
        batch = ParallelSceneBatch(problem.operators, workers=workers)
        for term in batch._ordered(lambda pair: _spectrum_term(*pair), zip(problem.operators, problem.weights)):
            self.symbol += term
        if not np.isfinite(self.symbol).all() or np.any(self.symbol <= 0):
            raise ValueError('invalid positive preconditioner symbol')
        self.workers = batch.cache_info()

    def __call__(self, residual):
        residual = np.asarray(residual,dtype=float)
        if residual.shape != self.shape or not np.isfinite(residual).all():
            raise ValueError('finite matching residual required')
        return fft.irfft2(fft.rfft2(residual,axes=(0,1),workers=1)/self.symbol,
                          s=self.shape[:2], axes=(0,1),workers=1)

    def info(self):
        return {'kind': 'periodic_uniform_sampling_inverse_v1',
                'symbol_min': float(self.symbol.min()), 'symbol_max': float(self.symbol.max()),
                'workers': self.workers,
                'scope': 'Approximate positive inverse for preconditioning only; finite crop, spatial weights and sampling aliases remain in the exact Hessian.'}
