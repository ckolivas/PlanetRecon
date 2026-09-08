"""Exact native-cell Hessian diagonal for zero-translation scene observations.

This is a preconditioner candidate, NOT a proved step-size majorizer.
"""
import numpy as np
from scipy.signal import convolve, correlate
from tools.scene_parallel_reference import ParallelSceneBatch


def frame_diagonal(operator, weight):
    if hasattr(operator, 'base'):
        if operator.factor != 1: raise ValueError('native cells required for this diagonal')
        op = operator.base
    else:
        op = operator
    if any(any(v != 0 for v in shift) for shift in op.shifts_xy):
        raise ValueError('zero exposure translations required')
    if len({p.shape for p in op.psfs}) != 1: raise ValueError('common PSF shape required')
    weight = np.asarray(weight, dtype=float)
    if weight.shape != op.output_shape: raise ValueError('wrong weight shape')
    mask = op.valid_mask[..., None] if weight.ndim == 3 else op.valid_mask
    weight = np.where(mask, weight, 0.)
    if not np.isfinite(weight).all() or np.any(weight < 0): raise ValueError('finite nonnegative weights required')
    b = op.bin_factor
    psf = sum(w*p for w, p in zip(op.exposure_weights, op.psfs))
    # Integrate the pixel footprint BEFORE squaring: cross terms within a
    # detector pixel are part of diag(A' W A).
    integrated = convolve(psf, np.ones((b, b)), mode='full', method='direct')
    square = integrated**2
    shape = (op.scene_shape[0]+square.shape[0]-1, op.scene_shape[1]+square.shape[1]-1)
    kh, kw = psf.shape
    x0 = b*op.origin_xy[0]+(kw-1)//2+b-1
    y0 = b*op.origin_xy[1]+(kh-1)//2+b-1
    yy = y0+b*np.arange(op.detector_shape[0])
    xx = x0+b*np.arange(op.detector_shape[1])
    if yy[-1] >= shape[0] or xx[-1] >= shape[1]: raise ValueError('detector sampling outside convolution grid')
    channels = op.scene_shape[2] if len(op.scene_shape) == 3 else 1
    indices = op._cfa_indices() if op.cfa_pattern is not None else None
    output = []
    for channel in range(channels):
        if indices is not None:
            selected = np.where(indices == channel, weight, 0.)
        else:
            selected = weight[..., channel] if weight.ndim == 3 else weight
        impulses = np.zeros(shape)
        impulses[np.ix_(yy, xx)] = selected*op.flux**2
        diagonal = correlate(impulses, square, mode='valid', method='fft')
        # Exact diagonal is nonnegative; clamp only roundoff in zero FFT tails.
        output.append(np.maximum(diagonal, 0.))
    return output[0] if len(op.scene_shape) == 2 else np.stack(output, axis=-1)


def hessian_diagonal(problem, *, workers=8):
    batch = ParallelSceneBatch(problem.operators, workers=workers)
    diagonal = np.full(problem.shape, problem.ridge)
    degree = np.zeros(problem.shape)
    for axis in (0, 1):
        lo, hi = [slice(None)]*len(problem.shape), [slice(None)]*len(problem.shape)
        lo[axis], hi[axis] = slice(None, -1), slice(1, None)
        degree[tuple(lo)] += 1; degree[tuple(hi)] += 1
    diagonal += problem.smoothness*degree
    for term in batch._ordered(lambda pair: frame_diagonal(*pair), zip(problem.operators, problem.weights)):
        diagonal += term
    return diagonal, batch.cache_info()
