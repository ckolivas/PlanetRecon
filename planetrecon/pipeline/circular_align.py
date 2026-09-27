"""Sampling-sized circular matching, coarse anchors and local residual refinement.

The eight-FWHM minimum is a tunable engineering starting point, not a theorem
about optimal alignment. FWHM = 1.028 lambda * (f-ratio / pixel pitch in um):
https://www.eso.org/observing/etc/doc/helpkmos.html
All coordinates here are full detector pixels, including the colour proxy.
"""
import math

import numpy as np
from scipy.ndimage import map_coordinates

from planetrecon.pipeline.local_align import LocalRegistration, pull


def minimum_diameter(sampling_multiplier=5., wavelength_nm=550.):
    """Round eight diffraction FWHMs up to an odd supported sample diameter."""
    for value in (sampling_multiplier, wavelength_nm):
        if not np.isfinite(value) or value <= 0:
            raise ValueError('positive finite sampling and wavelength required')
    diameter = max(15, math.ceil(8 * 1.028 * wavelength_nm / 1000 * sampling_multiplier))
    return diameter + (diameter % 2 == 0)


def _costs(image, y, x, template, weight, strength, m=3):
    """Bounded weighted NCC search, also used for the reverse consistency test."""
    w = weight.shape[0]
    h = w // 2
    tiles = np.lib.stride_tricks.sliding_window_view(
        image[y-h-m:y+h+m+1, x-h-m:x+h+m+1], (w, w))
    scores = np.empty((2*m+1, 2*m+1))
    rows = max(1, 262144 // ((2*m+1)*w*w))
    for start in range(0, 2*m+1, rows):
        block = tiles[start:start+rows]
        mean = np.einsum('ijxy,xy->ij', block, weight)
        centred = block - mean[..., None, None]
        variance = np.einsum('ijxy,ijxy,xy->ij', centred, centred, weight)
        numerator = np.einsum('ijxy,xy,xy->ij', centred, template, weight)
        scores[start:start+rows] = numerator / np.maximum(np.sqrt(variance)*strength, 1e-15)
    return scores


def _peak(scores):
    """Require a unique, curved two-dimensional subpixel correlation peak."""
    if not np.isfinite(scores).all():
        return None
    y, x = np.unravel_index(scores.argmax(), scores.shape)
    m = scores.shape[0] // 2
    peak = scores[y, x]
    if x in (0, 2*m) or y in (0, 2*m) or peak < .8:
        return None
    yy, xx = np.indices(scores.shape)
    distant = (xx-x)**2 + (yy-y)**2 >= 4
    if distant.any() and peak - scores[distant].max() < .005:
        return None
    p = scores[y-1:y+2, x-1:x+2]
    gradient = [.5*(p[1, 2]-p[1, 0]), .5*(p[2, 1]-p[0, 1])]
    cross = .25*(p[2, 2]-p[2, 0]-p[0, 2]+p[0, 0])
    curvature = -np.array([[p[1, 0]-2*peak+p[1, 2], cross],
                           [cross, p[0, 1]-2*peak+p[2, 1]]])
    if np.linalg.eigvalsh(curvature)[0] <= .001:
        return None
    delta = np.zeros(2) if peak >= 1-1e-10 else np.linalg.solve(curvature, gradient)
    if not np.isfinite(delta).all() or np.any(np.abs(delta) > 1):
        return None
    return np.array([x-m, y-m]) + delta


class CircularMultiscaleRegistration:
    """Prefer fine reliable measurements; retain coarse motion elsewhere.

    Overlapping samples form a normalised field, never independent pixel votes.
    Only matching proxies are warped between scales; raw data are resampled by
    the stacker once using the final displacement field.
    """
    def __init__(self, reference, *, sampling_multiplier=5., wavelength_nm=550.,
                 use_cuda=False, should_cancel=None, valid_mask=None):
        reference = np.asarray(reference, dtype=float)
        if reference.ndim != 2 or min(reference.shape) < 5 or not np.isfinite(reference).all():
            raise ValueError('finite two-dimensional reference of at least 5 by 5 required')
        self.shape = reference.shape
        self.use_cuda = use_cuda
        self._cuda = None
        self.should_cancel = should_cancel
        self.minimum = minimum_diameter(sampling_multiplier, wavelength_nm)
        self.layers = []
        # Four overlapping scales cover a factor of ~2.8 in diameter. Never
        # shrink below the requested minimum merely to fit a small capture.
        for scale in range(4):
            self._check_cancel()
            size = math.ceil(self.minimum * 2**(scale/2))
            size += size % 2 == 0
            if size + 6 > min(self.shape):
                break
            self.layers.append(LocalRegistration(reference, window=size,
                               step=max(1, size//2), circular=True, valid_mask=valid_mask))
        self.provenance = {
            'version': 1, 'method': 'circular_multiscale',
            'sampling_multiplier': sampling_multiplier, 'wavelength_nm': wavelength_nm,
            'minimum_diameter_px': self.minimum, 'diameters_px': [p.window for p in self.layers],
            'steps_px': [p.step for p in self.layers],
            'minimum_rule': '8 * 1.028 * wavelength_um * sampling_multiplier, odd ceil; floor 15',
            'coordinate_system': 'full detector pixels',
            'matching_weight': 'radial cosine, zero outside circle',
            'contribution_weight': 'flat inner half-radius, outer cosine taper; normalised overlap',
            'refinement': 'coarse to fine residual; smallest reliable scale; coarse/global fallback',
            'gates': '2D texture, NCC >= 0.8, unique curved peak, reverse error <= 0.75 px',
            'maximum_residual_per_scale_px': 3,
            'maximum_total_residual_px': 6,
            'fold_guard': 'minimum Jacobian 0.25; reject scale if violated',
            'eligible_points': [int(np.count_nonzero(np.asarray(p.texture_valid)
                                & (p.strength >= max(p.strength.max()*.08, 1e-12))))
                                for p in self.layers],
            'cuda_execution': {'used': False},
        }

    def _check_cancel(self):
        if self.should_cancel is not None and self.should_cancel():
            raise InterruptedError('cancelled during circular alignment')

    def release_cuda(self):
        self._cuda = None

    def displacement_tensor(self, frame, global_shift):
        """Keep the final field on CUDA for direct raw-data backprojection."""
        frame = np.asarray(frame, dtype=float)
        if frame.shape != self.shape or not np.isfinite(frame).all():
            raise ValueError('finite frame matching reference required')
        if np.shape(global_shift) != (2,) or not np.isfinite(global_shift).all():
            raise ValueError('finite global displacement required')
        from planetrecon.backends.torch_circular import TorchCircularRegistration
        self._check_cancel()
        if self._cuda is None:
            self._cuda = TorchCircularRegistration(self)
            self.provenance['cuda_execution'].update(used=True, precision='float64',
                matching='GPU forward and reverse correlations, peaks, field blending and composition',
                fused_correlation=any(layer['fused'] for layer in self._cuda.layers),
                field_transfer='device field passed directly to raw backprojection')
        return self._cuda.displacement(LocalRegistration.proxy(frame), global_shift, self._check_cancel)

    def displacement(self, frame, global_shift):
        frame = np.asarray(frame, dtype=float)
        if frame.shape != self.shape or not np.isfinite(frame).all():
            raise ValueError('finite frame matching reference required')
        if np.shape(global_shift) != (2,) or not np.isfinite(global_shift).all():
            raise ValueError('finite global displacement required')
        if self.use_cuda:
            return tuple(self.displacement_tensor(frame, global_shift).cpu().numpy())
        field = np.array([np.full(self.shape, value, dtype=float) for value in global_shift])
        yy, xx = np.indices(self.shape)
        proxy = LocalRegistration.proxy(frame)
        for layer in reversed(self.layers):
            self._check_cancel()
            image = pull(proxy, field)
            # Include bilinear support, not just the patch centre. A masked or
            # padded border must not create an apparently excellent match.
            observed = ((xx+field[0] >= 0) & (xx+field[0] <= self.shape[1]-1)
                        & (yy+field[1] >= 0) & (yy+field[1] <= self.shape[0]-1))
            scores_gpu = layer._cuda_costs(image) if self.use_cuda else None
            numerator = np.zeros_like(field)
            denominator = np.zeros(self.shape)
            h, m = layer.window//2, layer.max_shift
            fy, fx = np.indices((layer.window, layer.window)) - h
            radius = np.hypot(fx, fy) / h
            # A broader footprint than the matching taper maintains coverage
            # between centres; a uniform shift must not acquire grid ripples.
            footprint = .5*(1 + np.cos(np.pi*np.clip(2*radius-1, 0., 1.)))
            for j, y in enumerate(layer.ys):
                self._check_cancel()
                for i, x in enumerate(layer.xs):
                    k = j*len(layer.xs)+i
                    if (not layer.texture_valid[k]
                            or layer.strength[k] < max(layer.strength.max()*.08, 1e-12)
                            or not observed[y-h-m:y+h+m+1, x-h-m:x+h+m+1].all()):
                        continue
                    scores = (scores_gpu[k] if scores_gpu is not None else
                              _costs(image, y, x, layer.templates[k], layer.weight, layer.strength[k]))
                    delta = _peak(scores)
                    if delta is None:
                        continue
                    # Match the observed subpixel patch back to the reference.
                    # A valid forward displacement must return to its origin.
                    py, px = np.indices((layer.window, layer.window), dtype=float)
                    patch = map_coordinates(image, [py+y-h+delta[1], px+x-h+delta[0]],
                                            order=1, mode='constant', prefilter=False)
                    patch -= (patch*layer.weight).sum()
                    strength = np.sqrt((patch*patch*layer.weight).sum())
                    reverse = _peak(_costs(layer.ref, y, x, patch, layer.weight, strength))
                    if reverse is None or np.linalg.norm(reverse) > .75:
                        continue
                    section = np.s_[y-h:y+h+1, x-h:x+h+1]
                    confidence = np.clip((scores.max()-.8)/.15, 0., 1.)
                    weight = footprint * confidence
                    numerator[:, section[0], section[1]] += delta[:, None, None]*weight
                    denominator[section] += weight
            residual = np.divide(numerator, denominator[None], out=np.zeros_like(field),
                                 where=denominator[None] > 0)
            # Fade sparse footprint edges into the parent; overlapping points
            # are averaged, not accumulated as repeated evidence.
            residual *= np.minimum(denominator, 1.)
            # Residuals are in the parent-warped proxy coordinates. Compose
            # maps, rather than adding displacements on mismatched coordinates.
            coords = [yy+residual[1], xx+residual[0]]
            proposed = np.array([residual[a] + map_coordinates(field[a], coords,
                                order=1, mode='nearest', prefilter=False) for a in range(2)])
            ux, uy = proposed - np.asarray(global_shift)[:, None, None]
            jacobian = ((1+np.gradient(ux, axis=1))*(1+np.gradient(uy, axis=0))
                        - np.gradient(ux, axis=0)*np.gradient(uy, axis=1))
            if (np.isfinite(proposed).all() and jacobian.min() >= .25
                    and np.max(np.hypot(ux, uy)) <= 6.):
                field = proposed
        return tuple(field)
