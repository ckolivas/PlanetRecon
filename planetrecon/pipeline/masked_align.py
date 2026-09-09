"""Normalized translation matching restricted to observed reference pixels."""

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.signal import correlate


class MaskedRegistration:
    """Linear correlation with real source support and distinct-peak guards."""

    def __init__(self, reference, mask):
        reference = np.asarray(reference, dtype=float)
        self.shape = reference.shape
        self.mask = np.asarray(mask, dtype=bool).copy()
        self.count = int(self.mask.sum())
        proxy = gaussian_filter(reference, 1.5)
        self.template = np.zeros(self.shape)
        self.energy = 0.
        if self.count >= 32:
            self.template[self.mask] = proxy[self.mask] - proxy[self.mask].mean()
            self.energy = float(np.sum(self.template**2))

    def displacement(self, frame, *, min_improvement=0.):
        """Return a supported displacement, or None when observed pixels do not constrain it."""
        if self.energy <= 1e-12:
            return None
        proxy = gaussian_filter(np.asarray(frame, dtype=float), 1.5)
        mask = self.mask.astype(float)
        # Linear correlations: padding is never treated as observed dark sky.
        dot = correlate(proxy, self.template, mode='full', method='fft')
        total = correlate(proxy, mask, mode='full', method='fft')
        squares = correlate(proxy**2, mask, mode='full', method='fft')
        support = correlate(np.ones(self.shape), mask, mode='full', method='fft')
        variance = np.maximum(squares - total**2/self.count, 0.)
        denominator = np.sqrt(self.energy*variance)
        valid = (support >= self.count - 1e-6) & (denominator > 1e-12)
        scores = np.divide(dot, denominator, out=np.full(dot.shape, -np.inf), where=valid)
        py, px = np.unravel_index(np.argmax(scores), scores.shape)
        peak = scores[py, px]
        if not np.isfinite(peak) or peak < .8:
            return None
        # Distinct peaks separated by more than the smoothing footprint must
        # not explain the same template equally well.
        competitors = scores.copy()
        competitors[max(0, py-3):py+4, max(0, px-3):px+4] = -np.inf
        if competitors.max() >= peak - .02:
            return None
        offsets = []
        for axis, p in enumerate((py, px)):
            if p == 0 or p == scores.shape[axis]-1:
                return None
            left, right = (scores[[py-1, py+1], px] if axis == 0 else
                           scores[py, [px-1, px+1]])
            curvature = left - 2*peak + right
            if not np.isfinite(curvature) or curvature >= 0:
                return None
            # A perfect normalized match is already an exact solution.
            delta = (0. if peak >= 1-1e-12 else
                     float(np.clip(.5*(left-right)/curvature, -.5, .5)))
            offsets.append(p - (self.shape[axis]-1) + delta)
        if min_improvement > 0:
            y, x = np.nonzero(self.mask)
            moved = map_coordinates(proxy, [y+offsets[0], x+offsets[1]],
                                    order=1, mode='constant', prefilter=False)
            moved -= moved.mean()
            norm = np.sqrt(self.energy*np.sum(moved**2))
            gain = (np.dot(self.template[self.mask], moved)/norm
                    - scores[self.shape[0]-1, self.shape[1]-1]) if norm > 0 else -np.inf
            if gain <= min_improvement:
                return 0., 0.
        return offsets[1], offsets[0]
