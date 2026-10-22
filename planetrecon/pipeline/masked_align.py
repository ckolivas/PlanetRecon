"""Normalized translation matching restricted to observed reference pixels."""

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates, maximum_filter
from scipy.fft import rfft2, irfft2, next_fast_len


def _fft_workers():
    """The job's configured CPU thread limit (see runtime.apply_thread_limits)."""
    import os
    from planetrecon.runtime import default_thread_count
    limit = os.environ.get('PLANETRECON_THREADS', '')
    return max(1, int(limit)) if limit.isdigit() else default_thread_count()


def observed_correlations(proxy, template, mask):
    """Four linear correlations, sharing the repeated proxy and mask FFTs."""
    full = tuple(2*n-1 for n in proxy.shape)
    shape = tuple(next_fast_len(n, real=True) for n in full)
    # Multithreaded transforms give the same values; the job's thread limit applies.
    workers = _fft_workers()
    observed = rfft2(proxy, s=shape, workers=workers)
    kernel = rfft2(mask[::-1,::-1], s=shape, workers=workers)
    def inverse(product):
        return irfft2(product, s=shape, workers=workers)[:full[0],:full[1]]
    dot = inverse(observed*rfft2(template[::-1,::-1], s=shape, workers=workers))
    total = inverse(observed*kernel)
    squares = inverse(rfft2(proxy**2, s=shape, workers=workers)*kernel)
    support = inverse(rfft2(np.ones(proxy.shape), s=shape, workers=workers)*kernel)
    return dot,total,squares,support


def peak_window(scores, py, px):
    """5x5 scores centred on (py, px); -inf outside the array."""
    window = np.full((5, 5), -np.inf)
    y0, x0 = max(0, py-2), max(0, px-2)
    y1, x1 = min(scores.shape[0], py+3), min(scores.shape[1], px+3)
    window[y0-py+2:y1-py+2, x0-px+2:x1-px+2] = scores[y0:y1, x0:x1]
    return window


def quadratic_peak(window):
    """Sub-pixel maximum (dy, dx) from the centre of a 5x5 score window, or None.

    The centre is the best grid score. Its 3x3 quadratic fit must be a
    maximum. On a flat, tilted ridge (thin rings in a dim frame) the fitted
    summit can lie just beyond the neighbouring grid point; then fit once more
    around the neighbour nearest it, which must contain the summit.
    """
    def fit(cy, cx):
        n = window[cy-1:cy+2, cx-1:cx+2]
        if not np.isfinite(n).all():
            return None
        # Tilted features couple horizontal and vertical offsets. Independent
        # axis fits find two different slices of the peak, not its joint centre.
        centre = n[1, 1]
        gy = .5*(n[2, 1]-n[0, 1])
        gx = .5*(n[1, 2]-n[1, 0])
        hyy = n[0, 1]-2*centre+n[2, 1]
        hxx = n[1, 0]-2*centre+n[1, 2]
        hxy = .25*(n[2, 2]-n[2, 0]-n[0, 2]+n[0, 0])
        hessian = np.array([[hyy, hxy], [hxy, hxx]])
        if np.linalg.eigvalsh(hessian)[-1] >= 0:
            return None
        # A perfect normalized match is already an exact solution.
        delta = np.zeros(2) if centre >= 1-1e-12 else np.linalg.solve(-hessian, [gy, gx])
        return delta if np.isfinite(delta).all() else None
    delta = fit(2, 2)
    if delta is None or np.all(np.abs(delta) <= 1.):
        return delta
    step = np.clip(np.rint(delta), -1, 1).astype(int)
    refit = fit(2+step[0], 2+step[1])
    if refit is None or np.any(np.abs(refit) > 1.):
        return None
    return step + refit


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

    def displacement(self, frame, *, min_improvement=0., distinct_peaks=False):
        """Return a supported displacement, or None when observed pixels do not constrain it."""
        if self.energy <= 1e-12:
            return None
        proxy = gaussian_filter(np.asarray(frame, dtype=float), 1.5)
        mask = self.mask.astype(float)
        # Linear correlations: padding is never treated as observed dark sky.
        dot,total,squares,support = observed_correlations(proxy,self.template,mask)
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
        if distinct_peaks:
            # A well-sampled planet can produce one broad, unique maximum.
            # Its shoulders outside the fixed exclusion box are not competing
            # translations. Keep plateaus and truly separate maxima eligible.
            competitors[scores < maximum_filter(scores, size=3)] = -np.inf
        competitors[max(0, py-3):py+4, max(0, px-3):px+4] = -np.inf
        if competitors.max() >= peak - .02:
            return None
        if py == 0 or px == 0 or py == scores.shape[0]-1 or px == scores.shape[1]-1:
            return None
        delta = quadratic_peak(peak_window(scores, py, px))
        if delta is None:
            return None
        offsets = np.array([py, px]) - (np.array(self.shape)-1) + delta
        if min_improvement > 0:
            y, x = np.nonzero(self.mask)
            moved = map_coordinates(proxy, [y+offsets[0], x+offsets[1]],
                                    order=1, mode='constant', prefilter=False)
            moved -= moved.mean()
            norm = np.sqrt(self.energy*np.sum(moved**2))
            gain = (np.einsum('i,i->', self.template[self.mask], moved)/norm
                    - scores[self.shape[0]-1, self.shape[1]-1]) if norm > 0 else -np.inf
            if gain <= min_improvement:
                return 0., 0.
        return offsets[1], offsets[0]
