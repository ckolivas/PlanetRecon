"""Non-production reference-coordinate CFA interpolation for local pull maps.

Fixed radius-four triweight and quadratic/affine noise-cap fallback. This is a
bounded CPU qualification probe, not a replacement for the application stacker.
"""
from dataclasses import replace
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree

from planetrecon.detector import cfa_labels
from planetrecon.pipeline.local_align import cpu_backproject
from planetrecon.pipeline.colour import complete_bayer_rgb
from planetrecon.result import ReconstructionResult


def completed(image, weights, color):
    return complete_bayer_rgb(ReconstructionResult(
        image, weights.copy(), weights > 0, 'adu', 'RGB', 'cpu', 'float64',
        'final', False, provenance={'color_mode': color}))


def inverse_pull_map(shift, shape):
    """Locate raw detector centres in the reference plane, with checked residuals.

    Solve r + displacement(r) = detector. Only the reference domain is defined;
    detector samples outside its image are excluded. A conservative contraction
    bound makes this inverse unique; unsupported maps are refused explicitly.
    """
    h, w = shape
    if min(shape) < 2:
        raise ValueError('local transport requires at least two rows and columns')
    dx, dy = (np.broadcast_to(np.asarray(v, dtype=float), shape) for v in shift)
    if not np.isfinite([dx, dy]).all():
        raise ValueError('finite local displacement required')
    # Bound every bilinear cell derivative, including both sides of grid knots.
    bound = np.sqrt(sum(float(np.max(np.abs(np.diff(v, axis=a))))**2
                        for v in (dx, dy) for a in (0, 1)))
    if bound >= .9:
        raise ValueError('local map is not qualified for a unique contractive inverse')
    yy, xx = np.indices(shape, dtype=float)
    rx, ry = xx-dx, yy-dy
    for _ in range(256):
        mx = map_coordinates(dx, [ry, rx], order=1, mode='nearest', prefilter=False)
        my = map_coordinates(dy, [ry, rx], order=1, mode='nearest', prefilter=False)
        nx, ny = xx-mx, yy-my
        error = max(np.max(np.abs(nx-rx)), np.max(np.abs(ny-ry)))
        rx, ry = nx, ny
        if error < 1e-11:
            break
    else:
        raise ValueError('local inverse did not converge')
    mx = map_coordinates(dx, [ry, rx], order=1, mode='nearest', prefilter=False)
    my = map_coordinates(dy, [ry, rx], order=1, mode='nearest', prefilter=False)
    valid = ((rx >= 0) & (rx <= w-1) & (ry >= 0) & (ry <= h-1)
             & (np.abs(rx+mx-xx) < 1e-9) & (np.abs(ry+my-yy) < 1e-9))
    return np.stack([rx, ry], axis=-1), valid


class LocalQuadraticAccumulator:
    def __init__(self, shape, color, *, chunk_rows=64, should_cancel=None, region=None):
        if len(shape) != 2 or min(shape) < 2:
            raise ValueError('two-dimensional detector of at least 2 by 2 required')
        if type(chunk_rows) is not int or chunk_rows < 1:
            raise ValueError('positive integer chunk_rows required')
        self.shape = tuple(shape)
        self.color = color
        self.chunk_rows = chunk_rows
        self.should_cancel = should_cancel
        self.labels = cfa_labels(*shape, color)
        self.region = (slice(0, shape[0]), slice(0, shape[1])) if region is None else region
        if (len(self.region) != 2 or any(not isinstance(v, slice) or v.step not in (None, 1)
                or v.start is None or v.stop is None or not 0 <= v.start < v.stop <= size
                for v, size in zip(self.region, shape))):
            raise ValueError('region must be two explicit in-bounds unit-step slices')
        y, x = (v[self.region] for v in np.indices(shape))
        self.output_shape = y.shape
        self.targets = np.stack([x.ravel(), y.ravel()], axis=-1)
        self.gram = np.zeros((y.size, 3, 6, 6))
        self.noise = np.zeros_like(self.gram)
        self.rhs = np.zeros((y.size, 3, 6))
        self.sums = np.zeros((*shape, 3))
        self.weights = np.zeros_like(self.sums)
        self.variance_sum = np.zeros_like(self.sums)
        self.n_used = 0

    def _check_cancel(self):
        if self.should_cancel is not None and self.should_cancel():
            raise InterruptedError('local interpolation cancelled before publishing frame')

    def add(self, raw, shift, quality):
        """Stage every contribution before publishing any part of a raw frame."""
        self._check_cancel()
        raw = np.asarray(raw, dtype=float)
        if raw.shape != self.shape or not np.isfinite(raw).all():
            raise ValueError('finite raw frame matching detector required')
        if not np.isfinite(quality) or quality <= 0:
            raise ValueError('positive finite quality required')
        positions, observed = inverse_pull_map(shift, self.shape)
        self._check_cancel()
        points = positions[observed]
        values = raw[observed]
        labels = self.labels[observed]
        tree = cKDTree(points)
        gram, noise, rhs = (np.zeros_like(v) for v in (self.gram, self.noise, self.rhs))
        for start in range(0, len(self.targets), self.chunk_rows):
            self._check_cancel()
            targets = self.targets[start:start+self.chunk_rows]
            neighbours = tree.query_ball_point(targets, 4., p=np.inf)
            for offset, (target, neighbours_at_target) in enumerate(zip(targets, neighbours)):
                ids = np.asarray(neighbours_at_target, dtype=int)
                u, v = ((points[ids]-target)/4).T
                phi = np.stack([np.ones_like(u), u, v, u*u, u*v, v*v], axis=-1)
                kernel = quality*np.maximum(1-u*u, 0)**3*np.maximum(1-v*v, 0)**3
                for c, name in enumerate('RGB'):
                    weight = kernel*(labels[ids] == name)
                    gram[start+offset, c] = (phi.T*weight)@phi
                    noise[start+offset, c] = (phi.T*weight**2)@phi
                    rhs[start+offset, c] = (phi.T*weight)@values[ids]
        add, weight, *_ = cpu_backproject(raw, shift, self.color)
        # Independent bilinear-coefficient variance for the ordinary fallback.
        h, w = self.shape
        yy, xx = np.indices(self.shape)
        sx, sy = xx+shift[0], yy+shift[1]
        ix, iy = np.floor(sx).astype(int), np.floor(sy).astype(int)
        fx, fy = sx-ix, sy-iy
        variance = np.zeros_like(self.variance_sum)
        for ox, wx in ((0, 1-fx), (1, fx)):
            for oy, wy in ((0, 1-fy), (1, fy)):
                x, y = ix+ox, iy+oy
                valid = (x >= 0) & (x < w) & (y >= 0) & (y < h)
                label = self.labels[y.clip(0, h-1), x.clip(0, w-1)]
                variance += ((quality*wx*wy)**2*valid)[..., None]*(label[..., None] == np.array(list('RGB')))
        self._check_cancel()
        self.gram += gram
        self.noise += noise
        self.rhs += rhs
        self.sums += quality*add
        self.weights += quality*weight
        self.variance_sum += variance
        self.n_used += 1

    def finish(self):
        if self.n_used == 0:
            raise ValueError('no complete frames')
        self._check_cancel()
        direct = np.divide(self.sums, self.weights, out=np.zeros_like(self.sums), where=self.weights > 0)
        before = completed(direct, self.weights, self.color)
        raw_variance = np.divide(self.variance_sum, self.weights**2,
                                 out=np.zeros_like(self.sums), where=self.weights > 0)
        variance = completed(raw_variance, self.weights, self.color).image[self.region].reshape(-1, 3).copy()
        before = replace(before, image=before.image[self.region].copy(),
                         coverage=before.coverage[self.region].copy(),
                         validity=before.validity[self.region].copy(),
                         layer_coverage={k: v[self.region].copy() for k, v in before.layer_coverage.items()})
        image = before.image.reshape(-1, 3).copy()
        degree = np.where(before.validity.reshape(-1, 3), 0, -1)
        for order, size in ((2, 6), (1, 3)):
            self._check_cancel()
            g = self.gram[..., :size, :size]
            eigen = np.linalg.eigvalsh(g)
            eligible = (degree == 0) & (eigen[..., 0] > eigen[..., -1]*1e-10)
            dual = np.zeros((*g.shape[:-1], 1))
            target = np.zeros((int(eligible.sum()), size, 1)); target[:, 0, 0] = 1
            dual[eligible] = np.linalg.solve(g[eligible], target)
            dual = dual[..., 0]
            predicted = np.einsum('nci,ncij,ncj->nc', dual, self.noise[..., :size, :size], dual)
            safe = eligible & (predicted >= 0) & (predicted <= variance*(1+1e-12))
            image[safe] = np.einsum('nci,nci->nc', dual, self.rhs[..., :size])[safe]
            variance[safe] = predicted[safe]
            degree[safe] = order
        self._check_cancel()
        # The caller retains ordinary direct layers; variance is not detector coverage.
        return image.reshape((*self.output_shape, 3)), variance.reshape((*self.output_shape, 3)), degree.reshape((*self.output_shape, 3)), before
