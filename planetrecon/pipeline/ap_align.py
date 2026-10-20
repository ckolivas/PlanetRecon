"""AutoStakkert-style alignment points, local quality and blended local selection.

A grid of square, Hann-tapered alignment points (APs) is placed on bright,
textured regions of a low-noise reference. Each frame is first registered
globally; every AP then searches +/-SEARCH pixels on the band-passed proxy for
its normalised cross-correlation peak, refined by a two-dimensional quadratic.

The displacement field is a per-pixel weighted least-squares fit of the AP
shifts, each weighted by its correlation-peak curvature (information matrix):
an AP on a ring edge or limb constrains only the direction across that edge.
A weak prior returns the field to the global shift away from accepted APs.
Accepted shifts are used at full strength (no shrinkage toward global motion).

Local quality ranks frames separately for each AP; per-pixel frame weights
blend the AP selections with normalised Gaussian footprints. Outside the AP
cloud the global frame selection applies.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from planetrecon.pipeline.local_align import LocalRegistration

SEARCH = 6
MIN_NCC = 0.5
PRIOR = 0.01
FIELD_STEP = 4
REFERENCE_FRAMES = 3000  # best selected frames averaged into the reference
MIN_BRIGHTNESS = 0.1
QUALITY_SMOOTH_PX = 1.5


def _grid(length, step):
    """Coarse field sample coordinates spanning the detector exactly."""
    count = max(2, int(np.ceil((length - 1) / step)) + 1)
    return np.linspace(0., length - 1., count)


def _gaussians(coords, centres, sigma):
    return np.exp(-.5 * ((coords[None, :] - centres[:, None]) / sigma) ** 2)


def _peak_fit(c):
    """Subpixel peaks, information matrices and acceptance for NCC surfaces.

    c has shape (n, 2R+1, 2R+1). Returns offset (n, 2) in (x, y) relative to
    the search centre, information matrix (n, 2, 2), acceptance (n,), peak (n,).
    """
    n, m2, _ = c.shape
    R = m2 // 2
    flat = c.reshape(n, -1)
    finite = np.isfinite(flat).all(axis=1)
    flat = np.where(np.isfinite(flat), flat, -np.inf)
    index = flat.argmax(axis=1)
    peak = flat[np.arange(n), index]
    py, px = index // m2, index % m2
    interior = (px > 0) & (px < m2 - 1) & (py > 0) & (py < m2 - 1)
    cy, cx = np.clip(py, 1, m2 - 2), np.clip(px, 1, m2 - 2)
    rows = np.arange(n)
    c = np.where(np.isfinite(c), c, 0.)

    def at(dy, dx):
        return c[rows, cy + dy, cx + dx]
    p0 = at(0, 0)
    gx, gy = .5 * (at(0, 1) - at(0, -1)), .5 * (at(1, 0) - at(-1, 0))
    hxx, hyy = at(0, 1) - 2 * p0 + at(0, -1), at(1, 0) - 2 * p0 + at(-1, 0)
    hxy = .25 * (at(1, 1) - at(1, -1) - at(-1, 1) + at(-1, -1))
    H = -np.stack([np.stack([hxx, hxy], -1), np.stack([hxy, hyy], -1)], -2)
    evals, evecs = np.linalg.eigh(H)
    evals = np.clip(evals, 0., None)
    ok = finite & interior & (peak >= MIN_NCC) & (evals[:, 1] > 0)
    # Newton step restricted to the constrained directions: a straight edge
    # leaves the along-edge coordinate to neighbouring points and the prior.
    usable = evals > 1e-6 * np.maximum(evals[:, 1:2], 1e-30)
    inverse = np.where(usable, 1. / np.maximum(evals, 1e-30), 0.)
    g = np.stack([gx, gy], -1)
    delta = np.einsum('nij,nj->ni', evecs, np.einsum('nij,ni->nj', evecs, g) * inverse)
    ok &= (np.abs(delta) <= 1).all(axis=1)
    offset = np.stack([cx - R, cy - R], -1) + delta
    H = np.einsum('nij,nj,nkj->nik', evecs, evals * usable, evecs) * ok[:, None, None]
    return np.where(ok[:, None], offset, 0.), H, ok, peak


class AlignmentPoints:
    """Local registration with an AutoStakkert-style AP grid.

    ``displacement`` returns a dense (x, y) sampling field in detector pixels,
    the convention of ``local_align.pull`` and the CUDA backprojection.
    """
    def __init__(self, reference, *, size=64, search=SEARCH, use_cuda=False, should_cancel=None):
        reference = np.asarray(reference, dtype=np.float64)
        if reference.ndim != 2 or not np.isfinite(reference).all():
            raise ValueError('finite two-dimensional reference required')
        if type(size) is not int or not 16 <= size <= 256:
            raise ValueError('alignment point size must be an integer from 16 to 256 pixels')
        self.shape = reference.shape
        self.size, self.search = size, search
        self.spacing = size // 2
        self.sigma = .6 * self.spacing
        self.use_cuda = use_cuda
        self.should_cancel = should_cancel
        self._cuda = None
        h, w = self.shape
        half = size // 2
        margin = half + search
        proxy = LocalRegistration.proxy(reference)
        window = np.outer(np.hanning(size), np.hanning(size))
        level = reference.max() * MIN_BRIGHTNESS
        centres, templates, energy = [], [], []
        for y in range(margin, h - margin - size % 2, self.spacing):
            for x in range(margin, w - margin - size % 2, self.spacing):
                section = np.s_[y - half:y - half + size, x - half:x - half + size]
                if reference[section].mean() < level:
                    continue
                t = proxy[section] - (proxy[section] * window).sum() / window.sum()
                centres.append((y, x))
                templates.append(t)
                energy.append(np.sqrt((t * t * window).sum() / window.sum()))
        energy = np.asarray(energy)
        # Relative to the typical point, with an absolute floor for blank regions.
        keep = ((energy >= .1 * np.median(energy)) & (energy > 1e-4 * abs(reference.max()))
                if len(energy) else np.zeros(0, bool))
        self.centres = np.asarray(centres, dtype=np.int64).reshape(-1, 2)[keep]
        self.templates = np.asarray(templates, dtype=np.float64).reshape(-1, size, size)[keep]
        self.window = window
        self.weighted = self.templates * window
        self.norm = np.sqrt((self.templates ** 2 * window).sum(axis=(1, 2)))
        self.gy = _grid(h, FIELD_STEP)
        self.gx = _grid(w, FIELD_STEP)
        self.wy = _gaussians(self.gy, self.centres[:, 0].astype(float), self.sigma)
        self.wx = _gaussians(self.gx, self.centres[:, 1].astype(float), self.sigma)
        self.accepted = []
        self.provenance = {
            'version': 1, 'method': 'alignment_points',
            'ap_size_px': size, 'spacing_px': self.spacing, 'points': int(len(self.centres)),
            'search_px': search, 'window': 'Hann (square)',
            'proxy': 'Gaussian 1.5 px minus Gaussian 8 px',
            'placement': f'grid at half the AP size; mean reference >= {MIN_BRIGHTNESS:g} of maximum; '
                         'proxy texture >= 0.1 of the median AP and 1e-4 of the reference maximum',
            'acceptance': f'interior NCC peak >= {MIN_NCC:g}, positive peak curvature, subpixel step <= 1 px',
            'field': f'curvature-weighted least squares, Gaussian sigma {self.sigma:g} px, '
                     f'prior {PRIOR:g} x median curvature toward global motion; {FIELD_STEP} px grid, bilinear',
            'cuda_execution': {'used': False},
        }

    @property
    def layers(self):
        """Compatibility with the circular matcher's 'no usable points' check."""
        return [self] if len(self.centres) else []

    def _check_cancel(self):
        if self.should_cancel is not None and self.should_cancel():
            raise InterruptedError('cancelled during alignment point registration')

    def _ncc(self, image):
        size, R = self.size, self.search
        half = size // 2
        span = size + 2 * R
        out = np.empty((len(self.centres), 2 * R + 1, 2 * R + 1))
        wsum = self.window.sum()
        for k, (y, x) in enumerate(self.centres):
            tile = image[y - half - R:y - half - R + span, x - half - R:x - half - R + span]
            cand = np.lib.stride_tricks.sliding_window_view(tile, (size, size))
            s1 = np.einsum('ijxy,xy->ij', cand, self.window)
            s2 = np.einsum('ijxy,ijxy,xy->ij', cand, cand, self.window)
            num = np.einsum('ijxy,xy->ij', cand, self.weighted[k])
            var = np.maximum(s2 - s1 * s1 / wsum, 1e-12)
            out[k] = num / (np.sqrt(var) * max(self.norm[k], 1e-15))
        return out

    def _solve(self, offset, H, ok, global_shift):
        """Coarse-grid weighted least squares, then bilinear upsampling."""
        Hd = np.einsum('nij,nj->ni', H, offset)
        A = np.einsum('ky,kx,kij->yxij', self.wy, self.wx, H)
        b = np.einsum('ky,kx,ki->yxi', self.wy, self.wx, Hd)
        trace = (H[:, 0, 0] + H[:, 1, 1])[ok]
        lam = PRIOR * (np.median(trace) if len(trace) else 1.)
        A[..., 0, 0] += lam
        A[..., 1, 1] += lam
        det = A[..., 0, 0] * A[..., 1, 1] - A[..., 0, 1] * A[..., 1, 0]
        ux = (A[..., 1, 1] * b[..., 0] - A[..., 0, 1] * b[..., 1]) / det
        uy = (A[..., 0, 0] * b[..., 1] - A[..., 1, 0] * b[..., 0]) / det
        h, w = self.shape
        yy, xx = np.arange(h, dtype=float), np.arange(w, dtype=float)
        def upsample(u):
            rows = np.stack([np.interp(xx, self.gx, line) for line in u])
            return np.stack([np.interp(yy, self.gy, column) for column in rows.T], axis=1)
        return upsample(ux) + global_shift[0], upsample(uy) + global_shift[1]

    def displacement(self, frame, global_shift):
        frame = np.asarray(frame, dtype=np.float64)
        if frame.shape != self.shape or not np.isfinite(frame).all():
            raise ValueError('finite frame matching the reference required')
        if np.shape(global_shift) != (2,) or not np.isfinite(global_shift).all():
            raise ValueError('finite global displacement required')
        if self.use_cuda:
            return tuple(self.displacement_tensor(frame, global_shift).cpu().numpy())
        if not len(self.centres):
            return tuple(np.full(self.shape, float(v)) for v in global_shift)
        self._check_cancel()
        from planetrecon.pipeline.local_align import pull
        image = pull(LocalRegistration.proxy(frame), tuple(float(v) for v in global_shift))
        offset, H, ok, _ = _peak_fit(self._ncc(image))
        self.accepted.append(float(ok.mean()))
        return self._solve(offset, H, ok, np.asarray(global_shift, dtype=float))

    def displacement_tensor(self, frame, global_shift):
        """CUDA field kept on the device for direct raw backprojection."""
        from planetrecon.backends.torch_ap import TorchAlignmentPoints
        self._check_cancel()
        if self._cuda is None:
            self._cuda = TorchAlignmentPoints(self)
            self.provenance['cuda_execution'].update(used=True, precision='float64',
                matching='grouped GPU correlations and field solve; CPU peak fits')
        return self._cuda.displacement(frame, global_shift)

    def release_cuda(self):
        self._cuda = None

    def acceptance(self):
        return float(np.mean(self.accepted)) if self.accepted else None


def refined_reference(reference, indices, read_plane, correlate, max_shift, *, size, use_cuda=False,
                      should_cancel=None, progress=None):
    """Average many frames twice: globally aligned, then locally aligned to that mean.

    A low-noise reference limits imprinting its noise on the matches. With CUDA,
    ``read_plane`` may return device tensors; the result is a NumPy array.
    """
    from planetrecon.pipeline.local_align import build_template, pull
    total_steps = 2 * len(indices)

    def report(step):
        if progress is not None and (step % 25 == 0 or step == total_steps):
            progress('Building alignment reference', step, total_steps)
    if use_cuda:
        from planetrecon.backends.torch_ap import device_template
        template = device_template(reference, indices, read_plane, correlate, max_shift,
                                   should_cancel=should_cancel, progress=report)
    else:
        seen = [0]

        def counted(index):
            seen[0] += 1
            report(seen[0])
            return read_plane(index)
        template = build_template(reference, indices, counted, correlate, max_shift, should_cancel)
    matcher = AlignmentPoints(template, size=size, use_cuda=use_cuda, should_cancel=should_cancel)
    if not len(matcher.centres):
        return template
    if use_cuda:
        import torch
        from planetrecon.backends.torch_circular import sample
        h, w = template.shape
        yy, xx = torch.meshgrid(torch.arange(h, device='cuda:0', dtype=torch.float64),
                                torch.arange(w, device='cuda:0', dtype=torch.float64), indexing='ij')
        total = torch.zeros((h, w), device='cuda:0', dtype=torch.float64)
        weight = torch.zeros_like(total)
        ones = torch.ones_like(total)

        def add(plane, field):
            image = torch.as_tensor(plane, device='cuda:0', dtype=torch.float64)
            total.add_(sample(image, yy + field[1], xx + field[0]))
            weight.add_(sample(ones, yy + field[1], xx + field[0]))
    else:
        total = np.zeros_like(template)
        weight = np.zeros_like(template)
        ones = np.ones_like(template)

        def add(plane, field):
            total.__iadd__(pull(plane, field))
            weight.__iadd__(pull(ones, field))
    for step, index in enumerate(indices, len(indices) + 1):
        if should_cancel is not None and should_cancel():
            raise InterruptedError('cancelled while building the alignment reference')
        report(step)
        plane = read_plane(int(index))
        shift = correlate(template, plane)
        if not np.isfinite(shift).all() or max(abs(v) for v in shift) > max_shift:
            continue
        add(plane, matcher.displacement_tensor(plane, shift) if use_cuda else matcher.displacement(plane, shift))
    if use_cuda:
        total, weight = total.cpu().numpy(), weight.cpu().numpy()
    return np.divide(total, weight, out=template.copy(), where=weight > .5)


class LocalQuality:
    """Squared Laplacian of the smoothed plane averaged around each AP.

    Normalised by the local brightness squared so that slow transparency or
    exposure changes do not dominate the ranking.
    """
    def __init__(self, matcher, *, use_cuda=False):
        h, w = matcher.shape
        self.centres = matcher.centres.astype(float)
        self.sigma = matcher.size / 4
        self.ys = np.arange(h, dtype=float)
        self.xs = np.arange(w, dtype=float)
        self.use_cuda = use_cuda

    def measure(self, plane, shift):
        if self.use_cuda:
            from planetrecon.backends.torch_ap import local_quality
            return local_quality(plane, shift, self.centres, self.sigma)
        smooth = gaussian_filter(np.asarray(plane, dtype=np.float64), QUALITY_SMOOTH_PX, mode='nearest')
        lap = np.zeros_like(smooth)
        lap[1:-1, 1:-1] = (smooth[:-2, 1:-1] + smooth[2:, 1:-1] + smooth[1:-1, :-2]
                           + smooth[1:-1, 2:] - 4 * smooth[1:-1, 1:-1])
        wy = _gaussians(self.ys, self.centres[:, 0] + float(shift[1]), self.sigma)
        wx = _gaussians(self.xs, self.centres[:, 1] + float(shift[0]), self.sigma)
        # Separable weights: one matrix product per map, not an n x h x w tensor.
        energy = ((wy @ (lap * lap)) * wx).sum(axis=1)
        level = ((wy @ smooth) * wx).sum(axis=1)
        norm = wy.sum(1) * wx.sum(1)
        return (energy / norm) / np.maximum((level / norm) ** 2, 1e-30)


class LocalSelection:
    """Per-AP best frames and their blended per-pixel weights."""
    def __init__(self, matcher, chosen, frames, global_mask, shifts=None):
        """chosen: (n_frames, n_ap) bool for ``frames`` capture indices.

        ``shifts`` optionally keeps each frame's global registration for reuse.
        """
        h, w = matcher.shape
        ys, xs = np.arange(h, dtype=float), np.arange(w, dtype=float)
        sigma = .6 * matcher.spacing
        self.wy = _gaussians(ys, matcher.centres[:, 0].astype(float), sigma)
        self.wx = _gaussians(xs, matcher.centres[:, 1].astype(float), sigma)
        total = self.wy.T @ self.wx
        cover = np.minimum(total / .5, 1.)
        self.scale = cover / np.maximum(total, 1e-12)
        self.rest = 1. - cover
        self.row = {int(f): k for k, f in enumerate(frames)}
        self.chosen = chosen
        self.global_mask = global_mask
        self.shifts = {} if shifts is None else dict(zip((int(f) for f in frames), shifts))

    def frames(self):
        """Capture indices with any nonzero weight."""
        used = np.zeros_like(self.global_mask)
        for index, k in self.row.items():
            used[index] = self.chosen[k].any()
        return used | (self.global_mask & (self.rest.max() > 0))

    def weights(self, index):
        k = self.row.get(int(index))
        selected = self.chosen[k].astype(float) if k is not None else np.zeros(self.wy.shape[0])
        local = (self.wy.T * selected) @ self.wx * self.scale
        return local + self.rest * float(self.global_mask[index])

    def weights_tensor(self, index, device='cuda:0'):
        """Device equivalent of ``weights``."""
        import torch
        if getattr(self, '_device', None) is None:
            self._device = [torch.as_tensor(a, device=device, dtype=torch.float64)
                            for a in (self.wy.T, self.wx, self.scale, self.rest)]
        wyT, wx, scale, rest = self._device
        k = self.row.get(int(index))
        selected = torch.as_tensor(self.chosen[k] if k is not None else np.zeros(wyT.shape[1], bool),
                                   device=device, dtype=torch.float64)
        return (wyT * selected) @ wx * scale + rest * float(self.global_mask[index])


def select_locally(matcher, frames, read_plane, correlate, reference, count, global_mask, should_cancel=None,
                   use_cuda=False, progress=None):
    """Measure local quality for ``frames`` and keep each AP's best ``count``."""
    quality = LocalQuality(matcher, use_cuda=use_cuda)
    scores = np.zeros((len(frames), len(matcher.centres)))
    shifts = []
    for k, index in enumerate(frames):
        if should_cancel is not None and should_cancel():
            raise InterruptedError('cancelled while measuring local quality')
        if progress is not None and (k % 50 == 0 or k == len(frames) - 1):
            progress('Ranking frames by local quality', k + 1, len(frames))
        plane = read_plane(int(index))
        shifts.append(tuple(correlate(reference, plane)))
        scores[k] = quality.measure(plane, shifts[-1])
    count = max(1, min(int(count), len(frames)))
    order = np.argsort(-scores, axis=0, kind='stable')[:count]
    chosen = np.zeros(scores.shape, dtype=bool)
    np.put_along_axis(chosen, order, True, axis=0)
    return LocalSelection(matcher, chosen, frames, global_mask, shifts)
