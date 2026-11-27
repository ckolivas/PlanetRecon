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

Where a point's reference varies along one direction only (Saturn's belts at
the disc centre), its correlation peak is a ridge and the position along the
ridge is chosen by noise. Following those offsets would shift each frame
until its noise best matches the reference's, which imprints the reference's
fine noise on the stack as streaks at the point. A calibration pass measures
how each point's offsets spread along its principal directions: seeing moves
points about as far as the best-measured directions, while noise-chosen
offsets spread evenly across the search window. Each direction keeps the
share of its information that behaves like seeing.

Local quality ranks frames separately for each AP; per-pixel frame weights
blend the AP selections with normalised Gaussian footprints. Outside the AP
cloud the global frame selection applies.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from planetrecon.pipeline.local_align import proxy as registration_proxy

SEARCH = 6
MIN_NCC = 0.5
PRIOR = 0.01
FIELD_STEP = 4
REFERENCE_FRAMES = 3000  # best selected frames averaged into the reference
MIN_BRIGHTNESS = 0.1
QUALITY_SMOOTH_PX = 1.5
CALIBRATION_FRAMES = 400  # frames whose peak fits measure each point's trustworthy directions
MIN_CALIBRATION = 30  # accepted fits a point needs before its directions are judged


def _grid(length, step):
    """Coarse field sample coordinates spanning the detector exactly."""
    count = max(2, int(np.ceil((length - 1) / step)) + 1)
    return np.linspace(0., length - 1., count)


def noise_spread(search=SEARCH):
    """Spread of offsets chosen by noise: even over the interior of the search window."""
    return (2 * search - 1) / np.sqrt(12)


def direction_trust(offsets, information, ok, search=SEARCH):
    """(n, 2, 2) trust matrices from peak fits (frames, n, ...) of n points, and a summary.

    Each point's offsets are projected on the principal directions of its mean
    information. A direction whose offsets spread as widely as the seeing (the
    median spread of the points' best-measured directions) is fully trusted;
    one spreading as evenly as noise over the search window is not. Between
    the two, the trusted share is the fraction of offsets that follow the
    seeing in a mixture of the two spreads. Points with too few accepted fits
    keep all their information. Information H becomes T H T.
    """
    offsets, information, ok = np.asarray(offsets), np.asarray(information), np.asarray(ok, bool)
    n = information.shape[1]
    vectors = np.tile(np.eye(2), (n, 1, 1))
    spreads = np.full((n, 2), np.nan)
    for k in range(n):
        m = ok[:, k]
        if m.sum() < MIN_CALIBRATION:
            continue
        _, vectors[k] = np.linalg.eigh(information[m, k].mean(0))  # Weakest direction first.
        spreads[k] = (offsets[m, k] @ vectors[k]).std(0)
    noise = noise_spread(search)
    measured = np.isfinite(spreads[:, 1])
    seeing = float(np.median(spreads[measured, 1])) if measured.any() else np.nan
    keep = np.ones((n, 2))
    if np.isfinite(seeing) and seeing < noise:
        keep = np.where(np.isfinite(spreads),
                        np.clip((noise ** 2 - spreads ** 2) / (noise ** 2 - seeing ** 2), 0., 1.), 1.)
    trust = np.einsum('kij,kj,klj->kil', vectors, np.sqrt(keep), vectors)
    summary = {'calibration_frames': int(len(ok)), 'points_measured': int(measured.sum()),
               'seeing_spread_px': None if not np.isfinite(seeing) else seeing,
               'noise_spread_px': float(noise),
               'directions_mostly_noise': int(np.sum(keep < .5)),
               'directions_partly_noise': int(np.sum((keep >= .5) & (keep < .9)))}
    return trust, summary


def trusted(information, trust):
    """Information (m, 2, 2) restricted to trusted directions; ``trust`` (m, 2, 2) or None."""
    if trust is None:
        return information
    return np.einsum('mij,mjk,mlk->mil', trust, information, trust)


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
    # An exact match has no subpixel offset (as in the other matchers).
    delta[peak >= 1 - 1e-10] = 0.
    ok &= (np.abs(delta) <= 1).all(axis=1)
    offset = np.stack([cx - R, cy - R], -1) + delta
    H = np.einsum('nij,nj,nkj->nik', evecs, evals * usable, evecs) * ok[:, None, None]
    return np.where(ok[:, None], offset, 0.), H, ok, peak


class AlignmentPoints:
    """Local registration with an AutoStakkert-style AP grid.

    ``displacement`` returns a dense (x, y) sampling field in detector pixels,
    the convention of ``local_align.pull`` and the CUDA backprojection.
    """
    def __init__(self, reference, *, size=64, search=SEARCH, use_cuda=False, should_cancel=None, valid_mask=None,
                 trust=None):
        """``trust`` optionally maps point centres (row, column) to calibrated trust matrices (``trust_map``)."""
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
        proxy = registration_proxy(reference)
        window = np.outer(np.hanning(size), np.hanning(size))
        level = reference.max() * MIN_BRIGHTNESS
        if valid_mask is not None:
            valid_mask = np.asarray(valid_mask, dtype=bool)
            if valid_mask.shape != reference.shape:
                raise ValueError('alignment point support must match the reference shape')
        centres, templates, energy = [], [], []
        for y in range(margin, h - margin - size % 2, self.spacing):
            for x in range(margin, w - margin - size % 2, self.spacing):
                section = np.s_[y - half:y - half + size, x - half:x - half + size]
                if reference[section].mean() < level:
                    continue
                if valid_mask is not None and not valid_mask[y-margin:y-margin+size+2*search,
                                                             x-margin:x-margin+size+2*search].all():
                    continue  # The whole search footprint must be observed (e.g. a predicted reference).
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
        self.trust = None
        if trust:
            self.trust = np.stack([trust.get(tuple(c), np.eye(2)) for c in self.centres.tolist()]) \
                if len(self.centres) else None
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
        """[self] when any point is usable, else empty."""
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

    def peak_fits(self, frame, global_shift):
        """(offset, information, accepted) of every point for one frame, before any trust."""
        if self.use_cuda:
            return self._device().peak_fits(frame, global_shift)
        from planetrecon.pipeline.local_align import pull
        image = pull(registration_proxy(np.asarray(frame, dtype=np.float64)), tuple(float(v) for v in global_shift))
        offset, H, ok, _ = _peak_fit(self._ncc(image))
        return offset, H, ok

    def calibrate(self, samples):
        """Measure each point's trusted directions from (plane, global shift) samples."""
        fits = []
        for plane, shift in samples:
            self._check_cancel()
            fits.append(self.peak_fits(plane, shift))
        self._calibrate(fits)

    def calibrate_groups(self, groups):
        """``calibrate`` from (indices, device planes, global shifts) groups, matched on CUDA together."""
        fits = []
        for _, planes, shifts in groups:
            self._check_cancel()
            if len(self.centres):
                fits.extend(zip(*self._device().peak_fits_batch(planes, shifts)))
        self._calibrate(fits)

    def _calibrate(self, fits):
        if not len(self.centres) or not fits:
            return
        offsets, information, ok = (np.stack(a) for a in zip(*fits))
        self.trust, summary = direction_trust(offsets, information, ok, self.search)
        self.provenance['direction_trust'] = {
            **summary, 'method': 'offset spread along each point\'s principal directions: seeing-like '
                                 'spread kept, even spread over the search window (noise) dropped'}

    def trust_map(self):
        """Calibrated trust by point centre, for matchers of predicted references on the same grid."""
        if self.trust is None:
            return None
        return {tuple(c): t for c, t in zip(self.centres.tolist(), self.trust)}

    def _solve(self, offset, H, ok, global_shift):
        """Coarse-grid weighted least squares, then bilinear upsampling."""
        H = trusted(H, self.trust)
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
        offset, H, ok = self.peak_fits(frame, global_shift)
        self.accepted.append(float(ok.mean()))
        return self._solve(offset, H, ok, np.asarray(global_shift, dtype=float))

    def displacement_tensor(self, frame, global_shift):
        """CUDA field kept on the device for direct raw backprojection."""
        self._check_cancel()
        return self._device().displacement(frame, global_shift)

    def displacements_tensor(self, frames, global_shifts):
        """CUDA fields (B, 2, h, w) of frames (B, h, w), kept on the device."""
        self._check_cancel()
        return self._device().displacements(frames, global_shifts)

    def _device(self):
        if self._cuda is None:
            from planetrecon.backends.torch_ap import TorchAlignmentPoints
            from planetrecon.backends import precision
            self._cuda = TorchAlignmentPoints(self)
            self.provenance['cuda_execution'].update(
                used=True, precision='float64',
                registration_precision=str(precision.REGISTRATION).removeprefix('torch.'),
                blend_precision=str(precision.BLEND).removeprefix('torch.'),
                matching='grouped GPU correlations (registration precision) and float64 field solve; CPU peak fits')
        return self._cuda

    def release_cuda(self):
        self._cuda = None

    def acceptance(self):
        return float(np.mean(self.accepted)) if self.accepted else None


def calibration_indices(indices):
    """Up to CALIBRATION_FRAMES of ``indices``, spread evenly."""
    indices = np.asarray(indices)
    if len(indices) > CALIBRATION_FRAMES:
        indices = indices[np.linspace(0, len(indices) - 1, CALIBRATION_FRAMES).round().astype(int)]
    return indices


def _within(shift, max_shift):
    return bool(np.isfinite(shift).all()) and max(abs(v) for v in shift) <= max_shift


def calibration_samples(indices, read_plane, correlate, reference, max_shift, *, plan=None):
    """(plane, global shift) of up to CALIBRATION_FRAMES of ``indices``, spread evenly.

    ``plan`` (optional) is told the reading order before the reads, e.g. to read ahead.
    """
    indices = calibration_indices(indices)
    if plan is not None:
        plan(indices)
    for index in indices:
        plane = read_plane(int(index))
        shift = correlate(reference, plane)
        if _within(shift, max_shift):
            yield plane, shift


class FrameGroups:
    """Frames read, registered and globally aligned on CUDA in groups.

    ``planes(indices)`` returns their registration planes (B, h, w) float64 on
    the device; ``backend`` correlates them against a reference together.
    ``plan`` (optional) is told each reading order first, e.g. to read ahead.
    """

    def __init__(self, planes, backend, size, *, plan=None, should_cancel=None):
        self.planes, self.backend, self.size = planes, backend, max(1, int(size))
        self.plan, self.should_cancel = plan, should_cancel

    def aligned(self, reference, indices, max_shift, *, progress=None, message='cancelled'):
        """(indices, planes, global shifts) of each group, keeping frames whose shift is finite and within
        ``max_shift``; ``progress(n)`` after each group of the n frames read so far."""
        import torch
        indices = np.asarray(indices, dtype=np.int64)
        if self.plan is not None:
            self.plan(indices)
        for start in range(0, len(indices), self.size):
            if self.should_cancel is not None and self.should_cancel():
                raise InterruptedError(message)
            chunk = indices[start:start + self.size]
            planes = self.planes(chunk)
            shifts = self.backend.phase_correlations(reference, planes)
            if progress is not None:
                progress(start + len(chunk))
            keep = [k for k, shift in enumerate(shifts) if _within(shift, max_shift)]
            if len(keep) < len(chunk):
                planes = planes[torch.as_tensor(keep, device=planes.device, dtype=torch.long)]
            if keep:
                yield chunk[keep], planes, [shifts[k] for k in keep]


def refined_reference(reference, indices, read_plane, correlate, max_shift, *, size, use_cuda=False,
                      should_cancel=None, progress=None, plan=None, groups=None):
    """Average many frames twice: globally aligned, then locally aligned to that mean.

    A low-noise reference limits imprinting its noise on the matches. With CUDA,
    ``read_plane`` may return device tensors; the result is a NumPy array.
    The averages read the frames in capture order; ``plan`` (optional) is told
    each reading order first, e.g. to read ahead. ``groups`` (a ``FrameGroups``,
    CUDA only) reads, registers and aligns the frames in groups instead.
    """
    from planetrecon.pipeline.local_align import build_template, pull
    total_steps = 2 * len(indices)
    ordered = np.sort(np.asarray(indices))
    plan = plan or (lambda order: None)

    def report(step):
        if progress is not None and (step % 25 == 0 or step == total_steps):
            progress('Building alignment reference', step, total_steps)
    if groups is not None:
        return _grouped_reference(reference, indices, ordered, groups, correlate, max_shift, size=size,
                                  should_cancel=should_cancel, progress=progress)
    plan(ordered)
    if use_cuda:
        from planetrecon.backends.torch_ap import device_template
        template = device_template(reference, ordered, read_plane, correlate, max_shift,
                                   should_cancel=should_cancel, progress=report)
    else:
        seen = [0]

        def counted(index):
            seen[0] += 1
            report(seen[0])
            return read_plane(index)
        template = build_template(reference, ordered, counted, correlate, max_shift, should_cancel)
    matcher = AlignmentPoints(template, size=size, use_cuda=use_cuda, should_cancel=should_cancel)
    if not len(matcher.centres):
        return template
    matcher.calibrate(calibration_samples(indices, read_plane, correlate, template, max_shift, plan=plan))
    if use_cuda:
        import torch
        from planetrecon.backends.torch_ap import sample
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
    plan(ordered)
    for step, index in enumerate(ordered, len(indices) + 1):
        if should_cancel is not None and should_cancel():
            raise InterruptedError('cancelled while building the alignment reference')
        report(step)
        plane = read_plane(int(index))
        shift = correlate(template, plane)
        if not _within(shift, max_shift):
            continue
        add(plane, matcher.displacement_tensor(plane, shift) if use_cuda else matcher.displacement(plane, shift))
    if use_cuda:
        total, weight = total.cpu().numpy(), weight.cpu().numpy()
    return np.divide(total, weight, out=template.copy(), where=weight > .5)


def _grouped_reference(reference, indices, ordered, groups, correlate, max_shift, *, size, should_cancel, progress):
    """``refined_reference`` with each pass's frames read, registered, aligned and pulled in CUDA groups."""
    from planetrecon.backends.torch_ap import device_template_groups, pulled_sums
    total_steps = 2 * len(indices)
    message = 'cancelled while building the alignment reference'

    def report(offset):
        def at(n):
            if progress is not None:
                progress('Building alignment reference', offset + n, total_steps)
        return at
    template = device_template_groups(reference, groups.aligned(reference, ordered, max_shift, progress=report(0),
                                                                message=message), correlate, max_shift)
    matcher = AlignmentPoints(template, size=size, use_cuda=True, should_cancel=should_cancel)
    if not len(matcher.centres):
        return template
    # Every later correlation is against the template: prepare it once.
    template = groups.backend.prepare_reference(template)
    matcher.calibrate_groups(groups.aligned(template, calibration_indices(indices), max_shift, message=message))
    total, weight = pulled_sums(
        (planes, matcher.displacements_tensor(planes, shifts))
        for _, planes, shifts in groups.aligned(template, ordered, max_shift, progress=report(len(indices)),
                                                message=message))
    if total is None:
        return np.array(template)
    total, weight = total.cpu().numpy(), weight.cpu().numpy()
    return np.divide(total, weight, out=np.array(template), where=weight > .5)


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
        return self.measure_at(plane, self.centres + np.array([float(shift[1]), float(shift[0])]))

    def measure_at(self, plane, centres):
        """Local quality around explicit (row, column) centres, e.g. after a motion model."""
        centres = np.asarray(centres, dtype=float)
        if self.use_cuda:
            from planetrecon.backends.torch_ap import local_quality
            return local_quality(plane, (0., 0.), centres, self.sigma)
        smooth = gaussian_filter(np.asarray(plane, dtype=np.float64), QUALITY_SMOOTH_PX, mode='nearest')
        lap = np.zeros_like(smooth)
        lap[1:-1, 1:-1] = (smooth[:-2, 1:-1] + smooth[2:, 1:-1] + smooth[1:-1, :-2]
                           + smooth[1:-1, 2:] - 4 * smooth[1:-1, 1:-1])
        wy = _gaussians(self.ys, centres[:, 0], self.sigma)
        wx = _gaussians(self.xs, centres[:, 1], self.sigma)
        # Separable weights: one matrix product per map, not an n x h x w tensor.
        energy = ((wy @ (lap * lap)) * wx).sum(axis=1)
        level = ((wy @ smooth) * wx).sum(axis=1)
        norm = wy.sum(1) * wx.sum(1)
        return (energy / norm) / np.maximum((level / norm) ** 2, 1e-30)


class LocalSelection:
    """Per-AP best frames and their blended per-pixel weights."""
    def __init__(self, matcher, chosen, frames, global_mask, shifts=None, bins=None, global_bins=None):
        """chosen: (n_frames, n_ap) bool for ``frames`` capture indices.

        ``shifts`` optionally keeps each frame's global registration for reuse.
        ``bins`` (n_frames, n_ap) and ``global_bins`` (capture length) give
        each chosen frame's local-quality rank bin (-1 when not chosen), for
        Fourier weighting; the global bin applies outside the points.
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
        self.bins, self.global_bins = bins, global_bins

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

    def _device_maps(self, selections, device):
        """Blended footprints (k, h, w) float64 of (k, n_ap) point selections, on the device.

        The footprint products are summed in ``precision.BLEND`` (one batched
        product for all k); a relative error of about 1e-7 in a frame's weight.
        """
        import torch
        from planetrecon.backends import precision
        if getattr(self, '_device', None) is None or self._device[0].device != torch.device(device):
            self._device = [torch.as_tensor(a, device=device, dtype=torch.float64)
                            for a in (self.wy.T, self.wx, self.scale, self.rest)]
        wyT, wx, scale, _ = self._device
        dtype = precision.BLEND
        selected = torch.as_tensor(np.asarray(selections, dtype=np.float64), device=device, dtype=dtype)
        maps = (wyT.to(dtype)[None] * selected[:, None, :]) @ wx.to(dtype)
        return maps.to(torch.float64) * scale

    def weights_tensor(self, index, device='cuda:0'):
        """Device equivalent of ``weights``."""
        k = self.row.get(int(index))
        selected = self.chosen[k] if k is not None else np.zeros(self.wy.shape[0], bool)
        return self._device_maps(selected[None], device)[0] + self._device[3] * float(self.global_mask[index])

    def bin_weights(self, index):
        """[(bin, per-pixel weight)] for one frame; the weights sum to ``weights(index)``."""
        k = self.row.get(int(index))
        local = self.bins[k] if k is not None else np.full(self.wy.shape[0], -1)
        outside = int(self.global_bins[index]) if self.global_mask[index] else -1
        result = []
        for b in sorted(set(local[local >= 0].tolist()) | ({outside} if outside >= 0 else set())):
            weight = (self.wy.T * (local == b)) @ self.wx * self.scale
            if b == outside:
                weight = weight + self.rest
            result.append((b, weight))
        return result

    def pairs(self, index, main):
        """Weight terms of one frame for ``torch_ap.DeviceSums``: [(slot, points, outside)], the
        ordinary stack in slot ``main`` and, with rank bins, each bin's half 2 b + h. A term weighs
        the blended footprints of its points plus, when ``outside``, the weight outside the points."""
        from planetrecon.pipeline import fourier_weights
        k = self.row.get(int(index))
        chosen = np.flatnonzero(self.chosen[k]) if k is not None else np.zeros(0, np.int64)
        terms = [(main, chosen, float(self.global_mask[index]))]
        if self.bins is None:
            return terms
        local = self.bins[k] if k is not None else np.full(self.wy.shape[0], -1)
        outside = int(self.global_bins[index]) if self.global_mask[index] else -1
        for b in sorted(set(local[local >= 0].tolist()) | ({outside} if outside >= 0 else set())):
            terms.append((2 * b + fourier_weights.half(index), np.flatnonzero(local == b), float(b == outside)))
        return terms

    def blend_tensors(self, device='cuda:0'):
        """(wy (h, K), wx (K, w)) in blend precision and (scale, rest) (h*w,) float64, on the device."""
        import torch
        from planetrecon.backends import precision
        as_device = lambda a, dtype: torch.as_tensor(np.ascontiguousarray(a), device=device, dtype=dtype)
        return (as_device(self.wy.T, precision.BLEND), as_device(self.wx, precision.BLEND),
                as_device(self.scale.ravel(), torch.float64), as_device(self.rest.ravel(), torch.float64))

    def bin_weights_tensor(self, index, device='cuda:0'):
        """Device equivalent of ``bin_weights``."""
        k = self.row.get(int(index))
        local = self.bins[k] if k is not None else np.full(self.wy.shape[0], -1)
        outside = int(self.global_bins[index]) if self.global_mask[index] else -1
        order = sorted(set(local[local >= 0].tolist()) | ({outside} if outside >= 0 else set()))
        if not order:
            return []
        maps = self._device_maps(np.stack([local == b for b in order]), device)
        rest = self._device[3]
        return [(b, maps[j] + rest if b == outside else maps[j]) for j, b in enumerate(order)]


def rank_bins(count, n_bins):
    """Bin of each rank position: equal-count bins of the best ``count`` ranks."""
    return np.minimum(np.arange(count) * n_bins // max(count, 1), n_bins - 1)


def assign_bins(order, chosen, n_bins, global_mask, global_scores):
    """Rank bins per point and outside the points, for Fourier weighting.

    ``order`` (count, n_ap) lists each point's best frames first; ``chosen``
    marks the frames each point stacks. Returns (bins, global_bins): each
    chosen frame's bin at every point (-1 otherwise), and each globally
    selected capture frame's bin by ``global_scores``.
    """
    bins = np.full(chosen.shape, -1, dtype=np.int64)
    np.put_along_axis(bins, order, rank_bins(len(order), n_bins)[:, None], axis=0)
    bins[~chosen] = -1
    selected = np.flatnonzero(global_mask)
    ranked = selected[np.argsort(-global_scores[selected], kind='stable')]
    global_bins = np.full(len(global_mask), -1, dtype=np.int64)
    global_bins[ranked] = rank_bins(len(ranked), n_bins)
    return bins, global_bins


def select_locally(matcher, frames, read_plane, correlate, reference, count, global_mask, should_cancel=None,
                   use_cuda=False, progress=None, n_bins=0, global_scores=None, plan=None, measure_group=None,
                   group=1):
    """Measure local quality for ``frames`` and keep each AP's best ``count``.

    With ``n_bins``, each point's chosen frames are also split into equal-count
    local-quality rank bins, best first; ``global_scores`` (capture length)
    ranks the frames of ``global_mask`` the same way for use outside the points.
    ``measure_group(indices)``, when given, returns the global shifts and point
    scores of ``group`` frames at a time (e.g. together on the device).
    """
    quality = LocalQuality(matcher, use_cuda=use_cuda)
    if plan is not None:
        plan(frames)
    scores = np.zeros((len(frames), len(matcher.centres)))
    shifts = []
    step = max(1, int(group)) if measure_group is not None else 1
    for start in range(0, len(frames), step):
        if should_cancel is not None and should_cancel():
            raise InterruptedError('cancelled while measuring local quality')
        chunk = frames[start:start + step]
        if progress is not None and (start // 50 != (start + len(chunk)) // 50 or start + len(chunk) == len(frames)
                                     or start == 0):
            progress('Ranking frames by local quality', start + len(chunk), len(frames))
        if measure_group is not None:
            found, measured = measure_group([int(index) for index in chunk])
            shifts.extend(tuple(shift) for shift in found)
            scores[start:start + len(chunk)] = measured
            continue
        plane = read_plane(int(chunk[0]))
        shifts.append(tuple(correlate(reference, plane)))
        scores[start] = quality.measure(plane, shifts[-1])
    count = max(1, min(int(count), len(frames)))
    order = np.argsort(-scores, axis=0, kind='stable')[:count]
    chosen = np.zeros(scores.shape, dtype=bool)
    np.put_along_axis(chosen, order, True, axis=0)
    bins = global_bins = None
    if n_bins:
        bins, global_bins = assign_bins(order, chosen, n_bins, global_mask, global_scores)
    return LocalSelection(matcher, chosen, frames, global_mask, shifts, bins, global_bins)
