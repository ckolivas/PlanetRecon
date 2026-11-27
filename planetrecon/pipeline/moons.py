"""Moons beside the planet: found, tracked and stacked on their own positions.

Moons lie tens of arcseconds from the planet, beyond the patch of sky whose
seeing tilt the planet's registration removes. Each frame tilts them by its
own amount (about 0.3" rms per axis on a 16" telescope), and they drift along
their orbits by a pixel or more in a few minutes. Registered with the planet
they blur to about twice their size.

A calibration pass over up to MOON_FRAMES frames, spread through the
selection, stacks a detection image with the planet's global registration
and finds compact sources whose windows stay clear of the planet's glow and
of the alignment points. In the same frames each moon's position is a
Gaussian-weighted centroid. A straight line (a quadratic for long captures)
through its positions relative to the planet gives its orbital drift; the
scatter about it is seeing tilt plus centroid noise. The noise is measured
by placing copies of the brightest moon's profile, at each moon's
brightness, in clear sky and finding them again. Sources fixed on the
detector while the planet moves (hot pixels) are dropped.

While stacking, each frame's moon position is the drift model, plus the
tilt its well-measured neighbours predict for it (their noise is not its
noise), plus the share k = s^2 / (s^2 + n^2) of the rest of its own
deviation, s^2 being the seeing variance the neighbours do not predict and
n^2 the noise variance. A bright moon follows its measured tilt (k near 1);
a moon whose centroid cannot beat its noise follows its drift and its
neighbours, so noise cannot choose positions and sharpen itself into a
false point. A small window round each moon is accumulated with the
frame's weight, then replaces the planet-registered moon in the result at
its position at the output epoch, feathered at the window edge;
neighbouring moons share their windows along the line halfway between them.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from functools import lru_cache

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter, maximum_filter, median_filter

VERSION = 1
METHOD = ('compact sources clear of the glow found on a planet-registered stack of calibration frames; '
          'Gaussian-weighted centroids each frame; polynomial drift relative to the planet, plus the tilt '
          'predicted from well-measured neighbours, plus the share seeing^2/(seeing^2+noise^2) of the rest '
          'of each frame\'s own deviation, noise measured on copies of the brightest moon injected into clear '
          'sky; windows stacked with the frame weights and feathered in at the output epoch')
MOON_FRAMES = 400  # frames that find the moons and calibrate their tracks
MIN_MEASURED = 30  # measured positions a moon needs before it is stacked
DETECTION_SNR = 10.  # detection-image peak over the sky noise
GLOW_FRACTION = .02  # smooth background over the sky, as a fraction of the planet's brightness
MAX_MOONS = 16
QUADRATIC_S = 600.  # captures at least this long fit curved tracks
OUTLIER_SIGMA = 4.  # a deviation beyond this many standard deviations is not a moon position
NEIGHBOUR_SOFTNESS = 1.5  # px; width of the hand-over between neighbouring moons


@dataclass
class Moon:
    """One moon: drift model relative to the planet and its stacking state."""
    y: float  # Output position (reference pixels) at the output epoch.
    x: float
    sigma: float  # Centroid weighting Gaussian, px.
    half: int  # Centroid window half-size, px.
    coef_y: list  # Polynomial in (t - t_out), highest power first, for the position relative to the planet.
    coef_x: list
    gain: float  # Share of a frame's own deviation from the neighbours' prediction that is followed.
    detail: dict = field(default_factory=dict)
    # Tilt shared with well-measured moons: {index of that moon: [coefficient y, coefficient x]}
    # predicting this moon's deviation from its drift from theirs.
    coupling: dict = field(default_factory=dict)

    def relative(self, t):
        return np.polyval(self.coef_y, t), np.polyval(self.coef_x, t)

    def as_dict(self):
        return {'y': self.y, 'x': self.x, 'sigma': self.sigma, 'half': self.half,
                'coef_y': list(self.coef_y), 'coef_x': list(self.coef_x), 'gain': self.gain,
                'detail': self.detail, 'coupling': {str(k): list(v) for k, v in self.coupling.items()}}

    @classmethod
    def from_dict(cls, data):
        return cls(float(data['y']), float(data['x']), float(data['sigma']), int(data['half']),
                   [float(v) for v in data['coef_y']], [float(v) for v in data['coef_x']],
                   float(data['gain']), dict(data.get('detail', {})),
                   {int(k): [float(v[0]), float(v[1])] for k, v in data.get('coupling', {}).items()})


def frame_times(source):
    """Seconds from the first frame, and whether they are real times (else frame numbers)."""
    ticks = source.timestamps()
    n = source.n_frames()
    if ticks is not None and len(ticks) == n and n > 1:
        t = (np.asarray(ticks, dtype=np.float64) - float(ticks[0])) * source.timestamp_scale_s()
        if np.isfinite(t).all() and t[-1] > 0 and np.all(np.diff(t) >= 0):
            return t, True
    return np.arange(n, dtype=np.float64), False


def spread(indices, count):
    indices = np.asarray(indices)
    if len(indices) > count:
        indices = indices[np.linspace(0, len(indices) - 1, count).round().astype(int)]
    return indices


@lru_cache(maxsize=32)
def _offsets(half):
    offsets = np.mgrid[-half:half + 1, -half:half + 1].astype(np.float64)
    offsets.flags.writeable = False
    return offsets


@lru_cache(maxsize=32)
def _border_fit(half):
    """(border mask, least-squares operator) of the plane fitted to a window's border."""
    yy, xx = _offsets(half)
    border = np.maximum(np.abs(yy), np.abs(xx)) > half - 3
    design = np.stack([np.ones(border.sum()), yy[border], xx[border]], 1)
    operator = np.linalg.pinv(design)
    border.flags.writeable = operator.flags.writeable = False
    return border, operator


def centroid(plane, cy, cx, sigma, half, parity=None, *, origin=(0, 0), shape=None):
    """Gaussian-weighted centroid (y, x) near (cy, cx), or None.

    A plane fitted to the window's border removes sky and glow gradients.
    ``parity`` 0 or 1 uses only pixels whose (row + column) has that parity.
    ``plane`` may be a crop of a frame of ``shape`` whose pixel (0, 0) is the
    frame's ``origin`` (row, column); coordinates stay those of the frame.
    """
    h, w = plane.shape if shape is None else shape
    iy, ix = int(round(cy)), int(round(cx))
    if iy - half < 0 or ix - half < 0 or iy + half >= h or ix + half >= w:
        return None
    oy, ox = origin
    win = np.asarray(plane[iy - half - oy:iy + half + 1 - oy, ix - half - ox:ix + half + 1 - ox], dtype=np.float64)
    yy, xx = _offsets(half)
    border, operator = _border_fit(half)
    coef = operator @ win[border]
    win = win - (coef[0] + coef[1] * yy + coef[2] * xx)
    use = np.ones_like(win) if parity is None else ((yy + xx) % 2 == parity).astype(np.float64)
    y0, x0 = cy - iy, cx - ix
    for _ in range(8):
        g = np.exp(-((yy - y0) ** 2 + (xx - x0) ** 2) / (2 * sigma * sigma)) * use
        f = (win * g).sum()
        if not f > 0:
            return None
        y1, x1 = (win * g * yy).sum() / f, (win * g * xx).sum() / f
        if abs(y1) > half - 2 or abs(x1) > half - 2:
            return None
        done = abs(y1 - y0) < 1e-3 and abs(x1 - x0) < 1e-3
        y0, x0 = y1, x1
        if done:
            break
    return iy + y0, ix + x0


def _crops(plane, centres, halves):
    """[(window, origin) or None]: each centroid window of ``plane`` around (row, column) ``centres``.

    A device tensor plane is copied to the host once for all windows, so the
    centroids need only the frame's registration plane where it already is.
    """
    h, w = plane.shape[:2]
    boxes = []
    for (cy, cx), half in zip(centres, halves):
        iy, ix = int(round(cy)), int(round(cx))
        inside = iy - half >= 0 and ix - half >= 0 and iy + half < h and ix + half < w
        boxes.append((iy - half, ix - half, 2 * half + 1) if inside else None)
    if not hasattr(plane, 'cpu'):
        return [None if b is None else (plane, (0, 0)) for b in boxes]
    import torch
    used = [b for b in boxes if b is not None]
    if not used:
        return [None] * len(boxes)
    flat = torch.cat([plane[y:y + n, x:x + n].reshape(-1) for y, x, n in used]).double().cpu().numpy()
    out, start = [], 0
    for b in boxes:
        if b is None:
            out.append(None)
            continue
        y, x, n = b
        out.append((flat[start:start + n * n].reshape(n, n), (y, x)))
        start += n * n
    return out


def _shifted(plane, dy, dx):
    """``plane`` moved by (dy, dx) with bilinear weights and zero outside: SciPy's order-one
    ``shift`` (``prefilter=False``, ``mode='grid-constant'``), and its support, from four slices."""
    h, w = plane.shape
    out, support = np.zeros_like(plane, dtype=np.float64), np.zeros((h, w))
    iy, ix = int(np.floor(-dy)), int(np.floor(-dx))
    fy, fx = -dy - iy, -dx - ix
    for oy, wy in ((iy, 1 - fy), (iy + 1, fy)):
        for ox, wx in ((ix, 1 - fx), (ix + 1, fx)):
            y0, y1, x0, x1 = max(0, -oy), min(h, h - oy), max(0, -ox), min(w, w - ox)
            if wy * wx == 0 or y1 <= y0 or x1 <= x0:
                continue
            out[y0:y1, x0:x1] += (wy * wx) * plane[y0 + oy:y1 + oy, x0 + ox:x1 + ox]
            support[y0:y1, x0:x1] += wy * wx
    return out, support


def pull(plane, sy, sx, radius, cy, cx):
    """Bilinear window (2r+1 square) whose pixel (i, j) samples ``plane`` at (oy+i+sy, ox+j+sx).

    The window origin is (round(cy) - r, round(cx) - r). None when the samples
    leave the frame. ``plane`` may carry a trailing channel axis.
    """
    n = 2 * radius + 1
    fy, fx = round(cy) - radius + sy, round(cx) - radius + sx
    iy, ix = int(np.floor(fy)), int(np.floor(fx))
    dy, dx = fy - iy, fx - ix
    if iy < 0 or ix < 0 or iy + n >= plane.shape[0] or ix + n >= plane.shape[1]:
        return None
    c = np.asarray(plane[iy:iy + n + 1, ix:ix + n + 1], dtype=np.float64)
    return ((1 - dy) * ((1 - dx) * c[:-1, :-1] + dx * c[:-1, 1:])
            + dy * ((1 - dx) * c[1:, :-1] + dx * c[1:, 1:]))


def _fwhm(image, y, x, half):
    """Equivalent-circle FWHM of a source after removing a border plane."""
    iy, ix = int(round(y)), int(round(x))
    win = np.asarray(image[iy - half:iy + half + 1, ix - half:ix + half + 1], dtype=np.float64)
    yy, xx = _offsets(half)
    border, operator = _border_fit(half)
    coef = operator @ win[border]
    win = win - (coef[0] + coef[1] * yy + coef[2] * xx)
    peak = win[half - 2:half + 3, half - 2:half + 3].max()
    if not peak > 0:
        return None
    return float(2 * np.sqrt(np.count_nonzero(win > peak / 2) / np.pi))


def detect(image, coverage, *, points=(), point_reach=0., edge=0):
    """Compact sources clear of the planet's glow.

    Returns (accepted, excluded, blank): lists of dicts, and the (y, x) of
    the clear sky farthest from any source, the glow and the field edge.
    """
    image = np.asarray(image, dtype=np.float64)
    h, w = image.shape
    small = image[::2, ::2]
    background = np.repeat(np.repeat(median_filter(small, size=21, mode='nearest'), 2, 0), 2, 1)[:h, :w]
    sky = float(np.median(image[coverage]))
    level = float(np.percentile(image[coverage], 99.9)) - sky
    if not level > 0:
        return [], [], None
    glow = background - sky >= GLOW_FRACTION * level
    clear = coverage & ~glow
    residual = gaussian_filter(image - background, 1.5)
    values = residual[clear]
    if values.size < 100:
        return [], [], None
    noise = 1.4826 * float(np.median(np.abs(values - np.median(values))))
    if not noise > 0:
        return [], [], None
    sources = maximum_filter(residual > 3 * noise, size=31)
    room = distance_transform_edt(np.pad(clear & ~sources, 1))[1:-1, 1:-1]
    blank = np.unravel_index(int(np.argmax(room)), room.shape)
    blank = (int(blank[0]), int(blank[1])) if room[blank] > 24 else None
    peaks = (residual == maximum_filter(residual, size=15)) & (residual > DETECTION_SNR * noise) & coverage
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    accepted, excluded = [], []
    for y, x in sorted(zip(*np.nonzero(peaks)), key=lambda p: -residual[p]):
        snr = float(residual[y, x] / noise)
        record = {'y': int(y), 'x': int(x), 'snr': round(snr, 1)}
        if glow[y, x]:
            # The planet, its rings and anything in their glow keep the planet's registration.
            if image[y, x] - sky < .5 * level:
                excluded.append({**record, 'reason': "in the planet's glow"})
            continue
        if len(points) and np.hypot(*(points - (y, x)).T).min() < point_reach:
            excluded.append({**record, 'reason': 'within reach of an alignment point'})
            continue
        half = 20
        if y < half + edge or x < half + edge or y >= h - half - edge or x >= w - half - edge:
            excluded.append({**record, 'reason': 'too close to the frame edge'})
            continue
        fwhm = _fwhm(image, y, x, half)
        if fwhm is None or fwhm < 1.:
            excluded.append({**record, 'reason': 'not a resolved point (hot pixel or noise)'})
            continue
        reach = max(half, window_radius(fwhm))
        if glow[max(y - reach, 0):y + reach + 1, max(x - reach, 0):x + reach + 1].any():
            excluded.append({**record, 'reason': "its window reaches the planet's glow"})
            continue
        if (y < reach + edge or x < reach + edge or y >= h - reach - edge or x >= w - reach - edge
                or not coverage[y - reach:y + reach + 1, x - reach:x + reach + 1].all()):
            excluded.append({**record, 'reason': 'too close to the edge of the stacked field'})
            continue
        accepted.append({**record, 'fwhm_px': round(fwhm, 2)})
    return accepted[:MAX_MOONS], excluded, blank


def window_radius(fwhm):
    """Stacking window half-size: covers the planet-registered moon's halo."""
    return int(max(12, np.ceil(2 * fwhm)))


def _robust_fit(t, v, degree):
    keep = np.isfinite(v)
    coef = np.zeros(degree + 1)
    for _ in range(4):
        if keep.sum() <= degree + 2:
            break
        coef = np.polyfit(t[keep], v[keep], degree)
        r = v - np.polyval(coef, t)
        scale = 1.4826 * np.median(np.abs(r[keep]))
        if not scale > 0:
            break
        keep = np.isfinite(v) & (np.abs(r) < OUTLIER_SIGMA * scale)
    return coef, keep


def _flat(window, half):
    """``window`` less a plane fitted to its border."""
    yy, xx = _offsets(half)
    border = np.maximum(np.abs(yy), np.abs(xx)) > half - 3
    design = np.stack([np.ones(border.sum()), yy[border], xx[border]], 1)
    coef = np.linalg.lstsq(design, window[border], rcond=None)[0]
    return window - (coef[0] + coef[1] * yy + coef[2] * xx)


def _clipped_variance(residual):
    """Per-axis variance of (n, 2) residuals within OUTLIER_SIGMA robust deviations."""
    residual = residual[np.isfinite(residual).all(1)]
    if len(residual) < MIN_MEASURED:
        return None
    scale = 1.4826 * np.median(np.abs(residual - np.median(residual, 0)), 0)
    keep = (np.abs(residual - np.median(residual, 0)) <= OUTLIER_SIGMA * np.maximum(scale, 1e-9)).all(1)
    return residual[keep].var(0)


def calibrate(candidates, frames, t_out, *, blank=None, real_time=True, seed=0):
    """Moons from candidates and calibration frames.

    ``frames()`` yields (time, global shift (sx, sy), plane) afresh on every
    call; it is read twice: to track the moons, then to measure how far
    centroids stray on noise alone. For that a copy of the brightest moon's
    profile, at each moon's own brightness, is placed in clear sky at
    ``blank`` (y, x) in each frame. Without a blank spot the noise is taken
    from the centroids of the two halves of a checkerboard. Returns (moons,
    dropped) where dropped lists candidates too faint to track.
    """
    if not candidates:
        return [], []
    for c in candidates:
        c['sigma'] = float(np.clip(c['fwhm_px'] / 4, 1.5, 8.))
        c['half'] = int(max(8, np.ceil(4 * c['sigma']) + 3))
    tracks = [[] for _ in candidates]
    profiles = [np.zeros((2 * c['half'] + 1,) * 2) for c in candidates]
    counts = np.zeros(len(candidates))
    times, shifts = [], []
    for t, shift, plane in frames():
        times.append(t)
        shifts.append((shift[1], shift[0]))
        for k, c in enumerate(candidates):
            full = centroid(plane, c['y'] + shift[1], c['x'] + shift[0], c['sigma'], c['half'])
            if full is None:
                tracks[k].append((np.nan,) * 6)
                continue
            even = centroid(plane, *full, c['sigma'], c['half'], parity=0)
            odd = centroid(plane, *full, c['sigma'], c['half'], parity=1)
            halves = (even + odd) if even is not None and odd is not None else (np.nan,) * 4
            tracks[k].append((full[0] - shift[1], full[1] - shift[0], *halves))
            window = pull(plane, full[0], full[1], c['half'], 0, 0)
            if window is not None:
                profiles[k] += window
                counts[k] += 1
    times = np.asarray(times, dtype=np.float64) - t_out
    shifts = np.asarray(shifts, dtype=np.float64).reshape(-1, 2)
    span = float(np.ptp(times)) if len(times) else 0.
    degree = 2 if real_time and span >= QUADRATIC_S else 1
    fits, dropped = [], []
    for k, c in enumerate(candidates):
        track = np.asarray(tracks[k], dtype=np.float64).reshape(-1, 6)
        measured = np.isfinite(track[:, 0])
        if measured.sum() < MIN_MEASURED or counts[k] < MIN_MEASURED:
            dropped.append({'y': c['y'], 'x': c['x'], 'snr': c['snr'],
                            'reason': f'measured in {int(measured.sum())} of {len(track)} calibration frames'})
            continue
        coef_y, keep_y = _robust_fit(times, track[:, 0], degree)
        coef_x, keep_x = _robust_fit(times, track[:, 1], degree)
        keep = keep_y & keep_x
        # A moon moves across the detector with the planet; a hot pixel or
        # sensor defect stays put while the planet moves past it.
        if keep.sum() >= MIN_MEASURED and (np.sum((track[keep, :2] + shifts[keep]).var(0))
                                           < .5 * np.sum(track[keep, :2].var(0))):
            dropped.append({'y': c['y'], 'x': c['x'], 'snr': c['snr'],
                            'reason': 'fixed on the detector (hot pixel or sensor defect)'})
            continue
        residual = np.stack([track[keep, 0] - np.polyval(coef_y, times[keep]),
                             track[keep, 1] - np.polyval(coef_x, times[keep])], 1)
        profile = _flat(profiles[k] / counts[k], c['half'])
        fits.append({'k': k, 'c': c, 'track': track, 'keep': keep, 'coef': (coef_y, coef_x),
                     'scatter': residual.var(0), 'profile': profile, 'flux': float(profile.sum())})
    noise = {}
    usable = [f for f in fits if f['flux'] > 0]
    if blank is not None and usable:
        # The brightest moon's own stack on its measured positions is the profile
        # every moon is injected with, scaled to that moon's flux.
        brightest = max(usable, key=lambda f: f['c']['snr'])
        template = brightest['profile'] / brightest['flux']
        th = brightest['c']['half']
        margin = max(f['c']['half'] for f in usable) + th + 2
        rng = np.random.default_rng(seed)
        errors = {f['k']: [] for f in usable}
        for t, shift, plane in frames():
            by, bx = blank
            if by - margin < 0 or bx - margin < 0 or by + margin >= plane.shape[0] or bx + margin >= plane.shape[1]:
                continue
            base = np.asarray(plane[by - margin:by + margin + 1, bx - margin:bx + margin + 1], dtype=np.float64)
            for f in usable:
                patch = base.copy()
                patch[margin - th:margin + th + 1, margin - th:margin + th + 1] += f['flux'] * template
                start = rng.normal(0., np.sqrt(f['scatter']))
                got = centroid(patch, margin + start[0], margin + start[1], f['c']['sigma'], f['c']['half'])
                errors[f['k']].append((np.nan, np.nan) if got is None else (got[0] - margin, got[1] - margin))
        for k, e in errors.items():
            variance = _clipped_variance(np.asarray(e, dtype=np.float64).reshape(-1, 2))
            if variance is not None:
                noise[k] = (float(np.mean(variance)), 'injected copies of the brightest moon in clear sky')
    # Deviations from each drift model; tilt shared with well-measured moons
    # predicts each moon's deviation without its own noise.
    deviation = np.full((len(times), len(fits), 2), np.nan)
    variances, methods = [], []
    for i, f in enumerate(fits):
        track, keep, (coef_y, coef_x) = f['track'], f['keep'], f['coef']
        deviation[keep, i, 0] = track[keep, 0] - np.polyval(coef_y, times[keep])
        deviation[keep, i, 1] = track[keep, 1] - np.polyval(coef_x, times[keep])
        if f['k'] in noise:
            variance, method = noise[f['k']]
        else:
            split = np.isfinite(track[:, 2]) & keep
            difference = np.stack([track[split, 2] - track[split, 4], track[split, 3] - track[split, 5]], 1)
            # Each half has twice the full centroid's noise variance; their difference four times.
            halves = _clipped_variance(difference)
            variance, method = ((float(np.mean(f['scatter'])), 'no clear sky to measure it: treated as all noise')
                                if halves is None else (float(np.mean(halves)) / 4, 'checkerboard half centroids'))
        variances.append(variance)
        methods.append(method)
    alone = [max(float(np.mean(f['scatter'])) - v, 0.) for f, v in zip(fits, variances)]
    trusted = [i for i in range(len(fits)) if alone[i] > variances[i]]  # Seeing dominates their noise.
    moons = []
    for i, f in enumerate(fits):
        c, track, keep, (coef_y, coef_x) = f['c'], f['track'], f['keep'], f['coef']
        scatter, variance = float(np.mean(f['scatter'])), variances[i]
        others = [j for j in trusted if j != i]
        coupling, explained = {}, 0.
        if others:
            rows = np.isfinite(deviation[:, [i] + others, :]).all((1, 2))
            if rows.sum() >= MIN_MEASURED + 2 * len(others):
                betas = []
                for axis in range(2):
                    x = deviation[rows][:, others, axis]
                    y = deviation[rows, i, axis]
                    x, y = x - x.mean(0), y - y.mean()
                    beta = np.linalg.lstsq(x, y, rcond=None)[0]
                    betas.append(beta)
                    explained += float(np.var(x @ beta)) / 2
                coupling = {j: [float(betas[0][n]), float(betas[1][n])] for n, j in enumerate(others)}
        residual_seeing = max(scatter - variance - explained, 0.)
        gain = residual_seeing / (residual_seeing + variance) if residual_seeing + variance > 0 else 0.
        y, x = float(np.polyval(coef_y, 0.)), float(np.polyval(coef_x, 0.))
        unit, scale = ('px_per_min', 60.) if real_time else ('px_per_1000_frames', 1000.)
        moons.append(Moon(y, x, c['sigma'], c['half'], coef_y.tolist(), coef_x.tolist(), float(gain), {
            'detected_at': [c['y'], c['x']], 'detection_snr': c['snr'],
            'planet_registered_fwhm_px': c['fwhm_px'],
            f'drift_{unit}': [round(float(coef_y[-2]) * scale, 4), round(float(coef_x[-2]) * scale, 4)],
            'track_degree': degree,
            'calibration_frames': int(len(track)), 'measured_frames': int(keep.sum()),
            'seeing_jitter_px': round(float(np.sqrt(max(scatter - variance, 0.))), 3),
            'centroid_noise_px': round(float(np.sqrt(variance)), 3),
            'noise_measurement': methods[i],
            'shared_with_neighbours_px': round(float(np.sqrt(explained)), 3),
            'gain': round(float(gain), 3)}, coupling))
    # Coupling indices refer to positions in ``fits``; those are the moons' positions too.
    return moons, dropped


class MoonStack:
    """Per-moon windows accumulated on each moon's own position."""

    def __init__(self, moons, shape, channels, times, t_out, *, bayer=None, accum=None, weight=None,
                 counts=None):
        self.moons = list(moons)
        self.times, self.t_out = times, float(t_out)
        self.bayer = bayer
        self.shape = tuple(shape[:2])
        fwhm = max([m.detail.get('planet_registered_fwhm_px', 6.) for m in self.moons] or [6.])
        self.radius = window_radius(fwhm)
        n = 2 * self.radius + 1
        tail = () if channels == 1 else (channels,)
        self.window_shape = (n, n) + tail
        size = (len(self.moons),) + self.window_shape
        self.accum = np.zeros(size) if accum is None else accum
        self.weight = np.zeros(size) if weight is None else weight
        counts = counts or {}
        self.followed = np.array(counts.get('followed', [0] * len(self.moons)), dtype=np.int64)
        self.modelled = np.array(counts.get('modelled', [0] * len(self.moons)), dtype=np.int64)
        self.masks = self._masks()

    def _origin(self, moon):
        return int(round(moon.y)) - self.radius, int(round(moon.x)) - self.radius

    def _masks(self):
        r = self.radius
        yy, xx = _offsets(r)
        masks = []
        for k, moon in enumerate(self.moons):
            oy, ox = self._origin(moon)
            gy, gx = yy + oy + r, xx + ox + r
            d = np.hypot(gy - moon.y, gx - moon.x)
            u = np.clip((.95 * r - d) / (.25 * r), 0., 1.)
            mask = u * u * (3 - 2 * u)
            for j, other in enumerate(self.moons):
                if j != k and np.hypot(other.y - moon.y, other.x - moon.x) < 2 * r:
                    dj = np.hypot(gy - other.y, gx - other.x)
                    mask = mask / (1 + np.exp(-(dj - d) / NEIGHBOUR_SOFTNESS))
            masks.append(mask)
        return np.asarray(masks)

    def contribute(self, index, calibrated, plane, shift):
        """Per-moon (window, weight, own tilt followed) of one frame; windows leaving the frame are None."""
        t = self.times[index] - self.t_out
        predicted, measured = [], []
        for moon in self.moons:
            ry, rx = moon.relative(t)
            predicted.append((ry + shift[1], rx + shift[0]))
        crops = _crops(plane, predicted, [moon.half for moon in self.moons])
        for moon, (py, px), crop in zip(self.moons, predicted, crops):
            got = None if crop is None else centroid(crop[0], py, px, moon.sigma, moon.half, origin=crop[1],
                                                     shape=plane.shape[:2])
            deviation = None
            if got is not None:
                limit = OUTLIER_SIGMA * (moon.detail.get('seeing_jitter_px', 0.) ** 2
                                         + moon.detail.get('centroid_noise_px', 0.) ** 2) ** .5 + 1.
                if np.hypot(got[0] - py, got[1] - px) <= limit:
                    deviation = np.array([got[0] - py, got[1] - px])
            measured.append(deviation)
        out = []
        for moon, (py, px), own in zip(self.moons, predicted, measured):
            shared = np.zeros(2)
            for j, beta in moon.coupling.items():
                if measured[j] is not None:
                    shared += np.asarray(beta) * measured[j]
            move = shared + (moon.gain * (own - shared) if own is not None else 0.)
            sy, sx = py + move[0] - moon.y, px + move[1] - moon.x
            out.append(self._window(calibrated, moon, sy, sx) + (own is not None,))
        return out

    def _window(self, calibrated, moon, sy, sx):
        r = self.radius
        if self.bayer is not None:
            # Each colour's CFA samples and their sites, pulled bilinearly: ``detector.cfa_accumulate``
            # restricted to the window, with the colour masks of the window's Bayer phase cached.
            n = 2 * r + 1
            fy, fx = round(moon.y) - r + sy, round(moon.x) - r + sx
            iy, ix = int(np.floor(fy)), int(np.floor(fx))
            if iy < 0 or ix < 0 or iy + n >= calibrated.shape[0] or ix + n >= calibrated.shape[1]:
                return None, None
            mask = self._cfa_masks(n + 1, iy % 2, ix % 2)
            crop = np.asarray(calibrated[iy:iy + n + 1, ix:ix + n + 1], dtype=np.float64)[..., None] * mask
            dy, dx = fy - iy, fx - ix

            def bilinear(c):
                return ((1 - dy) * ((1 - dx) * c[:-1, :-1] + dx * c[:-1, 1:])
                        + dy * ((1 - dx) * c[1:, :-1] + dx * c[1:, 1:]))
            return bilinear(crop), bilinear(mask)
        add = pull(calibrated, sy, sx, r, moon.y, moon.x)
        if add is None:
            return None, None
        return add, np.ones_like(add)

    def _cfa_masks(self, size, py, px):
        """(size, size, 3) colour masks of a window whose origin has row and column parity (py, px)."""
        cache = self.__dict__.setdefault('_mask_cache', {})
        key = (size, py, px)
        if key not in cache:
            from planetrecon.detector import CHANNEL_INDEX, cfa_labels
            labels = cfa_labels(size, size, self.bayer, (px, py))
            cache[key] = np.stack([labels == c for c in sorted(CHANNEL_INDEX, key=CHANNEL_INDEX.get)],
                                  -1).astype(np.float64)
        return cache[key]

    def add(self, contributions, score):
        for k, (add, wt, followed) in enumerate(contributions):
            if add is None or not score > 0:
                continue
            self.accum[k] += score * add
            self.weight[k] += score * wt
            if followed:
                self.followed[k] += 1
            else:
                self.modelled[k] += 1

    def composite(self, image):
        """``image`` with each moon's own stack feathered in at its output position."""
        if not self.moons:
            return image
        out = np.array(image, copy=True)
        r = self.radius
        for k, moon in enumerate(self.moons):
            oy, ox = self._origin(moon)
            region = out[oy:oy + 2 * r + 1, ox:ox + 2 * r + 1]
            if region.shape[:2] != self.window_shape[:2]:
                continue
            w = self.weight[k]
            stacked = np.divide(self.accum[k], w, out=np.zeros_like(w), where=w > 0)
            mask = self.masks[k] if region.ndim == 2 else self.masks[k][..., None]
            mask = np.where(w > 0, mask, 0.)
            region[...] = region * (1 - mask) + stacked * mask
        return out

    def provenance(self):
        moons = []
        for k, moon in enumerate(self.moons):
            w = self.weight[k]
            plane = np.divide(self.accum[k], w, out=np.zeros_like(w), where=w > 0)
            if plane.ndim == 3:
                plane = plane.mean(-1)
            fwhm = _fwhm(plane, self.radius, self.radius, min(20, self.radius)) if w.any() else None
            moons.append({**moon.detail, 'output_position_px': [round(moon.y, 3), round(moon.x, 3)],
                          'stacked_fwhm_px': None if fwhm is None else round(fwhm, 2),
                          'frames_following_measured_tilt': int(self.followed[k]),
                          'frames_on_drift_model': int(self.modelled[k])})
        return moons

    def state(self):
        return {'moons': [m.as_dict() for m in self.moons], 'radius': self.radius,
                'counts': {'followed': self.followed.tolist(), 'modelled': self.modelled.tolist()}}


def setup(indices, read, correlate, reference, max_shift, times, t_out, *, real_time=True,
          points=(), point_reach=0., progress=None, check_cancel=None, plan=None):
    """Find and calibrate moons from up to MOON_FRAMES of ``indices``.

    ``read(index)`` returns (measurement plane as a NumPy array, registration
    plane for ``correlate``). ``plan`` (optional) is told each pass's reading
    order first, e.g. to read ahead. Returns (moons, summary).
    """
    plan = plan or (lambda order: None)
    indices = spread(indices, MOON_FRAMES)
    shifts = {}
    accum = support = None
    plan(indices)
    for step, index in enumerate(indices):
        if check_cancel is not None:
            check_cancel()
        plane, registration = read(int(index))
        s = correlate(reference, registration)
        if progress is not None:
            progress('Finding moons', step + 1, len(indices))
        if not np.all(np.isfinite(s)) or max(abs(float(v)) for v in s) > max_shift:
            continue
        s = (float(s[0]), float(s[1]))
        shifts[int(index)] = s
        moved, cover = _shifted(plane, -s[1], -s[0])
        accum = moved if accum is None else accum + moved
        support = cover if support is None else support + cover
    summary = {'calibration_frames': len(shifts), 'output_epoch_s' if real_time else 'output_epoch_frame': t_out}
    if len(shifts) < MIN_MEASURED:
        return [], {**summary, 'skipped': 'too few registered calibration frames'}
    image = np.divide(accum, support, out=np.zeros_like(accum), where=support > 0)
    coverage = support > len(shifts) - .5
    candidates, excluded, blank = detect(image, coverage, points=points, point_reach=point_reach)
    summary['excluded'] = excluded
    if not candidates:
        return [], summary
    passes = iter(('Measuring moon tracks', 'Measuring moon centroid noise'))

    def frames():
        phase = next(passes, 'Measuring moons')
        plan(list(shifts))
        for step, (index, s) in enumerate(shifts.items()):
            if check_cancel is not None:
                check_cancel()
            if progress is not None:
                progress(phase, step + 1, len(shifts))
            yield times[index], s, read(index)[0]
    moons, dropped = calibrate(candidates, frames, t_out, blank=blank, real_time=real_time)
    summary['too_faint'] = dropped
    return moons, summary
