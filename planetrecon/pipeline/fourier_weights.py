"""Frequency-domain lucky-imaging weights from local-quality rank bins.

Frequency-domain lucky imaging (Garrel, Guyon & Baudoz 2012) weights each
frame separately at every spatial frequency, so coarse detail uses nearly all
frames while the finest detail comes from the luckiest ones. Weighting a frame
by its own Fourier amplitude needs that amplitude to be measurable: in
seeing-limited planetary video the per-frame signal above about 0.1 cycles/px
is far below the noise, and weighting by noisy amplitudes would weight noise.

Instead each alignment point splits its chosen frames into equal-count bins of
local-quality rank, and each bin into two halves (alternate blocks of 32
frames), accumulated separately. After stacking, every bin-stack holds
thousands of frames and is measured on narrow rings of radial frequency: its
transfer T, the cross-spectrum against the stack of all other bins
(independent noise, hence no noise bias) over that reference's power, and its
incoherent residual V from the difference of its two halves, which cancels all
common signal whatever its blur and leaves noise plus random distortion from
imperfect alignment (larger in poor frames). Image weights T^p / V, with T
non-increasing and V smooth over rank, are the inverse-variance matched filter
when p = 1 (best ring signal-to-noise); larger p favours sharpness over noise.

The rings form a smooth partition of unity in radial frequency, so equal
weights reproduce the ordinary stack exactly. The weighted ring stacks are
recombined as the ordinary stack plus each ring's difference from it. The
halves also give each result's signal-to-noise per ring from their Fourier
ring correlation.
"""
import numpy as np
from scipy.ndimage import binary_erosion, gaussian_filter

BINS = 16
MIN_BIN_FRAMES = 20  # Each rank bin keeps at least ten frames of each half.
# Halves alternate in blocks of frames, so frames sharing seeing (a few ms
# apart) fall in the same half and the halves' errors are independent.
HALF_BLOCK = 32


def half(index):
    """Half (0 or 1) of the capture-order block containing frame ``index``."""
    return (int(index) // HALF_BLOCK) % 2
# Radial ring centres in cycles per pixel; cos^2 transitions between them.
BAND_CENTRES = tuple(np.round(np.arange(0., .5001, .01), 2))
SMOOTH_BANDS = 1.5  # Gaussian smoothing of the measurements across neighbouring rings
RANK_TREND_DEGREE = 2
WINDOW_LEVEL = .1
WINDOW_TAPER_PX = 8.


def rank_bins(frames_per_point):
    """Rank bins for points stacking ``frames_per_point`` frames; accumulator 2b + h holds bin b, half h."""
    return int(np.clip(frames_per_point // MIN_BIN_FRAMES, 2, BINS))


def band_filters(shape):
    """(bands, h, w) radial filters summing to one at every frequency."""
    fy = np.fft.fftfreq(shape[0])[:, None]
    fx = np.fft.fftfreq(shape[1])[None, :]
    r = np.minimum(np.hypot(fy, fx), BAND_CENTRES[-1])
    c = np.asarray(BAND_CENTRES)
    filters = np.zeros((len(c),) + tuple(shape[:2]))
    for b in range(len(c)):
        if b > 0:
            rising = (r >= c[b-1]) & (r <= c[b])
            filters[b][rising] = np.sin(.5 * np.pi * (r[rising] - c[b-1]) / (c[b] - c[b-1])) ** 2
        if b < len(c) - 1:
            falling = (r >= c[b]) & (r < c[b+1])
            filters[b][falling] = np.cos(.5 * np.pi * (r[falling] - c[b]) / (c[b+1] - c[b])) ** 2
        else:
            filters[b][r >= c[b]] = 1.
    return filters


def luminance(image):
    return image @ np.array([.25, .5, .25]) if image.ndim == 3 else image


def _normalise(accum, weight):
    return np.divide(accum, weight, out=np.zeros_like(accum), where=weight > 0)


def window(image):
    """Tapered target support: bright pixels, eroded, then smoothed."""
    lum = luminance(image)
    mask = lum > WINDOW_LEVEL * lum.max()
    mask = binary_erosion(mask, iterations=int(WINDOW_TAPER_PX))
    return gaussian_filter(mask.astype(float), WINDOW_TAPER_PX)


def halves(bin_accum, bin_weight):
    """(bins, 2, ...) views of the accumulators: rank bin, then half."""
    return (bin_accum.reshape((-1, 2) + bin_accum.shape[1:]),
            bin_weight.reshape((-1, 2) + bin_weight.shape[1:]))


def _spectra(taper):
    def spectrum(image):
        lum = luminance(image)
        level = np.sum(lum * taper) / np.sum(taper)
        return np.fft.fft2((lum - level) * taper)
    return spectrum


def measure(bin_accum, bin_weight):
    """(bins, bands) transfer T and incoherent residual V, and (bins,) window weight D, per rank bin."""
    a, w = halves(bin_accum, bin_weight)
    full_a, full_w = a.sum(1), w.sum(1)
    total_accum, total_weight = full_a.sum(0), full_w.sum(0)
    taper = window(_normalise(total_accum, total_weight))
    filters = band_filters(taper.shape)
    spectrum = _spectra(taper)
    used = taper > 0
    D = np.array([[np.sum(luminance(w[k, h]) * taper) for h in (0, 1)] for k in range(len(a))]) / np.sum(taper)
    K, B = len(a), len(filters)
    T, V = np.full((K, B), np.nan), np.full((K, B), np.nan)
    for k in range(K):
        if not all((luminance(w[k, h])[used] > 0).all() for h in (0, 1)):
            continue  # A bin missing part of the target cannot be compared.
        own = spectrum(_normalise(full_a[k], full_w[k]))
        other = spectrum(_normalise(total_accum - full_a[k], total_weight - full_w[k]))
        power = np.abs(other) ** 2
        cross = np.real(own * np.conj(other))
        split = np.abs(spectrum(_normalise(a[k, 0], w[k, 0])) - spectrum(_normalise(a[k, 1], w[k, 1]))) ** 2
        for b, f in enumerate(filters):
            T[k, b] = np.sum(f * cross) / max(np.sum(f * power), 1e-300)
            # Half variances scale as 1/D_h; the full bin's is c / (D0 + D1).
            c = (np.sum(f * split) / np.sum(f)) / (1 / D[k, 0] + 1 / D[k, 1])
            V[k, b] = c / D[k].sum()
    return T, V, D.sum(1)


def smooth(values):
    """Gaussian smoothing across neighbouring rings, per bin."""
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(values, SMOOTH_BANDS, axis=1, mode='nearest')


def rank_trend(values, ranks):
    """Quadratic fit of log values over rank, per ring: residuals vary smoothly with quality rank,
    while their bin-to-bin scatter is estimation noise."""
    degree = min(RANK_TREND_DEGREE, len(ranks) - 1)
    if degree < 1:
        return values
    x = ranks / max(ranks.max(), 1)
    coefficients = np.polyfit(x, np.log(values), degree)
    return np.exp(np.polyval(coefficients, x[:, None]) if values.ndim == 1 else
                  np.stack([np.polyval(coefficients[:, b], x) for b in range(values.shape[1])], axis=1))


def monotone(values):
    """Non-increasing least-squares fit (pool adjacent violators), best bin first."""
    blocks = []
    for v in values:
        blocks.append([float(v), 1])
        while len(blocks) > 1 and blocks[-2][0] < blocks[-1][0]:
            v2, n2 = blocks.pop()
            v1, n1 = blocks.pop()
            blocks.append([(v1 * n1 + v2 * n2) / (n1 + n2), n1 + n2])
    return np.concatenate([[v] * n for v, n in blocks])


def weights(T, V, D, power):
    """(bins, bands) accumulator weights T^power / (V D): image weights T^power / V.

    T is made non-increasing over rank and clipped at zero, V follows a smooth
    trend over rank; a band without measurable transfer keeps equal weights.
    """
    valid = np.isfinite(T[:, 0])
    W = np.zeros_like(T)
    if not valid.any():
        return np.ones_like(T)
    t = smooth(T[valid])
    t = np.stack([monotone(t[:, b]) for b in range(t.shape[1])], axis=1)
    v = rank_trend(np.maximum(smooth(V[valid]), 1e-300), np.flatnonzero(valid))
    W[valid] = np.clip(t, 0., None) ** power / (v * D[valid, None])
    # A bin that could not be measured takes its nearest measured neighbour's weights.
    measured = np.flatnonzero(valid)
    for k in np.flatnonzero(~valid):
        W[k] = W[measured[np.argmin(abs(measured - k))]]
    W /= np.maximum(W.max(0), 1e-300)
    W[:, W.sum(0) <= 0] = 1.
    return W


def combine(bin_accum, bin_weight, W, half=None):
    """Ordinary stack plus each band's weighted-bin difference from it; one half with ``half``."""
    a, w = halves(bin_accum, bin_weight)
    a, w = (a.sum(1), w.sum(1)) if half is None else (a[:, half], w[:, half])
    accum, weight = a.sum(0), w.sum(0)
    base = _normalise(accum, weight)
    filters = band_filters(base.shape)
    out = base.copy()
    for b, f in enumerate(filters):
        if np.allclose(W[:, b], W[0, b], rtol=1e-12, atol=0.):
            continue  # Equal weights reproduce the ordinary stack.
        band = _normalise(np.tensordot(W[:, b], a, axes=1), np.tensordot(W[:, b], w, axes=1))
        difference = np.where(weight > 0, band - base, 0.)
        axes = (0, 1)
        out += np.real(np.fft.ifft2(np.fft.fft2(difference, axes=axes) * (f[..., None] if difference.ndim == 3 else f),
                                    axes=axes))
    return np.where(weight > 0, out, 0.)


def ring_snr(bin_accum, bin_weight, W):
    """Signal-to-noise power per band of the weighted stack, from its halves' Fourier ring correlation.

    Each half has half the frames, so the full stack's ratio is twice a half's.
    """
    first, second = (combine(bin_accum, bin_weight, W, half) for half in (0, 1))
    taper = window(first + second)
    spectrum = _spectra(taper)
    f0, f1 = spectrum(first), spectrum(second)
    cross, p0, p1 = np.real(f0 * np.conj(f1)), np.abs(f0) ** 2, np.abs(f1) ** 2
    frc = np.array([np.sum(f * cross) / np.sqrt(np.sum(f * p0) * np.sum(f * p1)) for f in band_filters(taper.shape)])
    return 2 * np.clip(frc, 0., 1 - 1e-12) / (1 - np.clip(frc, 0., 1 - 1e-12))


def effective_frames(bin_weight, W):
    """Per band, the effective share of frames: (sum w n)^2 / (n sum w^2 n) for bin weights w and sizes n."""
    n = halves(bin_weight, bin_weight)[1].reshape(W.shape[0], -1).sum(1)
    return [float((np.sum(W[:, b] * n) ** 2) / (np.sum(n) * np.sum(W[:, b] ** 2 * n))) for b in range(W.shape[1])]


def apply(bin_accum, bin_weight, power):
    """Fourier-weighted image and its diagnostics."""
    T, V, D = measure(bin_accum, bin_weight)
    W = weights(T, V, D, power)
    image = combine(bin_accum, bin_weight, W)
    clean = lambda a: np.where(np.isfinite(a), a, None).tolist()
    return image, {
        'method': 'local-quality rank bins split into halves of alternate 32-frame blocks; per-ring transfer T '
                  'against the stack of the other bins and incoherent residual V from the difference of the halves; image weights T^p / V '
                  'after smoothing across rings and a non-increasing fit of T over rank',
        'bins': len(T), 'power': power, 'band_centres_cycles_per_px': list(BAND_CENTRES),
        'transfer': clean(T), 'residual': clean(V), 'weights': W.tolist(),
        'effective_frame_share': effective_frames(bin_weight, W),
        'ring_snr': ring_snr(bin_accum, bin_weight, W).tolist(),
        'ordinary_ring_snr': ring_snr(bin_accum, bin_weight, np.ones_like(W)).tolist(),
    }
