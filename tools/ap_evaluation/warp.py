"""Dense registration of a candidate onto a reference (tile correlations) and
interior-only spectral comparison, for captures with rotation/seeing geometry
differences (e.g. Jupiter)."""
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates, binary_erosion, distance_transform_edt
from compare import subpixel_shift, fourier_shift, luminance, BANDS


def bandpass(a):
    a = gaussian_filter(a, 1.0)
    return a - gaussian_filter(a, 6.0)


def dense_field(ref, img, tile=48, step=12, mask=None):
    """Displacement (dy, dx) to sample img at so it matches ref."""
    R, I = bandpass(ref), bandpass(img)
    h, w = ref.shape
    ys = np.arange(tile//2, h-tile//2, step)
    xs = np.arange(tile//2, w-tile//2, step)
    F = np.full((2, len(ys), len(xs)), np.nan)
    wnd = np.outer(np.hanning(tile), np.hanning(tile))
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            if mask is not None and not mask[y, x]:
                continue
            s = np.s_[y-tile//2:y+tile//2, x-tile//2:x+tile//2]
            r, c = R[s], I[s]
            if r.std() < 1e-9 or c.std() < 1e-9:
                continue
            dy, dx = subpixel_shift(r*wnd, c*wnd, factor=50)
            if max(abs(dy), abs(dx)) < 4:
                F[:, j, i] = dy, dx
    # Fill gaps with the median and smooth.
    for a in range(2):
        g = F[a]
        g[np.isnan(g)] = np.nanmedian(g)
        F[a] = gaussian_filter(g, 1.0)
    yy, xx = np.indices((h, w), dtype=float)
    cy = np.clip((yy - ys[0])/step, 0, len(ys)-1)
    cx = np.clip((xx - xs[0])/step, 0, len(xs)-1)
    return np.array([map_coordinates(F[a], [cy, cx], order=1, mode='nearest') for a in range(2)])


def warp(img, field):
    yy, xx = np.indices(img.shape, dtype=float)
    # Sampling at y - dy, x - dx matches the "shift img by (dy, dx)" convention.
    return map_coordinates(img, [yy - field[0], xx - field[1]], order=5, mode='nearest')


def register(ref, cand_list, mask):
    """Global Fourier shift, then dense warp estimated on the first candidate."""
    def fit(c):
        out = np.zeros(ref.shape)
        h, w = min(ref.shape[0], c.shape[0]), min(ref.shape[1], c.shape[1])
        ry, rx = (ref.shape[0]-h)//2, (ref.shape[1]-w)//2
        cy, cx = (c.shape[0]-h)//2, (c.shape[1]-w)//2
        out[ry:ry+h, rx:rx+w] = c[cy:cy+h, cx:cx+w]
        return out
    cand_list = [fit(c) for c in cand_list]
    dy, dx = subpixel_shift(ref, cand_list[0])
    moved = [fourier_shift(c, dy, dx) for c in cand_list]
    field = dense_field(ref, moved[0], mask=mask)
    return [warp(c, field) for c in moved], field


def interior_window(mask, erode=10, taper=12):
    inner = binary_erosion(mask, iterations=erode)
    d = distance_transform_edt(inner)
    return np.clip(d/taper, 0, 1)**2 * (3-2*np.clip(d/taper, 0, 1))


def interior_spectra(C, R, mask):
    wnd = interior_window(mask)
    ys, xs = np.nonzero(wnd > 0)
    box = np.s_[ys.min():ys.max()+1, xs.min():xs.max()+1]
    wb = wnd[box]
    def spec(x):
        x = x[box]
        m = (x*wb).sum()/wb.sum()
        return np.fft.fft2((x-m)*wb)
    return spec(C), spec(R), box


def band_table(FC, FR):
    f = np.hypot(np.fft.fftfreq(FC.shape[0])[:, None], np.fft.fftfreq(FC.shape[1])[None, :])
    rows = []
    for lo, hi in BANDS:
        m = (f >= lo) & (f < hi)
        X = np.mean(FC[m]*np.conj(FR[m]))
        pc, pr = np.mean(abs(FC[m])**2), np.mean(abs(FR[m])**2)
        rows.append(dict(band=(lo, hi), rel=float(np.real(X)/pr), coh=float(abs(X)/np.sqrt(pc*pr)), amp=float(np.sqrt(pc/pr))))
    return rows
