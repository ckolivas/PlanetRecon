"""Compare a stack against an AutoStakkert reference of the same capture.

Both images are reduced to luminance, the candidate is registered to the
reference with a Fourier (non-blurring) subpixel shift and matched linearly in
intensity on the planet. Reported per radial frequency band (cycles/pixel):
  rel_mtf  Re<C conj(R)> / <|R|^2>  (>1: candidate retains more of that band)
  coh      |<C conj(R)>| / sqrt(<|C|^2><|R|^2>)  (shared signal fraction)
  pc, pr   band power of candidate / reference (matched intensity)
"""
import sys
import numpy as np
from scipy.ndimage import binary_dilation, gaussian_filter

BANDS = [(0.01, .04), (.04, .08), (.08, .12), (.12, .16), (.16, .20), (.20, .25), (.25, .30), (.30, .36), (.36, .42), (.42, .5)]


def luminance(a):
    a = np.asarray(a, float)
    return a @ np.array([.25, .5, .25]) if a.ndim == 3 else a


def _upsampled_dft(data, region, factor, offsets):
    """Matrix-multiply DFT of a small upsampled region (Guizar-Sicairos 2008)."""
    h, w = data.shape
    fy = np.fft.fftfreq(h)*h
    fx = np.fft.fftfreq(w)*w
    col = np.exp((2j*np.pi/(w*factor)) * fx[:, None] * (np.arange(region) - offsets[1])[None, :])
    row = np.exp((2j*np.pi/(h*factor)) * (np.arange(region) - offsets[0])[:, None] * fy[None, :])
    return row @ data @ col


def subpixel_shift(ref, img, factor=100):
    """Translation (dy, dx) to apply to img so it matches ref (upsampled cross-correlation)."""
    ref = ref - ref.mean()
    img = img - img.mean()
    wnd = np.outer(np.hanning(ref.shape[0]), np.hanning(ref.shape[1]))
    X = np.fft.fft2(ref*wnd) * np.conj(np.fft.fft2(img*wnd))
    c = np.real(np.fft.ifft2(X))
    h, w = c.shape
    y, x = np.unravel_index(c.argmax(), c.shape)
    y = y - h if y > h//2 else y
    x = x - w if x > w//2 else x
    region = int(np.ceil(factor*1.5))
    centre = region//2
    up = np.real(_upsampled_dft(X, region, factor, (centre - y*factor, centre - x*factor)))
    py, px = np.unravel_index(up.argmax(), up.shape)
    return y + (py - centre)/factor, x + (px - centre)/factor


def fourier_shift(img, dy, dx):
    h, w = img.shape
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    return np.real(np.fft.ifft2(np.fft.fft2(img) * np.exp(-2j*np.pi*(fy*dy + fx*dx))))


def prepare(candidate, reference):
    R = luminance(reference)
    C = luminance(candidate)
    h, w = min(R.shape[0], C.shape[0]), min(R.shape[1], C.shape[1])
    # Crop both to a common shape centred on the reference planet.
    def crop(a):
        y0, x0 = (a.shape[0]-h)//2, (a.shape[1]-w)//2
        return a[y0:y0+h, x0:x0+w]
    R, C = crop(R), crop(C)
    dy, dx = subpixel_shift(R, C)
    C = fourier_shift(C, dy, dx)
    mask = gaussian_filter(R, 2) > .08*R.max()
    core = mask & ~binary_dilation(~mask, iterations=6)
    A = np.column_stack([C[core], np.ones(core.sum())])
    a, b = np.linalg.lstsq(A, R[core], rcond=None)[0]
    return a*C + b, R, mask, (dy, dx)


def window(mask):
    """Planet bounding box with a cosine apodisation, 16 px margin."""
    ys, xs = np.nonzero(mask)
    y0, y1 = max(ys.min()-16, 0), min(ys.max()+17, mask.shape[0])
    x0, x1 = max(xs.min()-16, 0), min(xs.max()+17, mask.shape[1])
    wy, wx = np.hanning(y1-y0), np.hanning(x1-x0)
    return np.s_[y0:y1, x0:x1], np.sqrt(np.outer(wy, wx))


def spectra(C, R, mask):
    box, wnd = window(mask)
    c = (C[box] - C[box].mean())*wnd
    r = (R[box] - R[box].mean())*wnd
    FC, FR = np.fft.fft2(c), np.fft.fft2(r)
    fy = np.fft.fftfreq(c.shape[0])[:, None]
    fx = np.fft.fftfreq(c.shape[1])[None, :]
    f = np.hypot(fy, fx)
    rows = []
    for lo, hi in BANDS:
        m = (f >= lo) & (f < hi)
        cross = np.mean(FC[m]*np.conj(FR[m]))
        pc, pr = np.mean(abs(FC[m])**2), np.mean(abs(FR[m])**2)
        rows.append(dict(band=(lo, hi), rel_mtf=float(np.real(cross)/pr), coh=float(abs(cross)/np.sqrt(pc*pr)),
                         pc=float(pc), pr=float(pr)))
    return rows


def report(candidate, reference, label=''):
    C, R, mask, shift = prepare(candidate, reference)
    rows = spectra(C, R, mask)
    print(f'{label} shift=({shift[0]:+.2f},{shift[1]:+.2f})')
    print('  band        rel_mtf  coh    sqrt(pc/pr)')
    for r in rows:
        print(f"  {r['band'][0]:.2f}-{r['band'][1]:.2f}   {r['rel_mtf']:6.3f}  {r['coh']:.3f}  {np.sqrt(r['pc']/r['pr']):6.3f}")
    return rows


def load(path):
    if str(path).endswith('.npz'):
        with np.load(path) as d:
            return d['image']
    if str(path).endswith('.npy'):
        return np.load(path)
    from imgio import read_png
    return read_png(path)[0]


if __name__ == '__main__':
    ref = load(sys.argv[1])
    for p in sys.argv[2:]:
        report(load(p), ref, label=p)
