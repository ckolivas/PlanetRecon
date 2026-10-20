"""Score stacks of the synthetic capture against the known truth.

After global + dense registration and a linear intensity fit, per band:
  T   = Re<S truth*>/<|truth|^2>   (signal transfer: frame blur, alignment, resampling)
  N   = <|S - T truth|^2>          (everything not explained by the object: noise + errors)
  SNR = T^2 <|truth|^2> / N        (recoverable detail; invariant to linear filtering)
"""
import sys
import numpy as np
from scipy.ndimage import gaussian_filter
from compare import luminance, BANDS
from warp import register, interior_spectra


def score(path, truth):
    with np.load(path) as d:
        S = luminance(d['image'])
    mask = gaussian_filter(truth, 2) > .15*truth.max()
    (S,), field = register(truth, [S], mask)
    M = np.column_stack([S[mask], np.ones(mask.sum())])
    a, b = np.linalg.lstsq(M, truth[mask], rcond=None)[0]
    S = a*S + b
    FS, FT, box = interior_spectra(S, truth, mask)
    f = np.hypot(np.fft.fftfreq(FS.shape[0])[:, None], np.fft.fftfreq(FS.shape[1])[None, :])
    rows = []
    for lo, hi in BANDS:
        m = (f >= lo) & (f < hi)
        pt = np.mean(abs(FT[m])**2)
        T = np.real(np.mean(FS[m]*np.conj(FT[m])))/pt
        N = np.mean(abs(FS[m] - T*FT[m])**2)
        rows.append((T, T*T*pt/N))
    return rows


if __name__ == '__main__':
    truth = np.load(sys.argv[1])['truth']
    print('transfer T / SNR per band (cycles/px)')
    print(' '*26 + ''.join(f'{lo:.2f}-{hi:.2f}'.rjust(15) for lo, hi in BANDS[:8]))
    for p in sys.argv[2:]:
        r = score(p, truth)
        print(p.split('/')[-1][:25].ljust(26) + ''.join(f'{t:6.3f}/{s:<8.3g}' for t, s in r[:8]))
