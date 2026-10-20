"""Per-band SNR of a half-stacked candidate and of an AutoStakkert reference.

Candidate C=(A+B)/2 has noise power N_c=<|A-B|^2>/4 and signal S_c=P_c-N_c.
The reference shares only the object: X=<C R*>=Hc Hr* S, so S_r=|X|^2/S_c and
N_r=P_r-S_r. SNR is invariant to any linear filtering (interpolation kernels,
post-blur), so it measures recoverable detail independent of sharpness.
"""
import sys
import json
import numpy as np
from compare import luminance, subpixel_shift, fourier_shift, window, load, BANDS
from scipy.ndimage import binary_dilation, gaussian_filter


def analyse(npz, ref_path, quiet=False):
    with np.load(npz) as d:
        A, B, T = luminance(d['a']), luminance(d['b']), luminance(d['image'])
        info = json.loads(str(d['info']))
    R = luminance(load(ref_path))
    h, w = min(R.shape[0], T.shape[0]), min(R.shape[1], T.shape[1])
    def crop(a):
        y0, x0 = (a.shape[0]-h)//2, (a.shape[1]-w)//2
        return a[y0:y0+h, x0:x0+w]
    R, A, B, T = map(crop, (R, A, B, T))
    dy, dx = subpixel_shift(R, T)
    A, B, T = (fourier_shift(x, dy, dx) for x in (A, B, T))
    mask = gaussian_filter(R, 2) > .08*R.max()
    core = mask & ~binary_dilation(~mask, iterations=6)
    M = np.column_stack([T[core], np.ones(core.sum())])
    a, b = np.linalg.lstsq(M, R[core], rcond=None)[0]
    A, B = a*A+b, a*B+b
    box, wnd = window(mask)
    def spec(x):
        x = x[box]
        return np.fft.fft2((x - x.mean())*wnd)
    FA, FB, FR = spec(A), spec(B), spec(R)
    FC = (FA+FB)/2
    f = np.hypot(np.fft.fftfreq(FC.shape[0])[:, None], np.fft.fftfreq(FC.shape[1])[None, :])
    rows = []
    for lo, hi in BANDS:
        m = (f >= lo) & (f < hi)
        Pc = np.mean(abs(FC[m])**2); Nc = np.mean(abs(FA[m]-FB[m])**2)/4
        Sc = max(Pc-Nc, 1e-30)
        X = abs(np.mean(FC[m]*np.conj(FR[m])))
        Pr = np.mean(abs(FR[m])**2)
        Sr = min(X**2/Sc, Pr)
        Nr = max(Pr-Sr, 1e-30)
        rows.append(dict(band=[lo, hi], snr_c=Sc/Nc, snr_r=Sr/Nr, rel_mtf=float(np.sqrt(Sc/Sr)), noise_ratio=float(np.sqrt(Nc/Nr))))
    if not quiet:
        print(f"{npz}  n={info.get('n')}  shift=({dy:+.2f},{dx:+.2f})")
        print('  band        SNR_cand   SNR_AS   ratio   amp(c/AS)  noise(c/AS)')
        for r in rows:
            print(f"  {r['band'][0]:.2f}-{r['band'][1]:.2f}  {r['snr_c']:9.3g} {r['snr_r']:9.3g}  {r['snr_c']/r['snr_r']:6.2f}   {r['rel_mtf']:6.3f}    {r['noise_ratio']:6.3f}")
    return rows


DENSE = False


def self_snr(npz, quiet=False):
    """SNR of the full stack from its two independent halves (B registered to A)."""
    with np.load(npz) as d:
        A, B = luminance(d['a']), luminance(d['b'])
        info = json.loads(str(d['info']))
    dy, dx = subpixel_shift(A, B)
    B = fourier_shift(B, dy, dx)
    mask = gaussian_filter(A, 2) > .08*A.max()
    if DENSE:
        from warp import dense_field, warp
        B = warp(B, dense_field(A, B, mask=mask))
    box, wnd = window(mask)
    def spec(x):
        x = x[box]
        return np.fft.fft2((x - x.mean())*wnd)
    FA, FB = spec(A), spec(B)
    f = np.hypot(np.fft.fftfreq(FA.shape[0])[:, None], np.fft.fftfreq(FA.shape[1])[None, :])
    scale = np.median(A[mask])
    rows = []
    for lo, hi in BANDS:
        m = (f >= lo) & (f < hi)
        S = np.real(np.mean(FA[m]*np.conj(FB[m])))
        Nh = (np.mean(abs(FA[m])**2) + np.mean(abs(FB[m])**2))/2 - S
        rows.append(dict(band=[lo, hi], snr=S/(Nh/2), signal=np.sqrt(max(S, 0))/scale, noise=np.sqrt(Nh/2)/scale))
    if not quiet:
        print(f"{npz.split('/')[-1]}  n={info.get('n')}  B->A=({dy:+.2f},{dx:+.2f})")
        print('   ' + ' '.join(f"{r['band'][0]:.2f}" for r in rows))
        print('   ' + ' '.join(f"{r['snr']:4.3g}".rjust(4) for r in rows))
    return rows


def table(paths):
    rows = {p: self_snr(p, quiet=True) for p in paths}
    print('SNR of full stack per band (cycles/px); higher is better')
    print(' '*28 + ''.join(f"{lo:.2f}-{hi:.2f} ".rjust(11) for lo, hi in BANDS))
    for p, r in rows.items():
        print(p.split('/')[-1][:27].ljust(28) + ''.join(f"{x['snr']:10.3g} " for x in r))


if __name__ == '__main__' and sys.argv[1] in ('--self', '--dense'):
    DENSE = sys.argv[1] == '--dense'
    table(sys.argv[2:])
elif __name__ == '__main__':
    for npz in sys.argv[2:]:
        analyse(npz, sys.argv[1])
