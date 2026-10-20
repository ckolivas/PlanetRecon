"""Predict a stack's fine-scale noise from raw frame differences and compare.

Consecutive selected frames differ by noise plus seeing changes; above ~0.36
cycles/px the object contributes almost nothing (oversampled optics), so the
difference spectrum there is 2x the single-frame noise spectrum.
"""
import sys, json
import numpy as np
from scipy.ndimage import gaussian_filter
from planetrecon.io.ser import SERSource
from planetrecon.reconstruction import ReconstructionConfig
from planetrecon.pipeline.preprocess import best_frame_mask
from planetrecon.pipeline.preprocess_cache import load_cache
from compare import load, luminance, subpixel_shift, fourier_shift, BANDS


def band_power(img, box, wnd, f):
    x = img[box]
    F = np.fft.fft2((x - (x*wnd).sum()/wnd.sum())*wnd)
    return [np.mean(abs(F[(f >= lo) & (f < hi)])**2) for lo, hi in BANDS]


capture, percent, mode, stack_png, white, ref_png = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], float(sys.argv[5]), sys.argv[6]
src = SERSource(capture)
sel, _ = load_cache(src, ReconstructionConfig(frame_preselection=True))
mask = best_frame_mask(sel, percent, mode)
idx = np.flatnonzero(mask)
w = sel.measurements[idx, 0]
neff = w.sum()**2/(w**2).sum()
best = src.read_raw(int(idx[np.argmax(w)])).astype(float)
disc = gaussian_filter(best, 3) > .3*best.max()
ys, xs = np.nonzero(disc)
box = np.s_[ys.min():ys.max()+1, xs.min():xs.max()+1]
wnd = np.outer(np.hanning(box[0].stop-box[0].start), np.hanning(box[1].stop-box[1].start))
f = np.hypot(np.fft.fftfreq(wnd.shape[0])[:, None], np.fft.fftfreq(wnd.shape[1])[None, :])
rng = np.random.default_rng(0)
pairs = [k for k in range(len(idx)-1) if idx[k+1] == idx[k]+1]
P = []
for k in rng.choice(pairs, 150, replace=False):
    a = src.read_raw(int(idx[k])).astype(float); b = src.read_raw(int(idx[k+1])).astype(float)
    dy, dx = subpixel_shift(a, b)
    b = fourier_shift(b, dy, dx)   # remove image motion before differencing
    P.append(band_power(a-b, box, wnd, f))
frame = np.mean(P, axis=0)/2
centre = np.array([(lo+hi)/2 for lo, hi in BANDS])
# Mean noise power transfer of bilinear resampling over uniform fractional shifts (both axes, isotropic approx).
bil = (1 - (1-np.cos(2*np.pi*centre/np.sqrt(2)))/3)**2
pred = frame/neff
pred_bil = pred*bil
st = luminance(load(stack_png))*white/65535
R = luminance(load(ref_png))
def match(img):
    out = np.zeros(best.shape); h, ww = min(out.shape[0], img.shape[0]), min(out.shape[1], img.shape[1])
    out[(out.shape[0]-h)//2:(out.shape[0]-h)//2+h, :ww] = img[(img.shape[0]-h)//2:(img.shape[0]-h)//2+h, :ww]
    dy, dx = subpixel_shift(best, out); return fourier_shift(out, dy, dx)
st, R = match(st), match(R)
core = disc
M = np.column_stack([R[core], np.ones(core.sum())]); a, b = np.linalg.lstsq(M, st[core], rcond=None)[0]
R = a*R + b   # AS in the PlanetRecon stack's ADU scale
ps, pr = band_power(st, box, wnd, f), band_power(R, box, wnd, f)
print(f'selected={len(idx)}  effective N (linear weights)={neff:.0f}')
print('band        frame_noise   pred_stack  pred*bilinear   PR_stack    AS_stack   (power, ADU^2 units)')
for i, (lo, hi) in enumerate(BANDS):
    print(f'{lo:.2f}-{hi:.2f}  {frame[i]:11.4g}  {pred[i]:11.4g}  {pred_bil[i]:11.4g}  {ps[i]:11.4g}  {pr[i]:11.4g}')
