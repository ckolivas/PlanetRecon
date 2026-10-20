"""Zoomed, contrast-stretched crops of a reference and densely registered candidates."""
import sys, argparse
import numpy as np
from compare import load, luminance
from warp import register
from visual import sharpen, save_png
from scipy.ndimage import gaussian_filter

p = argparse.ArgumentParser()
p.add_argument('reference'); p.add_argument('candidates', nargs='+')
p.add_argument('--out', required=True); p.add_argument('--box', required=True)
p.add_argument('--zoom', type=int, default=3); p.add_argument('--gains', default='1,1,1,1,1;5,3,1.5,1,1')
p.add_argument('--blur', default=None, help='comma list of gaussian sigmas per candidate')
a = p.parse_args()
R = luminance(load(a.reference))
mask = gaussian_filter(R, 2) > .15*R.max()
cands = []
for c in a.candidates:
    if c.endswith('.npz'):
        with np.load(c) as d: T = luminance(d['image'])
    else:
        T = luminance(load(c))
    (T,), field = register(R, [T], mask)
    M = np.column_stack([T[mask], np.ones(mask.sum())]); k, b = np.linalg.lstsq(M, R[mask], rcond=None)[0]
    cands.append(k*T+b)
if a.blur:
    cands = [gaussian_filter(c, float(s)) if float(s) else c for c, s in zip(cands, a.blur.split(','))]
y0, y1, x0, x1 = map(int, a.box.split(','))
box = np.s_[y0:y1, x0:x1]
rows = []
for g in a.gains.split(';'):
    g = [float(v) for v in g.split(',')]
    imgs = [sharpen(x, g)[box] for x in [R] + cands]
    lo, hi = np.percentile(imgs[0], [0.5, 99.5])
    gap = np.ones(((y1-y0)*a.zoom, 4))
    tiles = [np.kron((i-lo)/(hi-lo), np.ones((a.zoom, a.zoom))) for i in imgs]
    rows.append(np.hstack(sum([[t, gap] for t in tiles], [])[:-1]))
    rows.append(np.ones((4, rows[-1].shape[1])))
save_png(a.out, np.vstack(rows[:-1]))
