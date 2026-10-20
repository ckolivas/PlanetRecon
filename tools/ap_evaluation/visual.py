"""Side-by-side crops with identical a-trous wavelet sharpening (after alignment/intensity match)."""
import sys
import numpy as np
from scipy.ndimage import convolve1d
from compare import load, prepare


def atrous(img, levels=5):
    k = np.array([1, 4, 6, 4, 1], float)/16
    layers, cur = [], img
    for j in range(levels):
        kern = np.zeros(4*2**j+1)
        kern[::2**j] = k
        smooth = convolve1d(convolve1d(cur, kern, axis=0, mode='reflect'), kern, axis=1, mode='reflect')
        layers.append(cur - smooth)
        cur = smooth
    return layers, cur


def sharpen(img, gains):
    layers, residual = atrous(img, len(gains))
    return residual + sum(g*l for g, l in zip(gains, layers))


def save_png(path, img):
    from PySide6.QtGui import QImage
    a = np.clip(img, 0, 1)
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, axis=2)
    u8 = np.ascontiguousarray((a*255+.5).astype(np.uint8))
    h, w = u8.shape[:2]
    QImage(u8.data, w, h, 3*w, QImage.Format.Format_RGB888).copy().save(str(path))


SCALE = 1.6


def panel(images, labels, gains, out, zoom=2, box=None):
    rows = []
    ref = images[0]
    scale = np.percentile(ref, 99.9) * SCALE
    for img in images:
        s = sharpen(img, gains)/scale
        crop = s[box] if box is not None else s
        rows.append(np.kron(crop, np.ones((zoom, zoom))))
    gap = np.ones((rows[0].shape[0], 6))
    save_png(out, np.hstack(sum([[r, gap] for r in rows], [])[:-1]))


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('reference')
    p.add_argument('candidates', nargs='+')
    p.add_argument('--out', required=True)
    p.add_argument('--gains', default='6,4,2,1.2,1')
    p.add_argument('--blur', type=float, default=0)
    p.add_argument('--box', default=None, help='y0,y1,x0,x1')
    p.add_argument('--zoom', type=int, default=2)
    p.add_argument('--scale', type=float, default=1.6)
    a = p.parse_args()
    SCALE = a.scale
    from scipy.ndimage import gaussian_filter
    ref = load(a.reference)
    from compare import luminance
    R = luminance(ref)
    imgs = [R]
    for c in a.candidates:
        C, R2, mask, shift = prepare(load(c), ref)
        if a.blur:
            C = gaussian_filter(C, a.blur)
        imgs.append(C)
    h = min(i.shape[0] for i in imgs); w = min(i.shape[1] for i in imgs)
    imgs = [i[:h, :w] for i in imgs]
    box = None if a.box is None else np.s_[tuple(slice(*map(int, a.box.split(',')[k:k+2])) for k in (0, 2))]
    panel(imgs, None, [float(g) for g in a.gains.split(',')], a.out, a.zoom, box)
