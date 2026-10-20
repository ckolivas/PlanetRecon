"""Synthetic lucky-imaging SER with known truth (mono, 8-bit).

Per frame: global tip/tilt, smooth local distortion field (seeing), Gaussian
blur of varying width (frame quality), Poisson photon noise, read noise,
8-bit quantisation. Truth and per-frame parameters are saved alongside.
"""
import argparse
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from planetrecon.io.ser import write_ser


def smooth_field(rng, shape, corr, rms):
    f = gaussian_filter(rng.normal(size=(2, *shape)), (0, corr, corr), mode='wrap')
    f /= f.std(axis=(1, 2), keepdims=True)
    return f*rms


def main():
    p = argparse.ArgumentParser()
    p.add_argument('truth')
    p.add_argument('--out', required=True)
    p.add_argument('--frames', type=int, default=2000)
    p.add_argument('--peak', type=float, default=180.)
    p.add_argument('--seed', type=int, default=1)
    a = p.parse_args()
    truth = np.load(a.truth)
    truth = truth/truth.max()*a.peak
    rng = np.random.default_rng(a.seed)
    h, w = truth.shape
    yy, xx = np.indices((h, w), dtype=float)
    frames = np.empty((a.frames, h, w), np.uint8)
    params = []
    for k in range(a.frames):
        sigma = rng.uniform(0.7, 2.2)
        tilt = rng.normal(0, 2.0, size=2)
        local = smooth_field(rng, (h, w), 25, rng.uniform(0.3, 1.0))
        img = gaussian_filter(truth, sigma)
        img = map_coordinates(img, [yy + tilt[1] + local[1], xx + tilt[0] + local[0]], order=3, mode='constant')
        img = rng.poisson(np.clip(img, 0, None)).astype(float) + rng.normal(0, 1.5, size=img.shape)
        frames[k] = np.clip(np.round(img), 0, 255).astype(np.uint8)
        params.append((sigma, *tilt, float(np.sqrt((local**2).mean()))))
    write_ser(a.out, frames)
    np.savez(a.out + '.truth.npz', truth=truth, params=np.array(params))
    print('wrote', a.out, frames.shape)


if __name__ == '__main__':
    main()
