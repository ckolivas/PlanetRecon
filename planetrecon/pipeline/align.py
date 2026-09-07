"""Truth-free, noise-suppressed subpixel translation registration."""
from __future__ import annotations

from functools import lru_cache
import numpy as np


@lru_cache(maxsize=4)
def correlation_filter(shape):
    """Spectrum of two Gaussian registration proxies (sigma 1.5 pixels).

    Retain signal amplitude instead of normalising noise and CFA lattice peaks
    to unit amplitude. Filtering affects registration only, never image data.
    """
    fy = np.fft.fftfreq(shape[0])[:, None]
    fx = np.fft.fftfreq(shape[1])[None, :]
    return np.exp(-4 * np.pi**2 * 1.5**2 * (fx*fx + fy*fy))


def correlation_peak(corr):
    """Frame displacement (x, y), with a three-point subpixel peak fit."""
    py, px = np.unravel_index(int(np.argmax(corr)), corr.shape)
    offsets = []
    for axis, p in enumerate((py, px)):
        if axis == 0:
            a, b, c = corr[[(p-1) % corr.shape[0], p, (p+1) % corr.shape[0]], px]
        else:
            a, b, c = corr[py, [(p-1) % corr.shape[1], p, (p+1) % corr.shape[1]]]
        curvature = a - 2*b + c
        delta = float(np.clip(.5*(a-c)/curvature, -.5, .5)) if curvature < 0 else 0.
        if abs(delta) < 1e-10:
            delta = 0.
        offsets.append(float(p if p < corr.shape[axis]/2 else p-corr.shape[axis]) + delta)
    return offsets[1], offsets[0]


def phase_correlation_shift(reference: np.ndarray, frame: np.ndarray) -> tuple[float, float]:
    """Amplitude-weighted correlation avoids phase-only locking to Bayer noise."""
    ref = np.asarray(reference, dtype=np.float64)
    img = np.asarray(frame, dtype=np.float64)
    cross = np.fft.fft2(img-img.mean()) * np.conj(np.fft.fft2(ref-ref.mean()))
    corr = np.fft.ifft2(cross * correlation_filter(ref.shape)).real
    return correlation_peak(corr)
