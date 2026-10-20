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


def peak_neighbours(py, px, shape):
    """Wrapped (rows, columns) of the vertical and horizontal three-point stencils."""
    h, w = shape
    return ([(py-1) % h, py, (py+1) % h], [px]*3), ([py]*3, [(px-1) % w, px, (px+1) % w])


def peak_offsets(py, px, shape, vertical, horizontal):
    """Displacement (x, y) from the integer peak and its two three-point stencils."""
    offsets = []
    for axis, (p, (a, b, c)) in enumerate(((py, vertical), (px, horizontal))):
        curvature = a - 2*b + c
        delta = float(np.clip(.5*(a-c)/curvature, -.5, .5)) if curvature < 0 else 0.
        if abs(delta) < 1e-10:
            delta = 0.
        offsets.append(float(p if p < shape[axis]/2 else p-shape[axis]) + delta)
    return offsets[1], offsets[0]


def correlation_peak(corr):
    """Frame displacement (x, y), with a three-point subpixel peak fit."""
    py, px = np.unravel_index(int(np.argmax(corr)), corr.shape)
    vertical, horizontal = peak_neighbours(py, px, corr.shape)
    return peak_offsets(py, px, corr.shape, corr[vertical], corr[horizontal])


def phase_correlation_shift(reference: np.ndarray, frame: np.ndarray, *, reference_spectrum=None) -> tuple[float, float]:
    """Amplitude-weighted correlation avoids phase-only locking to Bayer noise."""
    ref = np.asarray(reference, dtype=np.float64)
    img = np.asarray(frame, dtype=np.float64)
    if reference_spectrum is None:
        reference_spectrum = np.conj(np.fft.fft2(ref-ref.mean()))
    cross = np.fft.fft2(img-img.mean()) * reference_spectrum
    corr = np.fft.ifft2(cross * correlation_filter(ref.shape)).real
    return correlation_peak(corr)
