"""Truth-free registration helpers."""

from __future__ import annotations

import numpy as np


def phase_correlation_shift(reference: np.ndarray, frame: np.ndarray) -> tuple[float, float]:
    ref = np.asarray(reference, dtype=np.float64)
    img = np.asarray(frame, dtype=np.float64)
    ref = ref - ref.mean()
    img = img - img.mean()
    fa = np.fft.fft2(ref)
    fb = np.fft.fft2(img)
    # Peak of F_frame * conj(F_ref) is the (y, x) displacement of frame vs ref.
    cross = fb * np.conj(fa)
    denom = np.maximum(np.abs(cross), 1e-15)
    corr = np.fft.ifft2(cross / denom).real
    peak_y, peak_x = np.unravel_index(int(np.argmax(corr)), corr.shape)
    sy = peak_y if peak_y < corr.shape[0] / 2.0 else peak_y - corr.shape[0]
    sx = peak_x if peak_x < corr.shape[1] / 2.0 else peak_x - corr.shape[1]
    return float(sx), float(sy)
