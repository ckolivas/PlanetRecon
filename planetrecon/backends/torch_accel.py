"""Optional PyTorch accelerator. Selected only after a successful live probe."""

from __future__ import annotations

import numpy as np
import torch

from planetrecon.backends.base import Backend


class TorchBackend(Backend):
    name = "cuda"
    precision = "float32"

    def to_numpy(self, array: np.ndarray) -> np.ndarray:
        return np.asarray(array, dtype=np.float64)

    def phase_correlation(self, reference: np.ndarray, frame: np.ndarray) -> tuple[float, float]:
        ref = torch.as_tensor(reference, device="cuda", dtype=torch.float32)
        img = torch.as_tensor(frame, device="cuda", dtype=torch.float32)
        ref = ref - ref.mean()
        img = img - img.mean()
        fa = torch.fft.fft2(ref)
        fb = torch.fft.fft2(img)
        cross = fb * torch.conj(fa)
        cross = cross / torch.clamp(torch.abs(cross), min=1e-15)
        corr = torch.fft.ifft2(cross).real
        flat = torch.argmax(corr)
        peak_y = int(flat // corr.shape[1])
        peak_x = int(flat % corr.shape[1])
        sy = peak_y if peak_y < corr.shape[0] / 2 else peak_y - corr.shape[0]
        sx = peak_x if peak_x < corr.shape[1] / 2 else peak_x - corr.shape[1]
        return float(sx), float(sy)

    def shift(self, image: np.ndarray, shift_xy: tuple[float, float]) -> np.ndarray:
        from scipy.ndimage import shift as ndshift

        # Spatial bilinear shift stays on CPU; FFT correlation is the GPU cost.
        sy, sx = float(shift_xy[1]), float(shift_xy[0])
        return ndshift(np.asarray(image, dtype=np.float64), shift=(sy, sx), order=1, prefilter=False)
