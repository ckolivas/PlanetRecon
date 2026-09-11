"""CUDA numerics for ring drift matching; peak acceptance remains shared with CPU."""

import torch
from scipy.fft import next_fast_len

from planetrecon.backends.torch_globe import TorchGlobeWarp


class TorchRingRegistrationOps(TorchGlobeWarp):
    """Field-only reference pulls and float64 normalized linear correlations.

    Padding represents unobserved pixels, with the same complete-mask support
    test as the CPU matcher. FFT storage is bounded by one detector, not by
    capture length. Only the score map returns to the shared CPU peak checks.
    """

    def __init__(self, shape, device='cuda:0'):
        super().__init__(shape, None, device)
        self.full_shape = tuple(2*n-1 for n in shape)
        self.fft_shape = tuple(next_fast_len(n, real=True) for n in self.full_shape)
        self.ones_fft = torch.fft.rfft2(torch.ones(shape, device=device, dtype=torch.float64),
                                      s=self.fft_shape)

    def map(self, src, ref):
        x, y = self.rotate(self.x-src.cx, src.cy-self.y, ref.field_angle_rad-src.field_angle_rad)
        x, y = x+ref.cx, ref.cy-y
        return x, y, torch.isfinite(x) & torch.isfinite(y)

    def scores(self, proxy, template, mask, count, energy):
        proxy = self.tensor(proxy)
        observed = torch.fft.rfft2(torch.stack((proxy, proxy.square())), s=self.fft_shape)
        kernels = torch.fft.rfft2(torch.stack((self.tensor(template), self.tensor(mask))).flip((-2,-1)),
                                  s=self.fft_shape)
        products = torch.stack((observed[0]*kernels[0], observed[0]*kernels[1],
                                observed[1]*kernels[1], self.ones_fft*kernels[1]))
        correlations = torch.fft.irfft2(products, s=self.fft_shape)
        dot, total, squares, support = correlations[:, :self.full_shape[0], :self.full_shape[1]]
        denominator = (energy*(squares-total.square()/count).clamp_min(0.)).sqrt()
        valid = (support >= count-1e-6) & (denominator > 1e-12)
        return torch.where(valid, dot/denominator, -torch.inf).cpu().numpy()
