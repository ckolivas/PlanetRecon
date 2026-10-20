"""CUDA field-only reference pulls for ring drift tracking (torch_surface.TorchRingTracker)."""

import torch

from planetrecon.backends.torch_globe import TorchGlobeWarp


class TorchRingRegistrationOps(TorchGlobeWarp):
    """Field-only reference pulls with the CPU operator's pixel centres and support."""

    def __init__(self, shape, device='cuda:0'):
        super().__init__(shape, None, device)

    def map(self, src, ref):
        x, y = self.rotate(self.x-src.cx, src.cy-self.y, ref.field_angle_rad-src.field_angle_rad)
        x, y = x+ref.cx, ref.cy-y
        return x, y, torch.isfinite(x) & torch.isfinite(y)
