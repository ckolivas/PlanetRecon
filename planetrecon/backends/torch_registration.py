"""CUDA field-only reference pulls for ring drift tracking (torch_surface.TorchRingTracker)."""

import torch

from planetrecon.backends.torch_globe import TorchGlobeWarp


class TorchRingRegistrationOps(TorchGlobeWarp):
    """Field-only reference pulls with the CPU operator's pixel centres and support."""

    def __init__(self, shape, device='cuda:0'):
        super().__init__(shape, None, device)

    def map(self, src, ref):
        P = self.poses(src, ref)
        x, y = self.rotate(self.x-P([s.cx for s in P.srcs]), P([s.cy for s in P.srcs])-self.y,
                           P.angles([r.field_angle_rad-s.field_angle_rad for s, r in zip(P.srcs, P.refs)]))
        x, y = x+P([r.cx for r in P.refs]), P([r.cy for r in P.refs])-y
        return x, y, torch.isfinite(x) & torch.isfinite(y)
