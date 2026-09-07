"""Optional PyTorch accelerator. Selected only after a successful live probe."""

from __future__ import annotations

import numpy as np
import torch

from planetrecon.backends.base import Backend


class TorchBackend(Backend):
    name = "cuda"
    precision = "float64"

    def to_numpy(self, array: np.ndarray) -> np.ndarray:
        return np.asarray(array, dtype=np.float64)

    def phase_correlation(self, reference: np.ndarray, frame: np.ndarray) -> tuple[float, float]:
        ref = torch.as_tensor(reference, device="cuda:0", dtype=torch.float64)
        img = torch.as_tensor(frame, device="cuda:0", dtype=torch.float64)
        ref = ref - ref.mean()
        img = img - img.mean()
        fa = torch.fft.fft2(ref)
        fb = torch.fft.fft2(img)
        cross = fb * torch.conj(fa)
        from planetrecon.pipeline.align import correlation_filter, correlation_peak
        shape = tuple(ref.shape)
        if getattr(self, '_correlation_shape', None) != shape:
            self._correlation_filter = torch.as_tensor(correlation_filter(shape), device='cuda:0')
            self._correlation_shape = shape
        corr = torch.fft.ifft2(cross * self._correlation_filter).real
        return correlation_peak(corr.cpu().numpy())

    def shift(self, image: np.ndarray, shift_xy: tuple[float, float]) -> np.ndarray:
        from scipy.ndimage import shift as ndshift

        # Spatial bilinear shift stays on CPU; FFT correlation is the GPU cost.
        sy, sx = float(shift_xy[1]), float(shift_xy[0])
        return ndshift(np.asarray(image, dtype=np.float64), shift=(sy, sx), order=1, prefilter=False)

    @staticmethod
    def _regions(h, w, dy, dx):
        y0,y1=max(0,-dy),min(h,h-dy)
        x0,x1=max(0,-dx),min(w,w-dx)
        if y1<=y0 or x1<=x0:return None
        return (slice(y0,y1),slice(x0,x1)),(slice(y0+dy,y1+dy),slice(x0+dx,x1+dx))

    @classmethod
    def _pull(cls, image, shift_xy):
        """Exact pixel-coordinate bilinear pull with zero exterior support."""
        import math
        sx,sy=map(float,shift_xy); ix,iy=math.floor(sx),math.floor(sy)
        fx,fy=sx-ix,sy-iy
        result=torch.zeros_like(image)
        for dy,wy in ((iy,1-fy),(iy+1,fy)):
            for dx,wx in ((ix,1-fx),(ix+1,fx)):
                if wx*wy==0:continue
                regions=cls._regions(*image.shape[:2],dy,dx)
                if regions is not None:
                    dst,src=regions; result[dst]+=image[src]*(wx*wy)
        return result

    @classmethod
    def _neighbors(cls, image):
        result=torch.zeros_like(image)
        for dy in (-1,0,1):
            for dx in (-1,0,1):
                if dy==dx==0:continue
                regions=cls._regions(*image.shape[:2],dy,dx)
                if regions is not None:
                    dst,src=regions;result[dst]+=image[src]
        return result

    def backproject(self, raw, shift_xy, color):
        """CUDA mono/RGB/CFA contributions; CPU retains authoritative sums."""
        from planetrecon.detector import BAYER_PATTERNS
        image=torch.as_tensor(raw,device='cuda:0',dtype=torch.float64)
        if color in BAYER_PATTERNS:
            key=(tuple(raw.shape),color)
            if getattr(self,'_mask_key',None)!=key:
                tile=torch.tensor([["RGB".index(c) for c in row] for row in BAYER_PATTERNS[color]],device='cuda:0')
                y=torch.arange(raw.shape[0],device='cuda:0')[:,None]%2
                x=torch.arange(raw.shape[1],device='cuda:0')[None,:]%2
                self._masks=tile[y,x][...,None]==torch.arange(3,device='cuda:0')
                self._mask_weights=self._masks.to(torch.float64)
                self._demo_weights=torch.clamp(self._neighbors(self._mask_weights),min=1.)
                self._mask_key=key
            planes=image[...,None]*self._mask_weights
            demo=torch.where(self._masks,planes,self._neighbors(planes)/self._demo_weights)
            add=self._pull(planes,shift_xy);weight=self._pull(self._mask_weights,shift_xy)
            demo=self._pull(demo,shift_xy)
            support=self._pull(torch.ones_like(image),shift_xy)
            return tuple(v.cpu().numpy() for v in (add,weight,demo,support))
        add=self._pull(image,shift_xy)
        support=self._pull(torch.ones_like(image),shift_xy)
        return add.cpu().numpy(),support.cpu().numpy(),None,None
