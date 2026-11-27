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

    def prepare_reference(self, reference: np.ndarray) -> np.ndarray:
        snapshot = np.array(reference, dtype=np.float64, copy=True)
        ref = torch.tensor(snapshot, device='cuda:0', dtype=torch.float64)
        spectrum = torch.conj(torch.fft.rfft2(ref-ref.mean()))
        snapshot.flags.writeable = False
        self._phase_reference, self._reference_spectrum = snapshot, spectrum
        return snapshot

    def _correlation(self, reference, img):
        """Filtered cross-correlation of zero-mean frames (..., h, w) with ``reference``.

        The frames are real, so half spectra (rfft2) carry the same correlation.
        """
        if reference is getattr(self, '_phase_reference', None):
            spectrum = self._reference_spectrum
        else:
            ref = torch.tensor(reference, device='cuda:0', dtype=torch.float64)
            spectrum = torch.conj(torch.fft.rfft2(ref-ref.mean()))
        from planetrecon.pipeline.align import correlation_filter
        shape = tuple(reference.shape)
        if getattr(self, '_correlation_shape', None) != shape:
            # The filter depends on |frequency| only; the half spectrum keeps columns 0..w/2.
            self._correlation_filter = torch.as_tensor(correlation_filter(shape)[:, :shape[1]//2+1], device='cuda:0')
            self._correlation_shape = shape
        return torch.fft.irfft2(torch.fft.rfft2(img) * spectrum * self._correlation_filter, s=shape)

    def phase_correlation(self, reference: np.ndarray, frame: np.ndarray) -> tuple[float, float]:
        convert = (torch.tensor if isinstance(frame, np.ndarray) and not frame.flags.writeable
                   else torch.as_tensor)
        img = convert(frame, device="cuda:0", dtype=torch.float64)
        corr = self._correlation(reference, img - img.mean())
        # Locate the peak on the device; only the two three-point stencils are copied.
        from planetrecon.pipeline.align import peak_neighbours, peak_offsets
        shape = tuple(reference.shape)
        py, px = divmod(int(torch.argmax(corr)), shape[1])
        vertical, horizontal = peak_neighbours(py, px, shape)
        index = torch.as_tensor([vertical, horizontal], device=corr.device)
        values = corr[index[:, 0], index[:, 1]].cpu().numpy()
        return peak_offsets(py, px, shape, values[0], values[1])

    def phase_correlations(self, reference: np.ndarray, frames) -> list[tuple[float, float]]:
        """``phase_correlation`` of each of frames (B, h, w) on the device, with one copy back."""
        img = torch.as_tensor(frames, device='cuda:0', dtype=torch.float64)
        corr = self._correlation(reference, img - img.mean((-2, -1), keepdim=True))
        from planetrecon.pipeline.align import peak_neighbours, peak_offsets
        shape = tuple(reference.shape)
        peaks = torch.argmax(corr.reshape(len(corr), -1), 1).cpu().numpy()
        stencils = [peak_neighbours(*divmod(int(k), shape[1]), shape) for k in peaks]
        rows = torch.as_tensor([[*v[0], *hz[0]] for v, hz in stencils], device='cuda:0')
        cols = torch.as_tensor([[*v[1], *hz[1]] for v, hz in stencils], device='cuda:0')
        values = corr[torch.arange(len(corr), device='cuda:0')[:, None], rows, cols].cpu().numpy()
        return [peak_offsets(*divmod(int(k), shape[1]), shape, v[:3], v[3:]) for k, v in zip(peaks, values)]

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
        return cls._pull_many((image,), shift_xy)[0]

    # Stands for an all-ones (h, w) plane in ``_pull_many``: its pull is the support.
    SUPPORT = object()

    @classmethod
    def _pull_many(cls, images, shift_xy):
        """Share displacement upload and each corner across signal and support.

        ``images`` may include ``SUPPORT`` for the pulled support of an all-ones
        plane. Only one corner's indices and weights are alive at a time.
        """
        image = next(value for value in images if value is not cls.SUPPORT)
        h, w = image.shape[:2]
        if np.ndim(shift_xy[0]) or np.ndim(shift_xy[1]):
            sx = torch.as_tensor(shift_xy[0], device=image.device, dtype=torch.float64)
            sy = torch.as_tensor(shift_xy[1], device=image.device, dtype=torch.float64)
            if sx.shape != (h, w) or sy.shape != (h, w):
                raise ValueError('dense displacement must match the detector shape')
            if not bool(torch.isfinite(sx).all() & torch.isfinite(sy).all()):
                raise ValueError('dense displacement must be finite')
            y = torch.arange(h, device=image.device, dtype=torch.float64)[:, None] + sy
            x = torch.arange(w, device=image.device, dtype=torch.float64)[None, :] + sx
            iy, ix = torch.floor(y).long(), torch.floor(x).long()
            fy, fx = y-iy, x-ix
            results = [torch.zeros((h, w), dtype=torch.float64, device=image.device) if value is cls.SUPPORT
                       else torch.zeros_like(value) for value in images]
            for dy, wy in ((0, 1-fy), (1, fy)):
                for dx, wx in ((0, 1-fx), (1, fx)):
                    yy, xx = iy+dy, ix+dx
                    weight = wy*wx*((yy >= 0) & (yy < h) & (xx >= 0) & (xx < w))
                    yy, xx = yy.clamp(0, h-1), xx.clamp(0, w-1)
                    for value, result in zip(images, results):
                        if value is cls.SUPPORT:
                            result += weight  # The pull of an all-ones plane.
                        else:
                            result += value[yy, xx]*(weight[..., None] if value.ndim == 3 else weight)
            return results
        images = [torch.ones((h, w), dtype=torch.float64, device=image.device) if value is cls.SUPPORT else value
                  for value in images]
        results = [torch.zeros_like(value) for value in images]
        import math
        sx,sy=map(float,shift_xy); ix,iy=math.floor(sx),math.floor(sy)
        fx,fy=sx-ix,sy-iy
        for dy,wy in ((iy,1-fy),(iy+1,fy)):
            for dx,wx in ((ix,1-fx),(ix+1,fx)):
                if wx*wy==0:continue
                regions=cls._regions(*image.shape[:2],dy,dx)
                if regions is not None:
                    dst,src=regions
                    for value,result in zip(images,results):
                        result[dst]+=value[src]*(wx*wy)
        return results

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

    def _bayer(self, shape, color):
        from planetrecon.detector import BAYER_PATTERNS
        key=(tuple(shape),color)
        if getattr(self,'_mask_key',None)!=key:
            tile=torch.tensor([["RGB".index(c) for c in row] for row in BAYER_PATTERNS[color]],device='cuda:0')
            y=torch.arange(shape[0],device='cuda:0')[:,None]%2
            x=torch.arange(shape[1],device='cuda:0')[None,:]%2
            self._masks=tile[y,x][...,None]==torch.arange(3,device='cuda:0')
            self._mask_weights=self._masks.to(torch.float64)
            self._demo_weights=torch.clamp(self._neighbors(self._mask_weights),min=1.)
            self._mask_key=key

    def _demosaic(self, image, color):
        self._bayer(image.shape, color)
        planes=image[...,None]*self._mask_weights
        return planes,torch.where(self._masks,planes,self._neighbors(planes)/self._demo_weights)

    def registration_plane(self, raw, color):
        """Device equivalent of the colour registration plane (bilinear RGB luminance)."""
        from planetrecon.detector import BAYER_PATTERNS
        image=torch.as_tensor(raw,device='cuda:0')
        image=image if image.dtype==torch.float64 else image.to(torch.float64)
        if color in BAYER_PATTERNS:
            rgb=self._demosaic(image,color)[1]
            # Elementwise, not a float64 matrix-vector product (slow on consumer GPUs); same sum.
            return .25*rgb[...,0]+.5*rgb[...,1]+.25*rgb[...,2]
        if image.ndim==3:
            return .5*image[...,1]+.25*image[...,0]+.25*image[...,2]
        return image

    def backproject_tensors(self, raw, shift_xy, color):
        """Device contributions (signal, weight, None, support); Bayer CFA sites give their own weight."""
        from planetrecon.detector import BAYER_PATTERNS
        image=torch.as_tensor(raw,device='cuda:0')
        image=image if image.dtype==torch.float64 else image.to(torch.float64)
        if color in BAYER_PATTERNS:
            self._bayer(image.shape,color)
            return (*self._pull_many((image[...,None]*self._mask_weights,self._mask_weights),shift_xy),None,None)
        add,support=self._pull_many((image,self.SUPPORT if image.ndim==2 else torch.ones_like(image)),shift_xy)
        return add,support,None,None

    def backproject_group(self, images, fields, color):
        """Contributions of frames (B, h, w[, 3]) pulled by dense fields (B, 2, h, w): (B, h, w, 2C)
        float64, the C signal channels then their C weights (CFA sites for Bayer, else the support).

        Per sample the same products and corner order as ``backproject_tensors``.
        """
        from planetrecon.detector import BAYER_PATTERNS
        images=torch.as_tensor(images,device='cuda:0')
        images=images if images.dtype==torch.float64 else images.to(torch.float64)
        B,h,w=images.shape[:3]
        fields=torch.as_tensor(fields,device='cuda:0',dtype=torch.float64)
        if fields.shape!=(B,2,h,w):
            raise ValueError('dense displacement must match the detector shape')
        if not bool(torch.isfinite(fields).all()):
            raise ValueError('dense displacement must be finite')
        bayer=color in BAYER_PATTERNS
        if bayer:
            self._bayer((h,w),color)
            masks=self._mask_weights
            values=torch.cat((images[...,None]*masks,masks.expand(B,h,w,3)),-1)
        else:
            values=images if images.ndim==4 else images[...,None]
        y=torch.arange(h,device='cuda:0',dtype=torch.float64)[None,:,None]+fields[:,1]
        x=torch.arange(w,device='cuda:0',dtype=torch.float64)[None,None,:]+fields[:,0]
        iy,ix=torch.floor(y).long(),torch.floor(x).long()
        fy,fx=y-iy,x-ix
        frame=torch.arange(B,device='cuda:0')[:,None,None]
        total=torch.zeros(values.shape,dtype=torch.float64,device='cuda:0')
        support=None if bayer else torch.zeros((B,h,w),dtype=torch.float64,device='cuda:0')
        for dy,wy in ((0,1-fy),(1,fy)):
            for dx,wx in ((0,1-fx),(1,fx)):
                yy,xx=iy+dy,ix+dx
                weight=wy*wx*((yy>=0)&(yy<h)&(xx>=0)&(xx<w))
                yy,xx=yy.clamp(0,h-1),xx.clamp(0,w-1)
                total+=values[frame,yy,xx]*weight[...,None]
                if support is not None:
                    support+=weight
        if bayer:
            return total
        return torch.cat((total,support[...,None].expand(*total.shape)),-1)

    def backproject(self, raw, shift_xy, color):
        """CUDA mono/RGB/CFA contributions; CPU retains authoritative sums."""
        return tuple(None if v is None else v.cpu().numpy() for v in self.backproject_tensors(raw, shift_xy, color))
