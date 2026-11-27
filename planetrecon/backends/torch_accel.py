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
        snapshot.flags.writeable = False
        self._phase_reference, self._reference_terms = snapshot, self._terms(snapshot)
        return snapshot

    def _terms(self, reference):
        """(float32 half spectrum, float32 filter, float64 filtered reference) of a correlation reference.

        The filtered reference is the zero-mean reference convolved with the registration filter, so
        a frame's filtered correlation at a lag is its dot product with the frame shifted by that lag.
        """
        if reference is getattr(self, '_phase_reference', None):
            return self._reference_terms
        from planetrecon.pipeline.align import correlation_filter
        ref = torch.tensor(np.asarray(reference, dtype=np.float64), device='cuda:0')
        ref = ref - ref.mean()
        shape = tuple(ref.shape)
        # The filter depends on |frequency| only; the half spectrum keeps columns 0..w/2.
        filt = torch.as_tensor(correlation_filter(shape)[:, :shape[1]//2+1], device='cuda:0')
        spectrum = torch.fft.rfft2(ref)
        filtered = torch.fft.irfft2(spectrum * filt, s=shape)
        return torch.conj(spectrum).to(torch.complex64), filt.to(torch.float32), filtered

    def phase_correlation(self, reference: np.ndarray, frame: np.ndarray) -> tuple[float, float]:
        convert = (torch.tensor if isinstance(frame, np.ndarray) and not frame.flags.writeable
                   else torch.as_tensor)
        return self.phase_correlations(reference, convert(frame, device='cuda:0', dtype=torch.float64)[None])[0]

    def phase_correlations(self, reference: np.ndarray, frames) -> list[tuple[float, float]]:
        """Displacements (x, y) of frames (B, h, w) against ``reference``, with one or two copies back.

        A float32 correlation surface locates each integer peak; the 3 x 3 correlations around it
        are then formed exactly in float64 (dot products with the filtered reference), climbing to
        the neighbouring lag while one is higher, and fitted as before. The displacements are those
        of the float64 surface (to about 1e-12 px), at float32 transform cost.
        """
        from planetrecon.pipeline.align import peak_offsets
        img = torch.as_tensor(frames, device='cuda:0', dtype=torch.float64)
        img = img - img.mean((-2, -1), keepdim=True)
        spectrum, filt, filtered = self._terms(reference)
        shape = tuple(img.shape[-2:])
        coarse = torch.fft.irfft2(torch.fft.rfft2(img.to(torch.float32)) * spectrum * filt, s=shape)
        flat = torch.argmax(coarse.reshape(len(img), -1), 1)
        peaks = torch.stack([flat // shape[1], flat % shape[1]], 1)
        # One copy back: the integer peaks with their 3 x 3 correlations.
        host = torch.cat([peaks.to(torch.float64), self._stencil(img, filtered, peaks).reshape(-1, 9)], 1).cpu().numpy()
        peaks, values = host[:, :2].astype(np.int64), host[:, 2:].reshape(-1, 3, 3)
        for _ in range(16):
            # Move any frame whose exact peak is a neighbouring lag, and measure there again.
            best = values.reshape(len(values), 9).argmax(1)
            moving = np.flatnonzero(best != 4)
            if not len(moving):
                break
            peaks[moving] = (peaks[moving] + np.stack([best[moving] // 3 - 1, best[moving] % 3 - 1], 1)) % shape
            values[moving] = self._stencil(img[moving], filtered,
                                           torch.as_tensor(peaks[moving], device='cuda:0')).cpu().numpy()
        return [peak_offsets(int(py), int(px), shape, v[:, 1], v[1, :]) for (py, px), v in zip(peaks, values)]

    @staticmethod
    def _stencil(img, filtered, peaks):
        from planetrecon.backends.torch_ap import fused_kernels
        kernels = fused_kernels()
        if kernels is not None:
            return kernels.stencil(img, filtered, peaks)
        out = torch.empty((len(img), 3, 3), dtype=torch.float64, device=img.device)
        for b, (py, px) in enumerate(peaks.tolist()):
            for i in range(3):
                for j in range(3):
                    out[b, i, j] = (torch.roll(img[b], (-(py + i - 1), -(px + j - 1)), (0, 1)) * filtered).sum()
        return out

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

    def registration_planes(self, images, color):
        """``registration_plane`` of frames (B, h, w[, 3]) of any numeric dtype: (B, h, w) float64."""
        from planetrecon.detector import BAYER_PATTERNS
        images=torch.as_tensor(images,device='cuda:0')
        if color in BAYER_PATTERNS:
            from planetrecon.backends.torch_ap import fused_kernels
            kernels=fused_kernels()
            if kernels is not None and images.numel()<2**31:
                # One fused pass; the same sums and averages as the demosaic below.
                return kernels.luminance(images,[["RGB".index(c) for c in row] for row in BAYER_PATTERNS[color]])
        return torch.stack([self.registration_plane(image,color) for image in images])

    def registration_plane(self, raw, color):
        """Device equivalent of the colour registration plane (bilinear RGB luminance)."""
        from planetrecon.detector import BAYER_PATTERNS
        image=torch.as_tensor(raw,device='cuda:0')
        if color in BAYER_PATTERNS and getattr(self,'_fused_planes',True):
            from planetrecon.backends.torch_ap import fused_kernels
            if fused_kernels() is not None and image.numel()<2**31:
                return self.registration_planes(image[None],color)[0]
            self._fused_planes=False
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
