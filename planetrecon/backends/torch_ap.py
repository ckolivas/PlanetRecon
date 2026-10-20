"""CUDA alignment points: grouped correlations, peak fits and field solve.

Mirrors ``pipeline.ap_align.AlignmentPoints`` in float64; the CPU matcher
remains the reference implementation. The field stays on the device.
"""
import numpy as np
import torch
import torch.nn.functional as F

from planetrecon.backends.torch_circular import sample
from planetrecon.pipeline import ap_align


def _gaussian(image, sigma):
    """SciPy ``gaussian_filter`` (mode 'reflect', truncate 4) on the device."""
    radius = int(4. * sigma + .5)
    x = torch.arange(-radius, radius + 1, device=image.device, dtype=torch.float64)
    kernel = torch.exp(-.5 * (x / sigma) ** 2)
    kernel = kernel / kernel.sum()
    out = image
    for axis in (0, 1):
        n = out.shape[axis]
        # Symmetric (edge-inclusive) reflection, repeated for kernels longer than the axis.
        index = torch.arange(-radius, n + radius, device=image.device)
        period = 2 * n
        index = torch.remainder(index, period)
        index = torch.where(index >= n, period - 1 - index, index)
        padded = out.index_select(axis, index)
        moved = padded.movedim(axis, -1)[None, None] if axis == 1 else padded.T[None, None]
        filtered = F.conv1d(moved.reshape(-1, 1, moved.shape[-1]), kernel[None, None]).reshape(
            moved.shape[2], -1)
        out = filtered if axis == 1 else filtered.T
    return out


def gaussian_proxy(image):
    """Device equivalent of ``LocalRegistration.proxy``."""
    a = _gaussian(image, 1.5)
    return a - _gaussian(a, 8.0)


def local_quality(plane, shift, centres, sigma, device='cuda:0'):
    """Device equivalent of ``ap_align.LocalQuality.measure``."""
    image = torch.as_tensor(plane, device=device, dtype=torch.float64)
    h, w = image.shape
    if (h, w) != getattr(local_quality, 'shape', None):
        local_quality.ys = torch.arange(h, device=device, dtype=torch.float64)
        local_quality.xs = torch.arange(w, device=device, dtype=torch.float64)
        local_quality.shape = (h, w)
    # SciPy mode 'nearest' at the detector edge.
    radius = int(4. * ap_align.QUALITY_SMOOTH_PX + .5)
    x = torch.arange(-radius, radius + 1, device=device, dtype=torch.float64)
    kernel = torch.exp(-.5 * (x / ap_align.QUALITY_SMOOTH_PX) ** 2)
    kernel = (kernel / kernel.sum())
    padded = F.pad(image[None, None], (radius, radius, radius, radius), mode='replicate')
    smooth = F.conv2d(F.conv2d(padded, kernel[None, None, None, :]), kernel[None, None, :, None])[0, 0]
    lap = torch.zeros_like(smooth)
    lap[1:-1, 1:-1] = (smooth[:-2, 1:-1] + smooth[2:, 1:-1] + smooth[1:-1, :-2]
                       + smooth[1:-1, 2:] - 4 * smooth[1:-1, 1:-1])
    c = torch.as_tensor(centres, device=device, dtype=torch.float64)
    wy = torch.exp(-.5 * ((local_quality.ys[None, :] - c[:, 0:1] - float(shift[1])) / sigma) ** 2)
    wx = torch.exp(-.5 * ((local_quality.xs[None, :] - c[:, 1:2] - float(shift[0])) / sigma) ** 2)
    energy = ((wy @ (lap * lap)) * wx).sum(dim=1)
    level = ((wy @ smooth) * wx).sum(dim=1)
    norm = wy.sum(1) * wx.sum(1)
    return ((energy / norm) / ((level / norm) ** 2).clamp_min(1e-30)).cpu().numpy()


class TorchAlignmentPoints:
    def __init__(self, matcher, *, device='cuda:0'):
        self.device = device
        self.shape = matcher.shape
        self.size, self.search = matcher.size, matcher.search
        self.centres = matcher.centres
        tensor = lambda a: torch.as_tensor(np.ascontiguousarray(a), device=device, dtype=torch.float64)
        self.window = tensor(matcher.window)
        self.weighted = tensor(matcher.weighted)
        self.norm = tensor(matcher.norm).clamp_min(1e-15)
        self.wy, self.wx = tensor(matcher.wy), tensor(matcher.wx)
        h, w = self.shape
        self.yy, self.xx = torch.meshgrid(torch.arange(h, device=device, dtype=torch.float64),
                                          torch.arange(w, device=device, dtype=torch.float64), indexing='ij')
        half, R = self.size // 2, self.search
        span = self.size + 2 * R
        ar = torch.arange(span, device=device)
        iy = torch.as_tensor(self.centres[:, 0] - half - R, device=device)
        ix = torch.as_tensor(self.centres[:, 1] - half - R, device=device)
        self.rows = (iy[:, None] + ar[None, :])[:, :, None]
        self.cols = (ix[:, None] + ar[None, :])[:, None, :]
        self.matcher = matcher

    def _ncc(self, image):
        n = len(self.centres)
        tiles = image[self.rows, self.cols][None]
        window = self.window[None, None].expand(n, 1, self.size, self.size)
        num = F.conv2d(tiles, self.weighted[:, None], groups=n)[0]
        s1 = F.conv2d(tiles, window, groups=n)[0]
        s2 = F.conv2d(tiles * tiles, window, groups=n)[0]
        var = (s2 - s1 * s1 / self.window.sum()).clamp_min(1e-12)
        return num / (var.sqrt() * self.norm[:, None, None])

    def displacement(self, frame, global_shift):
        gx, gy = float(global_shift[0]), float(global_shift[1])
        if not len(self.centres):
            return torch.stack([torch.full(self.shape, gx, device=self.device, dtype=torch.float64),
                                torch.full(self.shape, gy, device=self.device, dtype=torch.float64)])
        proxy = gaussian_proxy(torch.as_tensor(frame, device=self.device, dtype=torch.float64))
        image = sample(proxy, self.yy + gy, self.xx + gx)
        # Peak decisions are tiny (n x 13 x 13); reuse the CPU reference logic.
        offset, H, ok, _ = ap_align._peak_fit(self._ncc(image).cpu().numpy())
        self.matcher.accepted.append(float(ok.mean()))
        H_t = torch.as_tensor(H, device=self.device, dtype=torch.float64)
        d = torch.as_tensor(offset, device=self.device, dtype=torch.float64)
        Hd = torch.einsum('nij,nj->ni', H_t, d)
        A = torch.einsum('ky,kx,kij->yxij', self.wy, self.wx, H_t)
        b = torch.einsum('ky,kx,ki->yxi', self.wy, self.wx, Hd)
        trace = (H[:, 0, 0] + H[:, 1, 1])[ok]
        lam = ap_align.PRIOR * (float(np.median(trace)) if len(trace) else 1.)
        A[..., 0, 0] += lam
        A[..., 1, 1] += lam
        det = A[..., 0, 0] * A[..., 1, 1] - A[..., 0, 1] * A[..., 1, 0]
        u = torch.stack([(A[..., 1, 1] * b[..., 0] - A[..., 0, 1] * b[..., 1]) / det,
                         (A[..., 0, 0] * b[..., 1] - A[..., 1, 0] * b[..., 0]) / det])
        # Coarse samples include both detector edges, so align_corners maps exactly.
        field = F.interpolate(u[None], size=self.shape, mode='bilinear', align_corners=True)[0]
        field[0] += gx
        field[1] += gy
        return field
