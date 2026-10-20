"""Per-AP local quality and blended per-pixel frame weights (AutoStakkert-style)."""
import numpy as np
import torch
import torch.nn.functional as F


def gaussian_kernel(sigma, device):
    r = int(np.ceil(3*sigma))
    x = torch.arange(-r, r+1, device=device, dtype=torch.float64)
    k = torch.exp(-.5*(x/sigma)**2)
    return k/k.sum()


def blur(img, sigma):
    k = gaussian_kernel(sigma, img.device)
    r = len(k)//2
    x = F.pad(img[None, None], (r, r, r, r), mode='replicate')
    x = F.conv2d(x, k[None, None, None, :])
    x = F.conv2d(x, k[None, None, :, None])
    return x[0, 0]


class LocalQuality:
    """Squared Laplacian of a smoothed plane, Gaussian-averaged around each AP."""
    def __init__(self, centres, shape, *, ap_size=48, smooth=1.5, device='cuda:0'):
        self.centres = np.asarray(centres)
        self.device = device
        self.smooth = smooth
        h, w = shape
        ys = torch.arange(h, device=device, dtype=torch.float64)
        xs = torch.arange(w, device=device, dtype=torch.float64)
        self.ys, self.xs = ys, xs
        self.sigma = ap_size/4
        self.lap = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], device=device, dtype=torch.float64)[None, None]

    def measure(self, plane, shift):
        p = torch.as_tensor(plane, device=self.device, dtype=torch.float64)
        s = blur(p, self.smooth)
        lap = F.conv2d(F.pad(s[None, None], (1, 1, 1, 1), mode='replicate'), self.lap)[0, 0]
        e = lap*lap
        cy = torch.as_tensor(self.centres[:, 0], device=self.device, dtype=torch.float64) + float(shift[1])
        cx = torch.as_tensor(self.centres[:, 1], device=self.device, dtype=torch.float64) + float(shift[0])
        wy = torch.exp(-.5*((self.ys[None, :]-cy[:, None])/self.sigma)**2)
        wx = torch.exp(-.5*((self.xs[None, :]-cx[:, None])/self.sigma)**2)
        num = torch.einsum('ky,yx,kx->k', wy, e, wx)
        den = torch.einsum('ky,kx->k', wy, wx.sum(1, keepdim=True).expand_as(wx)) / wx.shape[1]
        level = torch.einsum('ky,yx,kx->k', wy, s, wx)
        # Normalise by local brightness squared so transparency changes don't dominate.
        return (num/(level*level).clamp_min(1e-12)).cpu().numpy()


def blend_masks(centres, shape, spacing, device='cuda:0'):
    """Normalised Gaussian footprints per AP plus the leftover (outside APs) map."""
    h, w = shape
    ys = torch.arange(h, device=device, dtype=torch.float64)
    xs = torch.arange(w, device=device, dtype=torch.float64)
    sigma = 0.6*spacing
    c = torch.as_tensor(np.asarray(centres), device=device, dtype=torch.float64)
    wy = torch.exp(-.5*((ys[None, :]-c[:, 0:1])/sigma)**2)
    wx = torch.exp(-.5*((xs[None, :]-c[:, 1:2])/sigma)**2)
    M = torch.einsum('ky,kx->kyx', wy, wx)
    total = M.sum(0)
    coverage = (total/(total.max())).clamp(0, 1)
    # Inside the AP cloud the footprints sum to one; fade to global selection outside.
    cover = torch.clamp(total/0.5, max=1.)
    M = M/total.clamp_min(1e-12)*cover
    return M, 1-cover
