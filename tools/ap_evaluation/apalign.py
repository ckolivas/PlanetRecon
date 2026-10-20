"""AutoStakkert-style alignment points on CUDA.

Grid of square, Hann-tapered APs on textured planet regions of the template.
Each frame: global shift, then per-AP NCC over a +/-R integer search on the
band-passed proxy, 2-D quadratic subpixel peak; AP accepted if the peak is
interior and NCC >= min_ncc. A dense field is built by normalised Gaussian
interpolation of the accepted AP shifts (no shrinkage toward global motion).
"""
import numpy as np
import torch
import torch.nn.functional as F
from planetrecon.pipeline.local_align import LocalRegistration


class APRegistration:
    def __init__(self, reference, *, size=48, spacing=None, search=6, min_ncc=0.5, sigma=None,
                 min_brightness=0.1, device='cuda:0', tensor=True, prior=0.01):
        ref = np.asarray(reference, float)
        self.shape = ref.shape
        self.size, self.search, self.min_ncc = size, search, min_ncc
        self.spacing = spacing or size//2
        self.sigma = sigma or 0.6*self.spacing
        self.device = device
        self.tensor, self.prior = tensor, prior
        proxy = LocalRegistration.proxy(ref)
        h, w = ref.shape
        half, R = size//2, search
        ys = np.arange(half+R, h-half-R, self.spacing)
        xs = np.arange(half+R, w-half-R, self.spacing)
        win = np.outer(np.hanning(size), np.hanning(size))
        level = ref.max()*min_brightness
        centres, templates = [], []
        energies = []
        for y in ys:
            for x in xs:
                s = np.s_[y-half:y-half+size, x-half:x-half+size]
                if ref[s].mean() < level:
                    continue
                t = proxy[s]
                t = t - (t*win).sum()/win.sum()
                energies.append(float(np.sqrt((t*t*win).sum()/win.sum())))
                centres.append((y, x)); templates.append(t)
        energies = np.array(energies)
        keep = energies >= 0.1*np.median(energies)
        self.centres = np.array(centres)[keep]
        self.templates = torch.as_tensor(np.array(templates)[keep], device=device)
        self.win = torch.as_tensor(win, device=device)
        t = self.templates*self.win
        self.tnorm = torch.sqrt((self.templates**2*self.win).sum(dim=(1, 2)))
        self.twin = self.templates*self.win  # template weighted once
        yy, xx = torch.meshgrid(torch.arange(h, device=device, dtype=torch.float64),
                                torch.arange(w, device=device, dtype=torch.float64), indexing='ij')
        self.yy, self.xx = yy, xx
        cy = torch.as_tensor(self.centres[:, 0], device=device, dtype=torch.float64)
        cx = torch.as_tensor(self.centres[:, 1], device=device, dtype=torch.float64)
        # Gaussian interpolation kernels for each AP, evaluated on the full grid lazily in chunks.
        self.cy, self.cx = cy, cx
        self.stats = []

    def _ncc(self, image):
        """NCC surfaces (n_ap, 2R+1, 2R+1) of the pulled proxy against AP templates."""
        size, R = self.size, self.search
        half = size//2
        n = len(self.centres)
        span = size + 2*R
        iy = torch.as_tensor(self.centres[:, 0]-half-R, device=self.device)
        ix = torch.as_tensor(self.centres[:, 1]-half-R, device=self.device)
        ar = torch.arange(span, device=self.device)
        tiles = image[(iy[:, None]+ar[None, :])[:, :, None], (ix[:, None]+ar[None, :])[:, None, :]]  # n, span, span
        # Correlate with grouped conv2d: weighted template, and sliding weighted moments.
        x = tiles[None]                       # 1, n, span, span
        num = F.conv2d(x, self.twin[:, None], groups=n)[0]
        w = self.win[None, None].expand(n, 1, size, size)
        s1 = F.conv2d(x, w, groups=n)[0]
        s2 = F.conv2d(x*x, w, groups=n)[0]
        wsum = self.win.sum()
        var = (s2 - s1*s1/wsum).clamp_min(1e-12)
        # Template is zero-mean under the window, so the mean term in num vanishes.
        return num/(var.sqrt()*self.tnorm[:, None, None])

    def displacement_tensor(self, proxy_frame, global_shift):
        from planetrecon.backends.torch_circular import sample
        image0 = torch.as_tensor(LocalRegistration.proxy(proxy_frame), device=self.device)
        gx, gy = float(global_shift[0]), float(global_shift[1])
        image = sample(image0, self.yy+gy, self.xx+gx)
        c = self._ncc(image)
        n, m2, _ = c.shape
        R = self.search
        flat = c.flatten(1)
        peak, index = flat.max(dim=1)
        py, px = index//m2, index % m2
        interior = (px > 0) & (px < m2-1) & (py > 0) & (py < m2-1)
        cy_, cx_ = py.clamp(1, m2-2), px.clamp(1, m2-2)
        rows = torch.arange(n, device=self.device)
        at = lambda dy, dx: c[rows, cy_+dy, cx_+dx]
        p0 = at(0, 0)
        gxx, gyy = .5*(at(0, 1)-at(0, -1)), .5*(at(1, 0)-at(-1, 0))
        hxx, hyy = at(0, 1)-2*p0+at(0, -1), at(1, 0)-2*p0+at(-1, 0)
        hxy = .25*(at(1, 1)-at(1, -1)-at(-1, 1)+at(-1, -1))
        # Information matrix of each AP from the NCC peak curvature. Edges (rings,
        # limb) constrain only the perpendicular direction; clamp the flat one.
        H = -torch.stack([torch.stack([hxx, hxy], -1), torch.stack([hxy, hyy], -1)], -2)
        evals, evecs = torch.linalg.eigh(H)
        evals = evals.clamp_min(0)
        ok = interior & (peak >= self.min_ncc) & (evals[:, 1] > 0)
        inv = torch.where(evals > 1e-6*evals[:, 1:2].clamp_min(1e-30), 1/evals.clamp_min(1e-30), torch.zeros_like(evals))
        g = torch.stack([gxx, gyy], -1)
        # Newton step on the constrained subspace only.
        coeff = torch.einsum('nij,ni->nj', evecs, g)*inv
        delta = torch.einsum('nij,nj->ni', evecs, coeff)
        ok &= (delta.abs() <= 1).all(dim=1)
        H = torch.einsum('nij,nj,nkj->nik', evecs, evals, evecs)*ok[:, None, None]
        sx = (cx_-R).double()+delta[:, 0]
        sy = (cy_-R).double()+delta[:, 1]
        self.stats.append((float(ok.float().mean()), float(peak.median())))
        if self.tensor:
            return self._tensor_field(sx, sy, H, ok, gx, gy)
        return self._field(sx, sy, ok, gx, gy)

    def _tensor_field(self, sx, sy, H, ok, gx, gy):
        """Per-pixel weighted least squares: sum_k w_k H_k (u - d_k) + lam (u - g) = 0."""
        h, w = self.shape
        A = torch.zeros((h, w, 2, 2), device=self.device, dtype=torch.float64)
        b = torch.zeros((h, w, 2), device=self.device, dtype=torch.float64)
        ys = torch.arange(h, device=self.device, dtype=torch.float64)
        xs = torch.arange(w, device=self.device, dtype=torch.float64)
        d = torch.stack([sx, sy], -1)
        Hd = torch.einsum('nij,nj->ni', H, d)
        for start in range(0, len(sx), 32):
            sl = slice(start, start+32)
            wy = torch.exp(-.5*((ys[None, :]-self.cy[sl, None])/self.sigma)**2)
            wx = torch.exp(-.5*((xs[None, :]-self.cx[sl, None])/self.sigma)**2)
            wk = torch.einsum('ky,kx->kyx', wy, wx)
            A += torch.einsum('kyx,kij->yxij', wk, H[sl])
            b += torch.einsum('kyx,ki->yxi', wk, Hd[sl])
        trace = (H[:, 0, 0]+H[:, 1, 1])[ok]
        lam = self.prior*(trace.median() if len(trace) else torch.tensor(1., device=self.device, dtype=torch.float64))
        A[..., 0, 0] += lam; A[..., 1, 1] += lam
        u = torch.linalg.solve(A, b[..., None])[..., 0]     # residual relative to global
        return torch.stack([gx + u[..., 0], gy + u[..., 1]])

    def _field(self, sx, sy, ok, gx, gy):
        h, w = self.shape
        num = torch.zeros((2, h, w), device=self.device, dtype=torch.float64)
        den = torch.zeros((h, w), device=self.device, dtype=torch.float64)
        okf = ok.double()
        # Separable Gaussian weights per AP: evaluate in chunks to bound memory.
        ys = torch.arange(h, device=self.device, dtype=torch.float64)
        xs = torch.arange(w, device=self.device, dtype=torch.float64)
        for start in range(0, len(sx), 64):
            sl = slice(start, start+64)
            wy = torch.exp(-.5*((ys[None, :]-self.cy[sl, None])/self.sigma)**2)
            wx = torch.exp(-.5*((xs[None, :]-self.cx[sl, None])/self.sigma)**2)
            wk = torch.einsum('ky,kx->kyx', wy, wx)*okf[sl, None, None]
            den += wk.sum(0)
            num[0] += (wk*sx[sl, None, None]).sum(0)
            num[1] += (wk*sy[sl, None, None]).sum(0)
        # Far from any accepted AP the field returns to the global shift.
        floor = 1e-3
        field_x = gx + num[0]/(den+floor)
        field_y = gy + num[1]/(den+floor)
        return torch.stack([field_x, field_y])
