"""Resident float64 circular alignment with batched forward/reverse matching.

No per-patch host transfers or host-side peak decisions. The CPU matcher remains
an independent reference. Temporary candidate storage is bounded per chunk.
"""
import numpy as np
import torch
import torch.nn.functional as F


def sample(image, y, x, *, nearest=False):
    """Pixel-coordinate bilinear sampling, matching SciPy's order-one pull."""
    h, w = image.shape[-2:]
    if nearest:
        y, x = y.clamp(0, h-1), x.clamp(0, w-1)
    iy, ix = torch.floor(y).long(), torch.floor(x).long()
    fy, fx = y-iy, x-ix
    result = torch.zeros((*image.shape[:-2], *torch.broadcast_shapes(y.shape, x.shape)),
                         dtype=image.dtype, device=image.device)
    for dy, wy in ((0, 1-fy), (1, fy)):
        for dx, wx in ((0, 1-fx), (1, fx)):
            sy, sx = iy+dy, ix+dx
            weight = wy*wx*((sy >= 0) & (sy < h) & (sx >= 0) & (sx < w))
            result += image[..., sy.clamp(0, h-1), sx.clamp(0, w-1)]*weight
    return result


def scores(tiles, templates, strength, weight):
    w = weight.shape[0]
    candidates = tiles.unfold(1, w, 1).unfold(2, w, 1)
    mean = (candidates*weight).sum(dim=(-1, -2))
    centred = candidates-mean[..., None, None]
    variance = (centred.square()*weight).sum(dim=(-1, -2))
    numerator = (centred*templates[:, None, None]*weight).sum(dim=(-1, -2))
    return numerator/(variance.sqrt()*strength[:, None, None]).clamp_min(1e-15)


def peaks(costs):
    """Vectorised equivalent of circular_align._peak; invalid rows stay masked."""
    b, n, _ = costs.shape
    m = n//2
    flat = costs.flatten(1)
    peak, index = flat.max(dim=1)
    y, x = index//n, index % n
    valid = torch.isfinite(flat).all(dim=1) & (peak >= .8) & (x > 0) & (x < 2*m) & (y > 0) & (y < 2*m)
    yy = torch.arange(n, device=costs.device)[None, :, None]
    xx = torch.arange(n, device=costs.device)[None, None, :]
    distant = (xx-x[:, None, None]).square()+(yy-y[:, None, None]).square() >= 4
    competitor = costs.masked_fill(~distant, -torch.inf).flatten(1).max(dim=1).values
    valid &= peak-competitor >= .005
    cy, cx = y.clamp(1, n-2), x.clamp(1, n-2)
    rows = torch.arange(b, device=costs.device)
    def at(dy, dx):
        return costs[rows, cy+dy, cx+dx]
    centre = at(0, 0)
    gx, gy = .5*(at(0, 1)-at(0, -1)), .5*(at(1, 0)-at(-1, 0))
    a, c = 2*centre-at(0, -1)-at(0, 1), 2*centre-at(-1, 0)-at(1, 0)
    cross = -.25*(at(1, 1)-at(1, -1)-at(-1, 1)+at(-1, -1))
    eigen = .5*(a+c-torch.sqrt((a-c).square()+4*cross.square()))
    valid &= eigen > .001
    det = torch.where(valid, a*c-cross.square(), torch.ones_like(a))
    delta = torch.stack(((c*gx-cross*gy)/det, (a*gy-cross*gx)/det), dim=1)
    delta = torch.where((peak >= 1-1e-10)[:, None], torch.zeros_like(delta), delta)
    valid &= torch.isfinite(delta).all(dim=1) & (delta.abs() <= 1).all(dim=1)
    offset = torch.stack((x-m, y-m), dim=1)+delta
    return torch.where(valid[:, None], offset, torch.zeros_like(offset)), valid, peak


class TorchCircularRegistration:
    def __init__(self, matcher, *, device='cuda:0', chunk_bytes=64*1024**2):
        self.device = device
        self.fused_scores = None
        if str(device).startswith('cuda'):
            try:
                from planetrecon.backends.triton_correlation import scores as fused_scores
                self.fused_scores = fused_scores
            except (ImportError, OSError, RuntimeError):
                pass  # PyTorch-only builds retain the bounded resident path.
        self.shape = matcher.shape
        self.yy, self.xx = torch.meshgrid(
            torch.arange(self.shape[0], device=device, dtype=torch.float64),
            torch.arange(self.shape[1], device=device, dtype=torch.float64), indexing='ij')
        self.layers = []
        self.reference = None
        for layer in matcher.layers:
            if self.reference is None:
                self.reference = torch.as_tensor(layer.ref, device=device, dtype=torch.float64)
            active = np.flatnonzero(np.asarray(layer.texture_valid)
                                    & (layer.strength >= max(layer.strength.max()*.08, 1e-12)))
            if not len(active):
                continue
            y = torch.as_tensor(layer.ys[active//len(layer.xs)], device=device)
            x = torch.as_tensor(layer.xs[active % len(layer.xs)], device=device)
            w, h, m = layer.window, layer.window//2, layer.max_shift
            py, px = torch.meshgrid(torch.arange(-h, h+1, device=device),
                                    torch.arange(-h, h+1, device=device), indexing='ij')
            ty, tx = torch.meshgrid(torch.arange(-h-m, h+m+1, device=device),
                                    torch.arange(-h-m, h+m+1, device=device), indexing='ij')
            # Prepare on CPU to retain the exact reference taper coefficients.
            cy, cx = np.indices((w, w))-h
            footprint = .5*(1+np.cos(np.pi*np.clip(2*np.hypot(cx, cy)/h-1, 0., 1.)))
            fused = self.fused_scores is not None and w <= 127
            self.layers.append(dict(y=y, x=x, py=py, px=px, ty=ty, tx=tx, h=h, m=m, fused=fused,
                templates=torch.as_tensor(layer.templates[active], device=device, dtype=torch.float64),
                strength=torch.as_tensor(layer.strength[active], device=device, dtype=torch.float64),
                weight=torch.as_tensor(layer.weight, device=device, dtype=torch.float64),
                footprint=torch.as_tensor(footprint, device=device, dtype=torch.float64),
                chunk=max(1, min(1024 if fused else 128,
                                chunk_bytes//(w*w*8*(24 if fused else 49*5))))))

    def _scores(self, image, layer, section, templates, strength):
        cy, cx = layer['y'][section], layer['x'][section]
        if layer['fused']:
            try:
                return self.fused_scores(image, cy, cx, templates, strength, layer['weight'])
            except Exception as exc:
                raise RuntimeError(f'CUDA fused correlation failed: {exc}') from exc
        tiles = image[cy[:, None, None]+layer['ty'], cx[:, None, None]+layer['tx']]
        return scores(tiles, templates, strength, layer['weight'])

    def displacement(self, proxy, global_shift, should_cancel):
        image0 = torch.as_tensor(proxy, device=self.device, dtype=torch.float64)
        origin = torch.as_tensor(global_shift, device=self.device, dtype=torch.float64)[:, None, None]
        field = origin.expand(2, *self.shape).clone()
        for layer in reversed(self.layers):
            should_cancel()
            image = sample(image0, self.yy+field[1], self.xx+field[0])
            observed = ((self.xx+field[0] >= 0) & (self.xx+field[0] <= self.shape[1]-1)
                        & (self.yy+field[1] >= 0) & (self.yy+field[1] <= self.shape[0]-1))
            integral = F.pad((~observed).long().cumsum(0).cumsum(1), (1, 0, 1, 0))
            y, x = layer['y'], layer['x']
            margin = layer['h']+layer['m']
            supported = (integral[y+margin+1, x+margin+1]-integral[y-margin, x+margin+1]
                         -integral[y+margin+1, x-margin]+integral[y-margin, x-margin]) == 0
            numerator = torch.zeros_like(field)
            denominator = torch.zeros_like(image)
            for start in range(0, len(y), layer['chunk']):
                should_cancel()
                section = slice(start, start+layer['chunk'])
                cy, cx = y[section, None, None], x[section, None, None]
                costs = self._scores(image, layer, section, layer['templates'][section], layer['strength'][section])
                delta, valid, peak = peaks(costs)
                valid &= supported[section]
                patch = sample(image, cy+layer['py']+delta[:, 1, None, None],
                               cx+layer['px']+delta[:, 0, None, None])
                patch -= (patch*layer['weight']).sum(dim=(-1, -2))[:, None, None]
                strength = (patch.square()*layer['weight']).sum(dim=(-1, -2)).sqrt()
                reverse, reverse_valid, _ = peaks(self._scores(self.reference, layer, section, patch, strength))
                valid &= reverse_valid & (torch.linalg.vector_norm(reverse, dim=1) <= .75)
                confidence = ((peak-.8)/.15).clamp(0., 1.)*valid
                weight = layer['footprint']*confidence[:, None, None]
                indices = ((cy+layer['py'])*self.shape[1]+cx+layer['px']).flatten()
                denominator.flatten().scatter_add_(0, indices, weight.flatten())
                for axis in range(2):
                    numerator[axis].flatten().scatter_add_(0, indices, (weight*delta[:, axis, None, None]).flatten())
            residual = numerator/denominator.clamp_min(1e-300)
            residual *= denominator.clamp_max(1.)
            proposed = residual+sample(field, self.yy+residual[1], self.xx+residual[0], nearest=True)
            ux, uy = proposed-origin
            ux_y, ux_x = torch.gradient(ux)
            uy_y, uy_x = torch.gradient(uy)
            jacobian = (1+ux_x)*(1+uy_y)-ux_y*uy_x
            valid_field = (torch.isfinite(proposed).all() & (jacobian.min() >= .25)
                           & (torch.hypot(ux, uy).max() <= 6.))
            field = torch.where(valid_field, proposed, field)
        return field
