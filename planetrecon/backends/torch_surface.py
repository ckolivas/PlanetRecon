"""CUDA motion helpers: registration planes and camera-drift tracking.

Device equivalents of detector.extract_green_proxy / bilinear demosaic, of
globe_align.surface_displacement and of ring_align.RingRegistration
(MaskedRegistration with distinct peaks). The
CPU implementations remain the reference; only the final 3x3 peak neighbourhood
returns to the host for the shared subpixel decision.
"""
import numpy as np
import torch
import torch.nn.functional as F
from scipy.fft import next_fast_len
from scipy.ndimage import minimum_filter

from planetrecon.backends.torch_ap import _gaussian
from planetrecon.detector import BAYER_PATTERNS, is_bayer


def _shifted_sum(image, offsets):
    """Sum of image shifted by each (dy, dx), with zero outside (detector._neighbour_sum order)."""
    out = torch.zeros_like(image)
    h, w = image.shape[:2]
    for dy, dx in offsets:
        ty = slice(max(0, -dy), h - max(0, dy))
        sy = slice(max(0, dy), h - max(0, -dy))
        tx = slice(max(0, -dx), w - max(0, dx))
        sx = slice(max(0, dx), w - max(0, -dx))
        out[ty, tx] += image[sy, sx]
    return out


# detector._neighbour_sum and extract_green_proxy accumulation orders.
EIGHT = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1))
FOUR = ((1, 0), (-1, 0), (0, 1), (0, -1))


class TorchPlanes:
    """Registration planes for one capture's colour mode, on the device."""

    def __init__(self, shape, color, device='cuda:0'):
        self.device = device
        self.color = color
        self.bayer = is_bayer(color)
        if self.bayer:
            tile = torch.tensor([['RGB'.index(c) for c in row] for row in BAYER_PATTERNS[color]], device=device)
            y = torch.arange(shape[0], device=device)[:, None] % 2
            x = torch.arange(shape[1], device=device)[None, :] % 2
            labels = tile[y, x]
            self.masks = (labels[..., None] == torch.arange(3, device=device))
            self.mask_weights = self.masks.to(torch.float64)
            self.divisors = _shifted_sum(self.mask_weights, EIGHT).clamp_min(1.)
            self.green = self.masks[..., 1]
            self.green_divisor = _shifted_sum(self.green.to(torch.float64), FOUR).clamp_min(1.)
        self.luma = torch.tensor([.25, .5, .25], device=device, dtype=torch.float64)

    def upload(self, frame):
        return torch.as_tensor(np.asarray(frame) if not torch.is_tensor(frame) else frame,
                               device=self.device, dtype=torch.float64)

    def demosaic(self, image):
        planes = image[..., None] * self.mask_weights
        return torch.where(self.masks, planes, _shifted_sum(planes, EIGHT) / self.divisors)

    def tracking(self, image):
        """geometry_stack._alignment_plane: green proxy for Bayer, luminance for RGB."""
        if image.ndim == 3:
            return .5 * image[..., 1] + .25 * image[..., 0] + .25 * image[..., 2]
        if self.bayer:
            green = torch.where(self.green, image, torch.zeros_like(image))
            return torch.where(self.green, green, _shifted_sum(green, FOUR) / self.green_divisor)
        return image

    def luminance(self, image):
        """Colour registration plane: bilinear RGB luminance for Bayer."""
        if self.bayer:
            return self.demosaic(image) @ self.luma
        return self.tracking(image)


class TorchSurfaceTracker:
    """Device equivalent of globe_align.surface_displacement for one run.

    The reference globe support is fixed for the run and prepared once.
    """

    def __init__(self, warp, reference, reference_pose):
        from planetrecon.geometry.globe import field_rotate_sky, sky_to_body
        self.warp = warp
        self.model = warp.model
        self.device = warp.device
        self.reference_pose = reference_pose
        self.reference = torch.as_tensor(np.asarray(reference, dtype=np.float64), device=self.device)
        h, w = self.reference.shape
        y, x = np.indices((h, w))
        rx, ry = x + .5 - reference_pose.cx, reference_pose.cy - y - .5
        if self.model.apply_field:
            rx, ry = field_rotate_sky(rx, ry, -reference_pose.field_angle_rad)
        _, _, reference_globe, _ = sky_to_body(rx, ry, self.model.globe, reference_pose.t_s)
        support = minimum_filter(reference_globe, size=3, mode='constant', cval=0)
        self.reference_support = torch.as_tensor(support.astype(np.float64), device=self.device)
        self.matcher = MaskedMatcher((h, w), self.device)

    def on_globe(self, pose):
        """Visible near-side globe pixels at a frame pose (geometry.globe.globe_hit)."""
        from planetrecon.geometry.globe import body_to_obs_matrix
        g = self.model.globe
        sx, sy = self.warp.x - pose.cx, pose.cy - self.warp.y
        if self.model.apply_field:
            sx, sy = self.warp.rotate(sx, sy, -pose.field_angle_rad)
        a, c = g.equatorial_radius_px, g.polar_radius_px
        ia, ic = 1 / (a * a), 1 / (c * c)
        r = body_to_obs_matrix(g.pole_pa_rad, g.sub_obs_lat_rad, g.sub_obs_lon(pose.t_s))
        rt = r.T
        q = [rt[i, 0] * sx + rt[i, 1] * sy for i in range(3)]
        d = rt[:, 2]
        aa = (d[0] * d[0] + d[1] * d[1]) * ia + d[2] * d[2] * ic
        bb = 2 * ((q[0] * d[0] + q[1] * d[1]) * ia + q[2] * d[2] * ic)
        cc = (q[0] * q[0] + q[1] * q[1]) * ia + q[2] * q[2] * ic - 1
        disc = bb * bb - 4 * aa * cc
        t = torch.where(disc >= 0, (-bb + disc.clamp_min(0).sqrt()) / max(2 * aa, 1e-30), torch.nan)
        xb, yb, zb = (q[i] + t * d[i] for i in range(3))
        nz = r[2, 0] * (xb * ia) + r[2, 1] * (yb * ia) + r[2, 2] * (zb * ic)
        return (disc >= 0) & torch.isfinite(t) & (nz >= -1e-12)

    def _render(self, image, pose):
        x, y, valid = self.warp.map(pose, self.reference_pose)
        flat = image.reshape(-1)
        out = torch.zeros_like(flat)
        for index, weight in self.warp.corners(x, y, valid):
            out += flat[index] * weight
        return out.reshape(image.shape)

    def displacement(self, frame, pose):
        """Camera drift (dx, dy) matching the CPU surface_displacement, or None."""
        predicted = self._render(self.reference, pose)
        support = self._render(self.reference_support, pose)
        observed = ((support >= 1. - 1e-12) & self.on_globe(pose)).to(torch.float64)
        mask = _eroded(observed)
        template, count, energy = masked_template(predicted, mask)
        if count < 32 or energy <= 1e-12:
            return None
        return self.matcher(template, mask, count, energy, _gaussian(self.matcher.tensor(frame), 1.5))


def _eroded(observed, size=15):
    """minimum_filter(size, mode='constant', cval=0): erosion with an unobserved border."""
    r = size // 2
    return -F.max_pool2d(-F.pad(observed.to(torch.float64)[None, None], (r, r, r, r), value=0.), size,
                         stride=1)[0, 0] > .5


def masked_template(reference, mask):
    """MaskedRegistration's zero-mean smoothed template on the observed mask."""
    count = int(mask.sum())
    template = torch.zeros_like(reference)
    if count < 32:
        return template, count, 0.
    selected = _gaussian(reference, 1.5)[mask]
    template[mask] = selected - selected.mean()
    return template, count, float(template.square().sum())


class MaskedMatcher:
    """MaskedRegistration.displacement with distinct peaks, on the device.

    Float64 FFT correlations and the peak decisions stay on the GPU; only the
    3x3 neighbourhood of the accepted peak returns for the subpixel solve.
    """

    def __init__(self, shape, device='cuda:0'):
        self.device = device
        self.shape = tuple(shape)
        h, w = self.shape
        self.full = (2 * h - 1, 2 * w - 1)
        self.fft = tuple(next_fast_len(n, real=True) for n in self.full)
        self.ones_fft = torch.fft.rfft2(torch.ones((h, w), device=device, dtype=torch.float64), s=self.fft)

    def tensor(self, value):
        return torch.as_tensor(value if torch.is_tensor(value) else np.asarray(value), device=self.device,
                               dtype=torch.float64)

    def __call__(self, template, mask, count, energy, proxy, min_improvement=0.):
        maskf = mask.to(torch.float64)
        observed_fft = torch.fft.rfft2(torch.stack((proxy, proxy.square())), s=self.fft)
        kernels = torch.fft.rfft2(torch.stack((template, maskf)).flip((-2, -1)), s=self.fft)
        products = torch.stack((observed_fft[0] * kernels[0], observed_fft[0] * kernels[1],
                                observed_fft[1] * kernels[1], self.ones_fft * kernels[1]))
        dot, total, squares, overlap = torch.fft.irfft2(products, s=self.fft)[:, :self.full[0], :self.full[1]]
        variance = (squares - total.square() / count).clamp_min(0.)
        denominator = (energy * variance).sqrt()
        valid = (overlap >= count - 1e-6) & (denominator > 1e-12)
        scores = torch.where(valid, dot / denominator.clamp_min(1e-300), -torch.inf)
        index = int(torch.argmax(scores))
        py, px = divmod(index, scores.shape[1])
        peak = float(scores[py, px])
        if not np.isfinite(peak) or peak < .8:
            return None
        # Distinct maxima only (plateaus included), outside a 7x7 exclusion box.
        local = F.max_pool2d(scores[None, None], 3, stride=1, padding=1)[0, 0]
        competitors = torch.where(scores < local, -torch.inf, scores)
        competitors[max(0, py - 3):py + 4, max(0, px - 3):px + 4] = -torch.inf
        if float(competitors.max()) >= peak - .02:
            return None
        if py == 0 or px == 0 or py == scores.shape[0] - 1 or px == scores.shape[1] - 1:
            return None
        n = scores[py - 1:py + 2, px - 1:px + 2].cpu().numpy()
        if not np.isfinite(n).all():
            return None
        gy, gx = .5 * (n[2, 1] - n[0, 1]), .5 * (n[1, 2] - n[1, 0])
        hyy, hxx = n[0, 1] - 2 * peak + n[2, 1], n[1, 0] - 2 * peak + n[1, 2]
        hxy = .25 * (n[2, 2] - n[2, 0] - n[0, 2] + n[0, 0])
        hessian = np.array([[hyy, hxy], [hxy, hxx]])
        if np.linalg.eigvalsh(hessian)[-1] >= 0:
            return None
        delta = np.zeros(2) if peak >= 1 - 1e-12 else np.linalg.solve(-hessian, [gy, gx])
        if not np.isfinite(delta).all() or np.any(np.abs(delta) > 1.):
            return None
        offsets = np.array([py, px]) - (np.array(self.shape) - 1) + delta
        if min_improvement > 0:
            y, x = torch.nonzero(mask, as_tuple=True)
            moved = _bilinear_constant(proxy, y + offsets[0], x + offsets[1])
            moved = moved - moved.mean()
            norm = float((energy * moved.square().sum()).sqrt())
            gain = (float(template[mask] @ moved) / norm - float(scores[self.shape[0] - 1, self.shape[1] - 1])
                    if norm > 0 else -np.inf)
            if gain <= min_improvement:
                return 0., 0.
        return offsets[1], offsets[0]


def _bilinear_constant(image, y, x):
    """map_coordinates(order=1, mode='constant', cval=0): zero outside [0, n-1]."""
    h, w = image.shape
    inside = (y >= 0) & (y <= h - 1) & (x >= 0) & (x <= w - 1)
    y0 = y.floor().clamp(0, h - 1)
    x0 = x.floor().clamp(0, w - 1)
    fy, fx = (y - y0).clamp(0, 1), (x - x0).clamp(0, 1)
    y0, x0 = y0.long(), x0.long()
    y1, x1 = (y0 + 1).clamp(max=h - 1), (x0 + 1).clamp(max=w - 1)
    value = ((1 - fy) * ((1 - fx) * image[y0, x0] + fx * image[y0, x1])
             + fy * ((1 - fx) * image[y1, x0] + fx * image[y1, x1]))
    return torch.where(inside, value, torch.zeros_like(value))


class TorchRingTracker:
    """Device equivalent of ring_align.RingRegistration for one run's anchor.

    Exposed-ring masks, smoothed templates, field-rotated references and the
    interpolation-loss gate follow the CPU class; only peak neighbourhoods and
    scalars return to the host.
    """

    def __init__(self, reference, cx, cy, globe_radius, ring_radius, device='cuda:0'):
        from planetrecon.backends.torch_registration import TorchRingRegistrationOps
        self.ops = TorchRingRegistrationOps(np.shape(reference), device)
        self.matcher = MaskedMatcher(np.shape(reference), device)
        self.reference = self.matcher.tensor(reference)
        self.globe_radius, self.ring_radius = globe_radius, ring_radius
        self.mask = self._mask(cx, cy)
        self.template, self.count, self.energy = masked_template(self.reference, self.mask)

    def _mask(self, cx, cy, valid=None):
        radius = torch.hypot(self.ops.x - cx, self.ops.y - cy)
        mask = (radius > self.globe_radius + 8) & (radius < self.ring_radius + 8)
        mask[:7] = mask[-7:] = False
        mask[:, :7] = mask[:, -7:] = False
        return mask if valid is None else mask & valid

    def displacement(self, frame):
        if self.energy <= 1e-12:
            return None
        return self.matcher(self.template, self.mask, self.count, self.energy,
                            _gaussian(self.matcher.tensor(frame), 1.5))

    def displacement_at_pose(self, frame, pose, reference_pose):
        """Follow field rotation, gating drift against measured interpolation loss."""
        projected = self.ops.render_tensor(torch.stack((self.reference, torch.ones_like(self.reference)), -1),
                                           pose, reference_pose)
        predicted, support = projected[..., 0], projected[..., 1]
        roundtrip = self.ops.render_tensor(predicted, reference_pose, pose)
        values = _gaussian(roundtrip, 1.5)[self.mask]
        if values.numel():
            values = values - values.mean()
        norm = float((self.energy * values.square().sum()).sqrt())
        if norm <= 1e-12:
            return None
        loss = max(1e-12, 1. - float(self.template[self.mask] @ values) / norm)
        # Rotation may expose pixels outside the original capture.
        mask = self._mask(pose.cx, pose.cy, _eroded(support >= 1. - 1e-12))
        template, count, energy = masked_template(predicted, mask)
        if energy <= 1e-12:
            return None
        return self.matcher(template, mask, count, energy, _gaussian(self.matcher.tensor(frame), 1.5),
                            min_improvement=loss)
