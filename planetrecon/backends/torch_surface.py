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


def _shifted_sum(image, offsets, first=0):
    """Sum of image shifted by each (dy, dx) over axes (first, first+1), zero outside (detector._neighbour_sum order)."""
    out = torch.zeros_like(image)
    lead = (slice(None),) * first
    h, w = image.shape[first:first + 2]
    for dy, dx in offsets:
        ty = slice(max(0, -dy), h - max(0, dy))
        sy = slice(max(0, dy), h - max(0, -dy))
        tx = slice(max(0, -dx), w - max(0, dx))
        sx = slice(max(0, dx), w - max(0, -dx))
        out[lead + (ty, tx)] += image[lead + (sy, sx)]
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
            self.planar_masks = self.masks.permute(2, 0, 1)[None]
            self.planar_weights = self.mask_weights.permute(2, 0, 1)[None]
            self.planar_divisors = self.divisors.permute(2, 0, 1)[None]
            ones = torch.ones(3, 1, 1, 3, device=device, dtype=torch.float64)
            self.row_box, self.column_box = ones, ones.transpose(-1, -2)

    def upload(self, frame):
        return torch.as_tensor(np.asarray(frame) if not torch.is_tensor(frame) else frame,
                               device=self.device, dtype=torch.float64)

    def constant(self, name, dtype):
        """A precomputed float64 constant in the precision of the planes it meets."""
        value = getattr(self, name)
        if value.dtype == dtype:
            return value
        cache = self.__dict__.setdefault('_cast', {})
        if (name, dtype) not in cache:
            cache[name, dtype] = value.to(dtype)
        return cache[name, dtype]

    def demosaic(self, image):
        """Bilinear RGB of raw Bayer frames (optionally with a leading batch axis), in their precision."""
        h, w = image.shape[-2:]
        c = lambda name: self.constant(name, image.dtype)
        planes = (image.reshape(-1, 1, h, w) * c('planar_weights'))
        # Sum of the eight neighbours: a 3x3 box (two separable passes) less the centre.
        box = F.conv2d(F.conv2d(F.pad(planes, (1, 1, 1, 1)), c('row_box'), groups=3), c('column_box'), groups=3)
        filled = torch.where(self.planar_masks, planes, (box - planes) / c('planar_divisors'))
        return filled.movedim(1, -1).reshape(*image.shape, 3)

    def tracking(self, image):
        """geometry_stack._alignment_plane: green proxy for Bayer, luminance for RGB."""
        if self.color in ('RGB', 'BGR'):
            return .5 * image[..., 1] + .25 * image[..., 0] + .25 * image[..., 2]
        if self.bayer:
            green = torch.where(self.green, image, torch.zeros_like(image))
            return torch.where(self.green, green, _shifted_sum(green, FOUR, image.ndim - 2)
                               / self.constant('green_divisor', image.dtype))
        return image

    def luminance(self, image, demosaiced=None):
        """Colour registration plane: bilinear RGB luminance for Bayer (_alignment_plane order)."""
        if self.bayer:
            rgb = self.demosaic(image) if demosaiced is None else demosaiced
            return .5 * rgb[..., 1] + .25 * rgb[..., 0] + .25 * rgb[..., 2]
        return self.tracking(image)


class TorchSurfaceTracker:
    """Device equivalent of globe_align.surface_displacement for one run.

    The reference globe support is fixed for the run and prepared once.
    """

    def __init__(self, warp, reference, reference_pose, dtype=torch.float64):
        from planetrecon.geometry.globe import field_rotate_sky, sky_to_body
        self.warp = warp
        self.model = warp.model
        self.device = warp.device
        self.reference_pose = reference_pose
        self.reference = torch.as_tensor(np.asarray(reference, dtype=np.float64), device=self.device).to(dtype)
        h, w = self.reference.shape
        y, x = np.indices((h, w))
        rx, ry = x + .5 - reference_pose.cx, reference_pose.cy - y - .5
        if self.model.apply_field:
            rx, ry = field_rotate_sky(rx, ry, -reference_pose.field_angle_rad)
        _, _, reference_globe, _ = sky_to_body(rx, ry, self.model.globe, reference_pose.t_s)
        support = minimum_filter(reference_globe, size=3, mode='constant', cval=0)
        self.reference_support = torch.as_tensor(support.astype(np.float64), device=self.device).to(dtype)
        self.matcher = MaskedMatcher((h, w), self.device, dtype)

    def on_globe(self, pose):
        """Visible near-side globe pixels at a frame pose, or a batch (geometry.globe.globe_hit)."""
        g = self.model.globe
        P = self.warp.poses(pose, pose)
        sx, sy = self.warp.x - P([p.cx for p in P.srcs]), P([p.cy for p in P.srcs]) - self.warp.y
        if self.model.apply_field:
            sx, sy = self.warp.rotate(sx, sy, P.angles([-p.field_angle_rad for p in P.srcs]))
        a, c = g.equatorial_radius_px, g.polar_radius_px
        ia, ic = 1 / (a * a), 1 / (c * c)
        rt = P.matrices([g.sub_obs_lon(p.t_s) for p in P.srcs], g).transpose(0, 2, 1)
        q = [P(rt[:, i, 0]) * sx + P(rt[:, i, 1]) * sy for i in range(3)]
        d = [P(rt[:, i, 2]) for i in range(3)]
        aa = (rt[:, 0, 2] * rt[:, 0, 2] + rt[:, 1, 2] * rt[:, 1, 2]) * ia + rt[:, 2, 2] * rt[:, 2, 2] * ic
        bb = 2 * ((q[0] * d[0] + q[1] * d[1]) * ia + q[2] * d[2] * ic)
        cc = (q[0] * q[0] + q[1] * q[1]) * ia + q[2] * q[2] * ic - 1
        disc = bb * bb - 4 * P(aa) * cc
        t = torch.where(disc >= 0, (-bb + disc.clamp_min(0).sqrt()) / P(np.maximum(2 * aa, 1e-30)), torch.nan)
        xb, yb, zb = (q[i] + t * d[i] for i in range(3))
        nz = P(rt[:, 0, 2]) * (xb * ia) + P(rt[:, 1, 2]) * (yb * ia) + P(rt[:, 2, 2]) * (zb * ic)
        return (disc >= 0) & torch.isfinite(t) & (nz >= -1e-12)

    def displacement(self, frame, pose):
        """Camera drift (dx, dy) matching the CPU surface_displacement, or None."""
        return self.displacements(self.matcher.tensor(frame)[None], [pose])[0]

    def displacements(self, frames, poses):
        """Drift for a batch of frames (leading axis) at their poses: (dx, dy) or None each."""
        projected = self.warp.render_tensor(torch.stack((self.reference, self.reference_support), -1),
                                            list(poses), self.reference_pose)
        predicted, support = projected[..., 0], projected[..., 1]
        mask = _eroded((support >= 1. - 1e-12) & self.on_globe(list(poses)))
        template, count, energy = masked_templates(predicted, mask)
        count, energy = (v.cpu().numpy() for v in (count, energy))
        usable = (count >= 32) & (energy > 1e-12)
        proxy = _gaussian(self.matcher.tensor(frames), 1.5)
        return self.matcher.batch(template, mask, count, energy, proxy, usable=usable)


def _eroded(observed, size=15):
    """minimum_filter(size, mode='constant', cval=0): erosion with an unobserved border (last two axes)."""
    r = size // 2
    h, w = observed.shape[-2:]
    padded = F.pad(observed.to(torch.float32).reshape(-1, 1, h, w), (r, r, r, r), value=0.)
    # A square minimum separates exactly into row and column passes.
    eroded = -F.max_pool2d(F.max_pool2d(-padded, (size, 1), stride=1), (1, size), stride=1)
    return (eroded > .5).reshape(observed.shape)


def masked_templates(reference, mask):
    """MaskedRegistration's zero-mean smoothed templates on observed masks (last two axes).

    Returns device templates, pixel counts and energies; a count below 32 has no energy.
    """
    maskf = mask.to(reference.dtype)
    count = maskf.sum((-2, -1))
    smooth = _gaussian(reference, 1.5)
    mean = (smooth * maskf).sum((-2, -1)) / count.clamp_min(1.)
    template = torch.where(mask, smooth - mean[..., None, None], 0.)
    energy = torch.where(count >= 32, template.square().sum((-2, -1)), 0.)
    return template * (count >= 32)[..., None, None], count, energy


class MaskedMatcher:
    """MaskedRegistration.displacement with distinct peaks, on the device.

    Float64 FFT correlations and the peak decisions stay on the GPU for a batch
    of frames; one small array per batch returns for the subpixel solves.

    A shift is only scored when the whole mask stays inside the frame. For
    those shifts a circular correlation of the detector size is exact (no
    wrapped term can reach a sample), so the transforms need no padding and
    validity follows from the mask's bounding box. Scores are evaluated in a
    window covering the valid shifts plus one sample, indexed like the full
    linear correlation.
    """

    MEMORY = 1 << 29   # bytes of correlation workspace per chunk of frames

    def __init__(self, shape, device='cuda:0', dtype=torch.float64):
        self.device, self.dtype = device, dtype
        self.shape = tuple(shape)
        h, w = self.shape
        self.full = (2 * h - 1, 2 * w - 1)
        self.fft = tuple(next_fast_len(n, real=True) for n in self.shape)
        self.chunk = max(1, self.MEMORY // (96 * self.fft[0] * self.fft[1]))

    def tensor(self, value):
        return torch.as_tensor(value if torch.is_tensor(value) else np.asarray(value), device=self.device,
                               dtype=self.dtype)

    def __call__(self, template, mask, count, energy, proxy, min_improvement=0.):
        return self.batch(template[None], mask[None], [count], [energy], proxy[None], [min_improvement])[0]

    def batch(self, template, mask, count, energy, proxy, min_improvement=None, usable=None):
        """Displacements (dx, dy) or None for each frame of a batch.

        ``count``, ``energy``, ``min_improvement`` and ``usable`` are host
        values per frame; frames that are not usable return None.
        """
        n = len(count)
        count, energy = np.asarray(count, dtype=np.float64), np.asarray(energy, dtype=np.float64)
        gate = np.zeros(n) if min_improvement is None else np.asarray(min_improvement, dtype=np.float64)
        selected = np.flatnonzero(np.ones(n, bool) if usable is None else usable)
        results = [None] * n
        for begin in range(0, len(selected), self.chunk):
            part = selected[begin:begin + self.chunk]
            index = torch.as_tensor(part, device=self.device)
            for k, result in zip(part, self._chunk(template[index], mask[index], count[part], energy[part],
                                                   proxy[index], gate[part])):
                results[k] = result
        return results

    def _windows(self, mask):
        """Full-correlation index ranges of the valid shifts and a shared scoring window per axis."""
        h, w = self.shape
        lows, highs, starts, sizes = [], [], [], []
        bounds = []
        for axis, n in ((-1, h), (-2, w)):
            present = mask.any(axis)
            ar = torch.arange(n, device=self.device)
            bounds += [torch.where(present, ar, n).min(-1).values, torch.where(present, ar, -1).max(-1).values]
        first_y, last_y, first_x, last_x = torch.stack(bounds, 1).cpu().numpy().T
        for n, first, last in ((h, first_y, last_y), (w, first_x, last_x)):
            # Shift s keeps the mask inside the frame for -first <= s <= n-1-last; index s+n-1.
            low, high = n - 1 - first, 2 * n - 2 - last
            size = min(2 * n - 1, int((high - low).max()) + 3)
            start = np.clip(low - 1, 0, 2 * n - 1 - size)
            lows.append(low), highs.append(high), starts.append(start), sizes.append(size)
        return lows, highs, starts, sizes

    def _chunk(self, template, mask, count, energy, proxy, gate):
        h, w = self.shape
        H, W = self.full
        maskf = mask.to(self.dtype)
        (low_y, low_x), (high_y, high_x), (start_y, start_x), (size_y, size_x) = self._windows(mask)
        if self.dtype != torch.float64:
            # Zero-mean templates make the scores independent of a constant offset; removing each
            # frame's mean keeps the reduced-precision variance sums from cancelling.
            proxy = proxy - proxy.mean((-2, -1), keepdim=True)
        observed_fft = torch.fft.rfft2(torch.stack((proxy, proxy.square()), 1), s=self.fft)
        kernels = torch.fft.rfft2(torch.stack((template, maskf), 1).flip((-2, -1)), s=self.fft)
        products = torch.stack((observed_fft[:, 0] * kernels[:, 0], observed_fft[:, 0] * kernels[:, 1],
                                observed_fft[:, 1] * kernels[:, 1]), 1)
        del observed_fft, kernels
        circular = torch.fft.irfft2(products, s=self.fft)
        del products
        tensor = lambda a: torch.as_tensor(np.asarray(a), device=self.device)
        ky = tensor(start_y)[:, None] + torch.arange(size_y, device=self.device)
        kx = tensor(start_x)[:, None] + torch.arange(size_x, device=self.device)
        frames = torch.arange(len(count), device=self.device)[:, None, None]
        dot, total, squares = (circular[frames, k, (ky % self.fft[0])[:, :, None], (kx % self.fft[1])[:, None, :]]
                               for k in range(3))
        del circular
        inside = (((ky >= tensor(low_y)[:, None]) & (ky <= tensor(high_y)[:, None]))[:, :, None]
                  & ((kx >= tensor(low_x)[:, None]) & (kx <= tensor(high_x)[:, None]))[:, None, :])
        counts = self.tensor(count)[:, None, None]
        variance = (squares - total.square() / counts).clamp_min(0.)
        denominator = (self.tensor(energy)[:, None, None] * variance).sqrt()
        scores = torch.where(inside & (denominator > 1e-12), dot / denominator.clamp_min(torch.finfo(self.dtype).tiny),
                             -torch.inf)
        flat = scores.flatten(1)
        best = flat.argmax(1)
        peak = flat.gather(1, best[:, None])[:, 0]
        wy, wx = best // size_x, best % size_x
        # Distinct maxima only (plateaus included), outside a 7x7 exclusion box.
        local = F.max_pool2d(scores[:, None], 3, stride=1, padding=1)[:, 0]
        competitors = torch.where(scores < local, -torch.inf, scores)
        box = (((torch.arange(size_y, device=self.device)[:, None] - wy[:, None, None]).abs() <= 3)
               & ((torch.arange(size_x, device=self.device)[None, :] - wx[:, None, None]).abs() <= 3))
        rival = torch.where(box, -torch.inf, competitors).flatten(1).max(1).values
        ny = (wy[:, None] + torch.arange(-1, 2, device=self.device)).clamp(0, size_y - 1)
        nx = (wx[:, None] + torch.arange(-1, 2, device=self.device)).clamp(0, size_x - 1)
        neighbours = scores[frames, ny[:, :, None], nx[:, None, :]].flatten(1)
        centre = scores[frames[:, 0, 0], h - 1 - tensor(start_y), w - 1 - tensor(start_x)]
        host = torch.cat((torch.stack((peak, rival, centre, wy.to(self.dtype), wx.to(self.dtype)), 1),
                          neighbours), 1).double().cpu().numpy()
        host[:, 3] += start_y
        host[:, 4] += start_x
        offsets = [self._subpixel(row) for row in host]
        out = [None if o is None else (o[1], o[0]) for o in offsets]
        gated = [k for k, o in enumerate(offsets) if o is not None and gate[k] > 0]
        if gated:
            # Measured improvement over no drift must exceed the interpolation loss.
            # The gate compares small differences: evaluate it in float64.
            g = torch.as_tensor(gated, device=self.device)
            shift = torch.as_tensor(np.array([offsets[k] for k in gated]), device=self.device, dtype=torch.float64)
            yy = torch.arange(h, device=self.device, dtype=torch.float64)[:, None]
            xx = torch.arange(w, device=self.device, dtype=torch.float64)[None, :]
            moved = _bilinear_constant(proxy[g].double(), yy + shift[:, 0, None, None], xx + shift[:, 1, None, None])
            m = maskf[g].double()
            moved = (moved - (moved * m).sum((-2, -1), keepdim=True) / counts[g].double()) * m
            sums = torch.stack(((template[g].double() * moved).sum((-2, -1)), moved.square().sum((-2, -1))), 1)
            for k, (product, square) in zip(gated, sums.cpu().numpy()):
                norm = np.sqrt(energy[k] * square)
                gain = product / norm - host[k, 2] if norm > 0 else -np.inf
                if gain <= gate[k]:
                    out[k] = (0., 0.)
        return out

    def _subpixel(self, row):
        """Host peak acceptance and quadratic solve: (dy, dx) offsets or None."""
        peak, rival, _, py, px = row[:5]
        py, px = int(py), int(px)
        if not np.isfinite(peak) or peak < .8 or rival >= peak - .02:
            return None
        if py == 0 or px == 0 or py == self.full[0] - 1 or px == self.full[1] - 1:
            return None
        n = row[5:].reshape(3, 3)
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
        return np.array([py, px]) - (np.array(self.shape) - 1) + delta


def _bilinear_constant(image, y, x):
    """map_coordinates(order=1, mode='constant', cval=0) per frame: zero outside [0, n-1]."""
    h, w = image.shape[-2:]
    inside = (y >= 0) & (y <= h - 1) & (x >= 0) & (x <= w - 1)
    y0 = y.floor().clamp(0, h - 1)
    x0 = x.floor().clamp(0, w - 1)
    fy, fx = (y - y0).clamp(0, 1), (x - x0).clamp(0, 1)
    y0, x0 = y0.long(), x0.long()
    y1, x1 = (y0 + 1).clamp(max=h - 1), (x0 + 1).clamp(max=w - 1)
    flat = image.reshape(-1, h * w)
    frames = torch.arange(flat.shape[0], device=image.device)[:, None, None]
    at = lambda yy, xx: flat[frames, yy * w + xx]
    value = ((1 - fy) * ((1 - fx) * at(y0, x0) + fx * at(y0, x1))
             + fy * ((1 - fx) * at(y1, x0) + fx * at(y1, x1)))
    return torch.where(inside, value, torch.zeros_like(value))


class TorchRingTracker:
    """Device equivalent of ring_align.RingRegistration for one run's anchor.

    Exposed-ring masks, smoothed templates, field-rotated references and the
    interpolation-loss gate follow the CPU class; only peak neighbourhoods and
    scalars return to the host, once per batch of frames.
    """

    def __init__(self, reference, cx, cy, globe_radius, ring_radius, device='cuda:0', dtype=torch.float64):
        from planetrecon.backends.torch_registration import TorchRingRegistrationOps
        self.ops = TorchRingRegistrationOps(np.shape(reference), device)
        self.matcher = MaskedMatcher(np.shape(reference), device, dtype)
        self.reference = self.matcher.tensor(reference)
        self.globe_radius, self.ring_radius = globe_radius, ring_radius
        self.mask = self._mask(cx, cy)
        template, count, energy = masked_templates(self.reference, self.mask)
        self.template, self.count, self.energy = template, int(count), float(energy)

    def _mask(self, cx, cy, valid=None):
        radius = torch.hypot(self.ops.x - cx, self.ops.y - cy)
        mask = (radius > self.globe_radius + 8) & (radius < self.ring_radius + 8)
        mask[..., :7, :] = mask[..., -7:, :] = False
        mask[..., :, :7] = mask[..., :, -7:] = False
        return mask if valid is None else mask & valid

    def displacement(self, frame):
        return self.displacements(self.matcher.tensor(frame)[None])[0]

    def displacements(self, frames):
        """Drift for a batch of frames against the fixed anchor rings."""
        n = len(frames)
        if self.energy <= 1e-12:
            return [None] * n
        proxy = _gaussian(self.matcher.tensor(frames), 1.5)
        return self.matcher.batch(self.template.expand(n, -1, -1), self.mask.expand(n, -1, -1),
                                  [self.count] * n, [self.energy] * n, proxy)

    def displacement_at_pose(self, frame, pose, reference_pose):
        """Follow field rotation, gating drift against measured interpolation loss."""
        return self.displacements_at_poses(self.matcher.tensor(frame)[None], [pose], reference_pose)[0]

    def displacements_at_poses(self, frames, poses, reference_pose):
        """Batched ``displacement_at_pose`` for frames (leading axis) at their poses."""
        poses = list(poses)
        projected = self.ops.render_tensor(torch.stack((self.reference, torch.ones_like(self.reference)), -1),
                                           poses, reference_pose)
        predicted, support = projected[..., 0], projected[..., 1]
        roundtrip = self.ops.render_tensor(predicted, reference_pose, poses, per_pose=True)
        # The interpolation loss is a small difference: measure it in float64.
        maskf = self.mask.to(torch.float64)
        smooth = _gaussian(roundtrip, 1.5).double()
        values = (smooth - (smooth * maskf).sum((-2, -1), keepdim=True) / max(self.count, 1)) * maskf
        # Rotation may expose pixels outside the original capture.
        cx = torch.as_tensor([p.cx for p in poses], device=self.ops.device, dtype=torch.float64)
        cy = torch.as_tensor([p.cy for p in poses], device=self.ops.device, dtype=torch.float64)
        mask = self._mask(cx[:, None, None], cy[:, None, None], _eroded(support >= 1. - 1e-12))
        template, count, energy = masked_templates(predicted, mask)
        host = torch.stack(((self.template.double() * values).sum((-2, -1)), values.square().sum((-2, -1)),
                            count.double(), energy.double()), 1).cpu().numpy()
        norm = np.sqrt(self.energy * host[:, 1])
        usable = (norm > 1e-12) & (host[:, 3] > 1e-12)
        loss = np.maximum(1e-12, 1. - host[:, 0] / np.where(usable, norm, 1.))
        return self.matcher.batch(template, mask, host[:, 2], host[:, 3], _gaussian(self.matcher.tensor(frames), 1.5),
                                  min_improvement=loss, usable=usable)
