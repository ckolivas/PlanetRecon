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
        def tensor(a):
            if torch.is_tensor(a):
                return a.to(device=device, dtype=torch.float64)
            return torch.as_tensor(np.ascontiguousarray(a), device=device, dtype=torch.float64)
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


def device_template(reference, indices, read_plane, correlate, max_shift, *, should_cancel=None, progress=None):
    """Device equivalent of ``local_align.build_template`` (planes may be tensors)."""
    ref = torch.tensor(np.asarray(reference, dtype=np.float64), device='cuda:0')
    h, w = ref.shape
    yy, xx = torch.meshgrid(torch.arange(h, device='cuda:0', dtype=torch.float64),
                            torch.arange(w, device='cuda:0', dtype=torch.float64), indexing='ij')
    total = torch.zeros_like(ref)
    weight = torch.zeros_like(ref)
    ones = torch.ones_like(ref)
    used = 0
    for step, index in enumerate(indices, 1):
        if should_cancel is not None and should_cancel():
            raise InterruptedError('cancelled while building the alignment template')
        if progress is not None:
            progress(step)
        frame = torch.as_tensor(read_plane(int(index)), device='cuda:0', dtype=torch.float64)
        shift = correlate(reference, frame)
        if not np.isfinite(shift).all() or max(abs(v) for v in shift) > max_shift:
            continue
        total.add_(sample(frame, yy + shift[1], xx + shift[0]))
        weight.add_(sample(ones, yy + shift[1], xx + shift[0]))
        used += 1
    if used < 4:
        raise ValueError('Local alignment needs four valid aligned template frames; use global alignment.')
    template = torch.where(weight > 0, total / weight.clamp_min(1e-300), ref)
    origin = correlate(reference, template)
    if not np.isfinite(origin).all() or max(abs(v) for v in origin) > max_shift:
        raise ValueError('Local template origin is unreliable; use global alignment.')
    observed = (weight > 0).to(torch.float64)
    anchored = sample(template * observed, yy + origin[1], xx + origin[0])
    support = sample(observed, yy + origin[1], xx + origin[0])
    return torch.where(support > 0, anchored / support.clamp_min(1e-300), ref).cpu().numpy()


class DeviceAccumulator:
    """Weighted sums kept on the GPU for one batch, then merged into CPU sums.

    The CPU arrays stay authoritative for checkpoints and fallback; ``pending``
    lists the capture indices whose contributions have not been merged yet.
    """
    def __init__(self):
        self.sums = None
        self.pending = []

    def add(self, index, weight, add, wt, demo, support):
        if not torch.is_tensor(weight):
            weight = torch.tensor(float(weight), device=add.device, dtype=torch.float64)
        scale = weight[..., None] if weight.ndim == 2 and add.ndim == 3 else weight
        parts = [scale * add, scale * wt]
        if demo is not None:
            parts += [scale * demo, (weight * support)[..., None]]
        if self.sums is None:
            self.sums = parts
        else:
            for total, part in zip(self.sums, parts):
                total.add_(part)
        self.pending.append(int(index))

    def flush(self, *targets):
        """Add the device sums into ``targets`` (accum, weight[, demosaic_accum, demosaic_weight])."""
        if self.sums is not None:
            for total, target in zip(self.sums, targets):
                target += total.cpu().numpy()
        self.sums = None
        self.pending = []

    def discard(self):
        pending, self.sums, self.pending = self.pending, None, []
        return pending


def build_points(reference, size, *, search=ap_align.SEARCH, valid_mask=None, device='cuda:0'):
    """Device equivalent of ``ap_align.AlignmentPoints`` placement on a reference tensor.

    Returns a matcher description for ``TorchAlignmentPoints`` (same centres,
    templates and field weights as the CPU class for the same reference).
    """
    from types import SimpleNamespace
    reference = torch.as_tensor(reference, device=device, dtype=torch.float64)
    h, w = reference.shape
    half, spacing = size // 2, size // 2
    margin = half + search
    ys = np.arange(margin, h - margin - size % 2, spacing)
    xs = np.arange(margin, w - margin - size % 2, spacing)
    window = np.outer(np.hanning(size), np.hanning(size))
    empty = SimpleNamespace(shape=(h, w), size=size, search=search, spacing=spacing,
                            centres=np.zeros((0, 2), dtype=np.int64), accepted=[])
    if not len(ys) or not len(xs):
        return empty
    ar = torch.arange(size, device=device)
    rows = torch.as_tensor(ys - half, device=device)[:, None] + ar
    cols = torch.as_tensor(xs - half, device=device)[:, None] + ar

    def sections(image):
        return image[rows[:, None, :, None], cols[None, :, None, :]]   # (ny, nx, size, size)
    keep = sections(reference).mean(dim=(-1, -2)) >= float(reference.max()) * ap_align.MIN_BRIGHTNESS
    if valid_mask is not None:
        invalid = F.pad((~torch.as_tensor(valid_mask, device=device)).long().cumsum(0).cumsum(1), (1, 0, 1, 0))
        span = size + 2 * search
        y0 = torch.as_tensor(ys - margin, device=device)[:, None]
        x0 = torch.as_tensor(xs - margin, device=device)[None, :]
        count = invalid[y0 + span, x0 + span] - invalid[y0, x0 + span] - invalid[y0 + span, x0] + invalid[y0, x0]
        keep &= count == 0
    win = torch.as_tensor(window, device=device, dtype=torch.float64)
    t = sections(gaussian_proxy(reference))[keep]
    t = t - (t * win).sum(dim=(-1, -2), keepdim=True) / win.sum()
    energy = ((t * t * win).sum(dim=(-1, -2)) / win.sum()).sqrt().cpu().numpy()
    grid = np.argwhere(keep.cpu().numpy())
    if not len(energy):
        return empty
    chosen = (energy >= .1 * np.median(energy)) & (energy > 1e-4 * abs(float(reference.max())))
    centres = np.column_stack([ys[grid[:, 0]], xs[grid[:, 1]]])[chosen]
    templates = t[torch.as_tensor(chosen, device=device)]
    gy, gx = ap_align._grid(h, ap_align.FIELD_STEP), ap_align._grid(w, ap_align.FIELD_STEP)
    sigma = .6 * spacing
    return SimpleNamespace(shape=(h, w), size=size, search=search, spacing=spacing, centres=centres,
                           window=win, weighted=templates * win,
                           norm=(templates.square() * win).sum(dim=(-1, -2)).sqrt(),
                           wy=ap_align._gaussians(gy, centres[:, 0].astype(float), sigma),
                           wx=ap_align._gaussians(gx, centres[:, 1].astype(float), sigma), accepted=[])


def predicted_local_coordinates(template_with_support, frame, pose, reference_pose, warp, size):
    """Device equivalent of motion_local.local_coordinates with alignment points (tensors in and out)."""
    from planetrecon.pipeline.motion_local import cuda_inverse_local_coordinates
    projected = warp.render_tensor(template_with_support, pose, reference_pose)
    predicted, support = projected[..., 0], projected[..., 1]
    matcher = build_points(predicted, size, valid_mask=support >= 1. - 1e-9, device=warp.device)
    frame = torch.as_tensor(frame, device=warp.device, dtype=torch.float64)
    if not len(matcher.centres):
        field = torch.zeros((2, *frame.shape), device=warp.device, dtype=torch.float64)
    else:
        field = TorchAlignmentPoints(matcher, device=warp.device).displacement(frame, (0., 0.))
    return cuda_inverse_local_coordinates(field, tensors=True)
