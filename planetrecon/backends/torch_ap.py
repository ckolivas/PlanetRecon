"""CUDA alignment points: grouped correlations, peak fits and field solve.

Mirrors ``pipeline.ap_align.AlignmentPoints``; the CPU matcher remains the
reference implementation. Batched matching runs in the planes' precision
(``backends.precision``); peak fits and field solves are float64. The field
stays on the device.
"""
import numpy as np
import torch
import torch.nn.functional as F

from planetrecon.backends.torch_circular import sample
from planetrecon.pipeline import ap_align


def _gaussian_kernel(sigma, device, dtype=torch.float64):
    radius = int(4. * sigma + .5)
    x = torch.arange(-radius, radius + 1, device=device, dtype=torch.float64)
    kernel = torch.exp(-.5 * (x / sigma) ** 2)
    return (kernel / kernel.sum()).to(dtype), radius


def _reflected(n, radius, device):
    """Symmetric (edge-inclusive) reflection indices, repeated for kernels longer than the axis."""
    index = torch.remainder(torch.arange(-radius, n + radius, device=device), 2 * n)
    return torch.where(index >= n, 2 * n - 1 - index, index)


def _gaussian(image, sigma):
    """SciPy ``gaussian_filter`` (mode 'reflect', truncate 4) over the last two axes, rows first."""
    kernel, radius = _gaussian_kernel(sigma, image.device, image.dtype)
    h, w = image.shape[-2:]
    out = image.reshape(-1, 1, h, w).index_select(-2, _reflected(h, radius, image.device))
    out = F.conv2d(out, kernel.view(1, 1, -1, 1)).index_select(-1, _reflected(w, radius, image.device))
    return F.conv2d(out, kernel.view(1, 1, 1, -1)).reshape(image.shape)


def gaussian_proxy(image):
    """Device equivalent of ``LocalRegistration.proxy``."""
    a = _gaussian(image, 1.5)
    return a - _gaussian(a, 8.0)


def local_quality(plane, shift, centres, sigma, device='cuda:0'):
    """Device equivalent of ``ap_align.LocalQuality.measure``."""
    c = np.asarray(centres, dtype=np.float64) + np.array([float(shift[1]), float(shift[0])])
    return local_quality_batch(torch.as_tensor(plane, device=device, dtype=torch.float64)[None], c[None], sigma)[0]


def local_quality_batch(planes, centres, sigma):
    """Local quality of frames (B, h, w), each around its own (n, 2) row/column centres: (B, n) host array.

    Computed in the planes' precision.
    """
    device, dtype = planes.device, planes.dtype
    h, w = planes.shape[-2:]
    # SciPy mode 'nearest' at the detector edge.
    kernel, radius = _gaussian_kernel(ap_align.QUALITY_SMOOTH_PX, device, dtype)
    padded = F.pad(planes[:, None], (radius, radius, radius, radius), mode='replicate')
    smooth = F.conv2d(F.conv2d(padded, kernel[None, None, None, :]), kernel[None, None, :, None])[:, 0]
    lap = torch.zeros_like(smooth)
    lap[:, 1:-1, 1:-1] = (smooth[:, :-2, 1:-1] + smooth[:, 2:, 1:-1] + smooth[:, 1:-1, :-2]
                          + smooth[:, 1:-1, 2:] - 4 * smooth[:, 1:-1, 1:-1])
    c = torch.as_tensor(np.asarray(centres, dtype=np.float64), device=device)
    ys = torch.arange(h, device=device, dtype=torch.float64)
    xs = torch.arange(w, device=device, dtype=torch.float64)
    wy = torch.exp(-.5 * ((ys - c[..., 0:1]) / sigma) ** 2).to(dtype)
    wx = torch.exp(-.5 * ((xs - c[..., 1:2]) / sigma) ** 2).to(dtype)
    energy = ((wy @ (lap * lap)) * wx).double().sum(-1)
    level = ((wy @ smooth) * wx).double().sum(-1)
    norm = wy.double().sum(-1) * wx.double().sum(-1)
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


class PointGrid:
    """Every alignment-point position for one detector shape and point size.

    ``AlignmentPoints`` places points on this grid. A
    batch of references keeps the points each would choose; fields for all
    frames are solved together with zero information at unused points, which
    leaves each frame's least-squares solution unchanged. Sections and
    correlations run in the precision of the planes given; peak fits and the
    field solve are float64.
    """

    def __init__(self, shape, size, *, search=ap_align.SEARCH, device='cuda:0'):
        self.shape, self.size, self.search, self.device = tuple(shape), size, search, device
        h, w = self.shape
        half, spacing = size // 2, size // 2
        margin = half + search
        ys = np.arange(margin, h - margin - size % 2, spacing)
        xs = np.arange(margin, w - margin - size % 2, spacing)
        self.grid = (len(ys), len(xs))
        self.centres = np.stack(np.meshgrid(ys, xs, indexing='ij'), -1).reshape(-1, 2)
        self.window = torch.as_tensor(np.outer(np.hanning(size), np.hanning(size)), device=device,
                                      dtype=torch.float64)
        ar = torch.arange(size, device=device)
        self.rows = torch.as_tensor(ys - half, device=device)[:, None] + ar
        self.cols = torch.as_tensor(xs - half, device=device)[:, None] + ar
        self.ys, self.xs, self.margin = ys, xs, margin
        span = size + 2 * search
        ar = torch.arange(span, device=device)
        c = torch.as_tensor(self.centres, device=device)
        self.tile_rows = (c[:, 0] - half - search)[:, None] + ar
        self.tile_cols = (c[:, 1] - half - search)[:, None] + ar
        gy, gx = ap_align._grid(h, ap_align.FIELD_STEP), ap_align._grid(w, ap_align.FIELD_STEP)
        sigma = .6 * spacing
        self.wy = torch.as_tensor(ap_align._gaussians(gy, self.centres[:, 0].astype(float), sigma), device=device)
        self.wx = torch.as_tensor(ap_align._gaussians(gx, self.centres[:, 1].astype(float), sigma), device=device)

    def sections(self, images):
        """(B, ny, nx, size, size) point sections of (B, h, w) images, as a strided view."""
        start, step = self.margin - self.size // 2, self.size // 2
        view = images[:, start:, start:].unfold(1, self.size, step).unfold(2, self.size, step)
        return view[:, :self.grid[0], :self.grid[1]]

    def choose(self, references, valid=None):
        """Points each reference keeps (``AlignmentPoints`` placement): frame and point indices, zero-mean templates."""
        if not len(self.centres):
            empty = torch.zeros(0, dtype=torch.long, device=self.device)
            return empty, empty, torch.zeros((0, self.size, self.size), device=self.device, dtype=references.dtype)
        n = len(references)
        window = self.window.to(references.dtype)
        peak = references.amax((-2, -1))
        keep = self.sections(references).mean((-1, -2)) >= (peak * ap_align.MIN_BRIGHTNESS)[:, None, None]
        if valid is not None:
            # The whole search footprint must be observed (e.g. a predicted reference).
            invalid = F.pad((~valid).long().cumsum(-2).cumsum(-1), (1, 0, 1, 0))
            span = self.size + 2 * self.search
            y0 = torch.as_tensor(self.ys - self.margin, device=self.device)[:, None]
            x0 = torch.as_tensor(self.xs - self.margin, device=self.device)[None, :]
            count = (invalid[:, y0 + span, x0 + span] - invalid[:, y0, x0 + span]
                     - invalid[:, y0 + span, x0] + invalid[:, y0, x0])
            keep &= count == 0
        t = self.sections(gaussian_proxy(references))
        t = t - (t * window).sum((-1, -2), keepdim=True) / window.sum()
        energy = ((t * t * window).sum((-1, -2)) / window.sum()).sqrt()
        host = torch.cat((energy.flatten(1).double(), keep.flatten(1).double(), peak[:, None].double()),
                         1).cpu().numpy()
        points = len(self.centres)
        frames, chosen = [], []
        for b in range(n):
            e, kept = host[b, :points], host[b, points:2 * points] > 0
            index = np.flatnonzero(kept)
            if not len(index):
                continue
            # Relative to the typical point, with an absolute floor for blank regions.
            e = e[index]
            index = index[(e >= .1 * np.median(e)) & (e > 1e-4 * abs(host[b, -1]))]
            frames.append(np.full(len(index), b)), chosen.append(index)
        if not frames:
            return self.choose(references[:0])
        frame_index = torch.as_tensor(np.concatenate(frames), device=self.device)
        point_index = torch.as_tensor(np.concatenate(chosen), device=self.device)
        return frame_index, point_index, t.flatten(1, 2)[frame_index, point_index]

    def fields(self, proxies, frame_index, point_index, templates, shifts=None):
        """Dense (x, y) fields (B, 2, h, w) for proxies matched against per-pair templates.

        ``shifts`` optionally gives each frame's global (dx, dy), applied as in
        ``TorchAlignmentPoints.displacement``. Returns fields and per-frame acceptance.
        """
        n = len(proxies)
        h, w = self.shape
        acceptance = [None] * n
        if shifts is not None and np.any(shifts):
            yy = torch.arange(h, device=self.device, dtype=torch.float64)[:, None]
            xx = torch.arange(w, device=self.device, dtype=torch.float64)[None, :]
            s = torch.as_tensor(np.asarray(shifts, dtype=np.float64), device=self.device)
            proxies = sample(proxies, yy + s[:, 1, None, None], xx + s[:, 0, None, None])
        H = torch.zeros((n, len(self.centres), 2, 2), device=self.device, dtype=torch.float64)
        Hd = torch.zeros((n, len(self.centres), 2), device=self.device, dtype=torch.float64)
        lam = np.full(n, ap_align.PRIOR)
        if len(frame_index):
            tiles = proxies[frame_index[:, None, None], self.tile_rows[point_index][:, :, None],
                            self.tile_cols[point_index][:, None, :]]
            window, templates = self.window.to(proxies.dtype), templates.to(proxies.dtype)
            norm = (templates.square() * window).sum((-1, -2)).sqrt().clamp_min(1e-15)
            num, s1, s2 = self.correlate(tiles, templates * window)
            var = (s2 - s1 * s1 / window.sum()).clamp_min(1e-12)
            ncc = (num / (var.sqrt() * norm[:, None, None])).double().cpu().numpy()
            # Peak decisions are tiny (m x 13 x 13); reuse the CPU reference logic.
            offset, information, ok, _ = ap_align._peak_fit(ncc)
            frames = frame_index.cpu().numpy()
            trace = information[:, 0, 0] + information[:, 1, 1]
            for b in range(n):
                mine = frames == b
                if mine.any():
                    acceptance[b] = float(ok[mine].mean())
                    accepted = trace[mine & ok]
                    lam[b] = ap_align.PRIOR * (float(np.median(accepted)) if len(accepted) else 1.)
            information = torch.as_tensor(information, device=self.device)
            H[frame_index, point_index] = information
            Hd[frame_index, point_index] = torch.einsum(
                'nij,nj->ni', information, torch.as_tensor(offset, device=self.device))
        A = torch.einsum('ky,kx,bkij->byxij', self.wy, self.wx, H)
        b = torch.einsum('ky,kx,bki->byxi', self.wy, self.wx, Hd)
        lam = torch.as_tensor(lam, device=self.device)[:, None, None]
        A[..., 0, 0] += lam
        A[..., 1, 1] += lam
        det = A[..., 0, 0] * A[..., 1, 1] - A[..., 0, 1] * A[..., 1, 0]
        u = torch.stack([(A[..., 1, 1] * b[..., 0] - A[..., 0, 1] * b[..., 1]) / det,
                         (A[..., 0, 0] * b[..., 1] - A[..., 1, 0] * b[..., 0]) / det], 1)
        # Coarse samples include both detector edges, so align_corners maps exactly.
        field = F.interpolate(u, size=self.shape, mode='bilinear', align_corners=True)
        if shifts is not None:
            field = field + torch.as_tensor(np.asarray(shifts, dtype=np.float64), device=self.device)[:, :, None, None]
        return field, acceptance

    def correlate(self, tiles, weighted):
        """Search-window correlations of tiles (m, span, span) with weighted templates and the window.

        The template term is a grouped convolution; the window sums use the
        Hann window's separability (rows, then columns) instead of full 2-D sums.
        """
        m = len(tiles)
        num = F.conv2d(tiles[None], weighted[:, None], groups=m)[0]
        hann = torch.as_tensor(np.hanning(self.size), device=self.device, dtype=tiles.dtype)

        def window_sum(image):
            rows = F.conv2d(image[:, None], hann.view(1, 1, 1, -1))
            return F.conv2d(rows, hann.view(1, 1, -1, 1))[:, 0]
        return num, window_sum(tiles), window_sum(tiles * tiles)

    def fixed(self, matcher, dtype=torch.float64):
        """Grid indices and device templates of a fixed-reference ``AlignmentPoints``."""
        lookup = {tuple(c): k for k, c in enumerate(self.centres.tolist())}
        index = torch.as_tensor([lookup[tuple(c)] for c in matcher.centres.tolist()], device=self.device,
                                dtype=torch.long)
        templates = torch.as_tensor(np.asarray(matcher.templates), device=self.device, dtype=dtype)
        return index, templates

    def fixed_fields(self, proxies, index, templates, shifts=None):
        """Fields for frames all matched against one reference's points."""
        n, k = len(proxies), len(index)
        frame_index = torch.arange(n, device=self.device).repeat_interleave(k)
        return self.fields(proxies, frame_index, index.repeat(n), templates.repeat(n, 1, 1), shifts)


def predicted_local_coordinates_batch(template_with_support, frames, poses, reference_pose, warp, grid):
    """motion_local.local_coordinates with alignment points for frames (B, h, w) at their poses.

    The fixed anchor-geometry reference is predicted at each pose, points are
    chosen on its fully observed support, and the inverted residual fields
    are returned as sampling positions (qx, qy) of shape (B, h, w). Matching
    and inversion run in the frames' precision.
    """
    from planetrecon.pipeline.motion_local import cuda_inverse_local_coordinates
    projected = warp.render_tensor(template_with_support.to(frames.dtype), list(poses), reference_pose)
    predicted, support = projected[..., 0], projected[..., 1]
    frame_index, point_index, templates = grid.choose(predicted, support >= 1. - 1e-9)
    proxies = gaussian_proxy(frames.to(device=warp.device))
    field, _ = grid.fields(proxies, frame_index, point_index, templates)
    return cuda_inverse_local_coordinates(field.to(frames.dtype), tensors=True)


def sample_frames(images, field):
    """``torch_circular.sample`` pull of each image (B, h, w) through its own (B, 2, h, w) field."""
    n, h, w = images.shape
    yy = torch.arange(h, device=images.device, dtype=images.dtype)[:, None]
    xx = torch.arange(w, device=images.device, dtype=images.dtype)[None, :]
    field = field.to(images.dtype)
    y, x = yy + field[:, 1], xx + field[:, 0]
    iy, ix = torch.floor(y).long(), torch.floor(x).long()
    fy, fx = y - iy, x - ix
    flat = images.reshape(n, -1)
    result = torch.zeros_like(images)
    for dy, wy in ((0, 1 - fy), (1, fy)):
        for dx, wx in ((0, 1 - fx), (1, fx)):
            sy, sx = iy + dy, ix + dx
            weight = wy * wx * ((sy >= 0) & (sy < h) & (sx >= 0) & (sx < w))
            index = (sy.clamp(0, h - 1) * w + sx.clamp(0, w - 1)).reshape(n, -1)
            result += flat.gather(1, index).reshape(n, h, w) * weight
    return result
