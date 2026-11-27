"""CUDA alignment points: grouped correlations, peak fits and field solve.

Mirrors ``pipeline.ap_align.AlignmentPoints``; the CPU matcher remains the
reference implementation. Batched matching runs in the planes' precision
(``backends.precision``); peak fits and field solves are float64. The field
stays on the device.
"""
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F

from planetrecon.backends import precision
from planetrecon.pipeline import ap_align


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


def sample_batch(images, y, x):
    """``sample`` of each of images (B, h, w) at its own coordinates y, x (broadcasting to (B, ...))."""
    B, h, w = images.shape
    shape = torch.broadcast_shapes(y.shape, x.shape)
    iy, ix = torch.floor(y).long(), torch.floor(x).long()
    fy, fx = y-iy, x-ix
    flat = images.reshape(B, -1)
    result = torch.zeros(shape, dtype=images.dtype, device=images.device)
    for dy, wy in ((0, 1-fy), (1, fy)):
        for dx, wx in ((0, 1-fx), (1, fx)):
            sy, sx = iy+dy, ix+dx
            weight = wy*wx*((sy >= 0) & (sy < h) & (sx >= 0) & (sx < w))
            index = (sy.clamp(0, h-1)*w + sx.clamp(0, w-1)).expand(shape).reshape(B, -1)
            result += torch.gather(flat, 1, index).reshape(shape)*weight
    return result


@lru_cache(maxsize=64)
def _gaussian_kernel(sigma, device, dtype=torch.float64):
    radius = int(4. * sigma + .5)
    x = torch.arange(-radius, radius + 1, device=device, dtype=torch.float64)
    kernel = torch.exp(-.5 * (x / sigma) ** 2)
    return (kernel / kernel.sum()).to(dtype), radius


@lru_cache(maxsize=64)
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
    """Device equivalent of ``local_align.proxy``."""
    a = _gaussian(image, 1.5)
    return a - _gaussian(a, 8.0)


def local_quality(plane, shift, centres, sigma, device='cuda:0'):
    """Device equivalent of ``ap_align.LocalQuality.measure``."""
    c = np.asarray(centres, dtype=np.float64) + np.array([float(shift[1]), float(shift[0])])
    plane = torch.as_tensor(plane, device=device).to(precision.REGISTRATION)
    return local_quality_batch(plane[None], c[None], sigma)[0]


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
        half, R = self.size // 2, self.search
        span = self.size + 2 * R
        ar = torch.arange(span, device=device, dtype=torch.float64)
        iy = torch.as_tensor(self.centres[:, 0] - half - R, device=device, dtype=torch.float64)
        ix = torch.as_tensor(self.centres[:, 1] - half - R, device=device, dtype=torch.float64)
        # Search tiles in detector coordinates: (n, span, 1) rows and (n, 1, span) columns.
        self.rows = (iy[:, None] + ar[None, :])[:, :, None]
        self.cols = (ix[:, None] + ar[None, :])[:, None, :]
        self.hann = tensor(np.hanning(self.size))
        self.matcher = matcher

    def _ncc(self, tiles):
        """Normalised correlations of (..., n, span, span) tiles, in their precision.

        The Hann window sums use its separability (rows, then columns).
        """
        n, dtype = len(self.centres), tiles.dtype
        lead, span = tiles.shape[:-3], tiles.shape[-1]
        flat = tiles.reshape(-1, span, span)
        groups = len(flat)
        weighted = self.weighted.to(dtype).repeat(groups // n, 1, 1)
        num = F.conv2d(flat[None], weighted[:, None], groups=groups)[0]
        hann = self.hann.to(dtype)

        def window_sum(image):
            rows = F.conv2d(image[:, None], hann.view(1, 1, 1, -1))
            return F.conv2d(rows, hann.view(1, 1, -1, 1))[:, 0]
        s1, s2 = window_sum(flat), window_sum(flat * flat)
        var = (s2 - s1 * s1 / self.window.sum().to(dtype)).clamp_min(1e-12)
        norm = self.norm.to(dtype).repeat(groups // n)
        return (num / (var.sqrt() * norm[:, None, None])).reshape(*lead, n, *num.shape[-2:])

    def peak_fits_batch(self, frames, shifts):
        """Offsets (B, n, 2), information (B, n, 2, 2) and acceptance (B, n) of frames (B, h, w)
        with global shifts (B, 2), with the correlations on the device in registration precision."""
        frames = torch.as_tensor(frames, device=self.device)
        proxies = gaussian_proxy(frames.to(precision.REGISTRATION))
        s = torch.as_tensor(np.asarray(shifts, dtype=np.float64), device=self.device)
        # Only the search tiles are resampled: the same samples as shifting each whole proxy.
        tiles = sample_batch(proxies, self.rows[None] + s[:, 1, None, None, None],
                             self.cols[None] + s[:, 0, None, None, None])
        ncc = self._ncc(tiles).double().cpu().numpy()
        B, n = ncc.shape[:2]
        # Peak decisions are tiny (B x n x 13 x 13); reuse the CPU reference logic.
        offset, H, ok, _ = ap_align._peak_fit(ncc.reshape(B * n, *ncc.shape[2:]))
        return offset.reshape(B, n, 2), H.reshape(B, n, 2, 2), ok.reshape(B, n)

    def peak_fits(self, frame, global_shift):
        """``AlignmentPoints.peak_fits`` with the correlations on the device, in registration precision."""
        offset, H, ok = self.peak_fits_batch(torch.as_tensor(frame, device=self.device)[None], [global_shift])
        return offset[0], H[0], ok[0]

    def displacements(self, frames, shifts):
        """Dense (x, y) fields (B, 2, h, w) of frames (B, h, w) with global shifts (B, 2) (x, y)."""
        shifts = np.asarray(shifts, dtype=np.float64).reshape(-1, 2)
        B = len(shifts)
        g = torch.as_tensor(shifts, device=self.device)[:, :, None, None]
        if not len(self.centres):
            return g.expand(B, 2, *self.shape).clone()
        offset, H, ok = self.peak_fits_batch(frames, shifts)
        self.matcher.accepted.extend(float(v) for v in ok.mean(axis=1))
        n = len(self.centres)
        if self.matcher.trust is not None:
            H = ap_align.trusted(H.reshape(B * n, 2, 2), np.tile(self.matcher.trust, (B, 1, 1))).reshape(B, n, 2, 2)
        trace = H[..., 0, 0] + H[..., 1, 1]
        lam = ap_align.PRIOR * np.array([float(np.median(t[k])) if k.any() else 1. for t, k in zip(trace, ok)])
        H_t = torch.as_tensor(H, device=self.device, dtype=torch.float64)
        d = torch.as_tensor(offset, device=self.device, dtype=torch.float64)
        Hd = torch.einsum('bnij,bnj->bni', H_t, d)
        # Coarse-grid sums over points as matrix products: (gy, n) @ (n, gx) for every frame and entry.
        wyT = self.wy.T
        A = (wyT @ (H_t.permute(0, 2, 3, 1)[..., None] * self.wx)).permute(0, 3, 4, 1, 2)
        b = (wyT @ (Hd.permute(0, 2, 1)[..., None] * self.wx)).permute(0, 2, 3, 1)
        lam = torch.as_tensor(lam, device=self.device)[:, None, None]
        A[..., 0, 0] += lam
        A[..., 1, 1] += lam
        det = A[..., 0, 0] * A[..., 1, 1] - A[..., 0, 1] * A[..., 1, 0]
        u = torch.stack([(A[..., 1, 1] * b[..., 0] - A[..., 0, 1] * b[..., 1]) / det,
                         (A[..., 0, 0] * b[..., 1] - A[..., 1, 0] * b[..., 0]) / det], 1)
        # Coarse samples include both detector edges, so align_corners maps exactly.
        return F.interpolate(u, size=self.shape, mode='bilinear', align_corners=True) + g

    def displacement(self, frame, global_shift):
        return self.displacements(torch.as_tensor(frame, device=self.device)[None], [global_shift])[0]


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


def fused_kernels():
    """The fused Triton stacking kernels, or None (the torch equivalents are used)."""
    try:
        from planetrecon.backends import triton_stack
    except ImportError:
        return None
    return triton_stack


class DeviceSums:
    """Weighted sums of grouped alignment-point frames kept on the GPU, merged into CPU sums.

    Slots 0..n_bins-1 are Fourier weighting's rank-bin sums and slot n_bins the
    ordinary stack; each holds (h*w, 2C) float64: C signal channels, then their
    C weights. Frames arrive in groups as contributions (G, h*w, 2C) with pairs
    (frame in group, slot, alignment points, outside weight, score). A pair's
    per-pixel weight is its points' blended footprints (``blend``: wy (h, K),
    wx (K, w) in blend precision, scale and rest (h*w,) float64) as in
    ``LocalSelection``. The CPU arrays stay authoritative for checkpoints and
    fallback: the sums merge every ``BIN_FRAMES`` frames, before a checkpoint
    and at the end, and the ordinary sums also when a preview is formed. ``pending`` lists the capture indices whose ordinary sums have not
    been merged, ``bin_pending`` those whose bin sums have not.
    """
    BIN_FRAMES = 1024

    def __init__(self, shape, channels, n_bins, blend, *, device='cuda:0'):
        h, w = shape
        self.shape, self.channels, self.main = (h, w), channels, n_bins
        self.sums = torch.zeros((n_bins + 1, h * w, 2 * channels), dtype=torch.float64, device=device)
        self.blend = blend
        self.kernels = fused_kernels()
        self.pending, self.bin_pending, self.touched = [], [], set()

    def add(self, indices, contributions, pairs):
        """Add one group's (G, h*w, 2C) ``contributions`` for capture ``indices`` along ``pairs``."""
        wy, wx, scale, rest = self.blend
        if self.kernels is not None:
            self.kernels.accumulate(self.sums, contributions, wy, wx, scale, rest, pairs)
        else:
            for f, slot, points, outside, score in pairs:
                chosen = torch.zeros(wy.shape[1], dtype=wy.dtype, device=wy.device)
                chosen[torch.as_tensor(np.asarray(points, dtype=np.int64), device=wy.device)] = 1
                weight = ((wy * chosen) @ wx).reshape(-1).to(torch.float64) * scale + outside * rest
                self.sums[slot].addcmul_((score * weight)[:, None], contributions[f])
        self.pending.extend(int(i) for i in indices)
        binned = {pair[0] for pair in pairs if pair[1] != self.main}
        self.bin_pending.extend(int(indices[f]) for f in sorted(binned))
        self.touched.update(pair[1] for pair in pairs if pair[1] != self.main)

    def _merge(self, slot, accum, weight):
        values = self.sums[slot].cpu().numpy()
        accum += values[:, :self.channels].reshape(accum.shape)
        weight += values[:, self.channels:].reshape(weight.shape)
        self.sums[slot].zero_()

    def flush_main(self, accum, weight):
        """Add the ordinary sums into (accum, weight), e.g. for a preview."""
        if self.pending:
            self._merge(self.main, accum, weight)
        self.pending = []

    def flush(self, accum, weight, bins=None, everything=True):
        """When ``everything`` or ``BIN_FRAMES`` frames are waiting, add the ordinary sums into
        (accum, weight) and the rank-bin sums into ``bins`` (bin_accum, bin_weight)."""
        if everything or len(self.pending) >= self.BIN_FRAMES:
            self.flush_main(accum, weight)
        if everything or len(self.bin_pending) >= self.BIN_FRAMES:
            for slot in sorted(self.touched):
                self._merge(slot, bins[0][slot], bins[1][slot])
            self.touched, self.bin_pending = set(), []

    def discard(self):
        """(frames with no merged sums, frames whose bin sums alone are unmerged); clears both."""
        pending = self.pending
        bin_only = sorted(set(self.bin_pending) - set(pending))
        self.sums.zero_()
        self.pending, self.bin_pending, self.touched = [], [], set()
        return pending, bin_only


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
        self.trust = None
        gy, gx = ap_align._grid(h, ap_align.FIELD_STEP), ap_align._grid(w, ap_align.FIELD_STEP)
        sigma = .6 * spacing
        self.wy = torch.as_tensor(ap_align._gaussians(gy, self.centres[:, 0].astype(float), sigma), device=device)
        self.wx = torch.as_tensor(ap_align._gaussians(gx, self.centres[:, 1].astype(float), sigma), device=device)

    def set_trust(self, trust):
        """Apply a calibrated ``AlignmentPoints.trust_map`` to every frame's points (None clears it)."""
        self.trust = None if not trust else np.stack(
            [trust.get(tuple(c), np.eye(2)) for c in self.centres.tolist()])

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
            none = torch.zeros(0, dtype=torch.long, device=self.device)
            return none, none, t.flatten(1, 2)[none, none]
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
            if self.trust is not None:
                information = ap_align.trusted(information, self.trust[point_index.cpu().numpy()])
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
    projected = warp.render_tensor(template_with_support.to(frames.dtype), list(poses), reference_pose,
                                   gain_channels=1)
    predicted, support = projected[..., 0], projected[..., 1]
    frame_index, point_index, templates = grid.choose(predicted, support >= 1. - 1e-9)
    proxies = gaussian_proxy(frames.to(device=warp.device))
    field, _ = grid.fields(proxies, frame_index, point_index, templates)
    return cuda_inverse_local_coordinates(field.to(frames.dtype), tensors=True)


def sample_frames(images, field):
    """``sample`` pull of each image (B, h, w) through its own (B, 2, h, w) field."""
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
