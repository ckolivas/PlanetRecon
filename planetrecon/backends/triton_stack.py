"""Fused CUDA kernels for grouped translation stacking with alignment points.

``pull`` resamples a group of frames along their dense fields into per-pixel
contributions: C signal channels then their C weights (the CFA sites of each
colour for Bayer frames, else the bilinear support). Per sample it forms the
same corner weights and products as ``TorchBackend.backproject_tensors``.

``accumulate`` adds a group's contributions to slot sums (Fourier rank bins and
the ordinary stack). Each pair (frame, slot) is weighted per pixel by its
alignment points' blended footprints, computed in the kernel as in
``LocalSelection._device_maps``: sum over the pair's points of wy x wx in the
blend precision, then in float64 x scale, plus the outside-point weight when the
pair takes it, x the frame's score. Every slot is read and written once per
group instead of once per frame, and no footprint map is stored.
"""
import torch
import triton
import triton.language as tl

BLOCK = 128


@triton.jit
def _pull(image_ptr, field_ptr, out_ptr, total, P, W, H,
          L00: tl.constexpr, L01: tl.constexpr, L10: tl.constexpr, L11: tl.constexpr,
          BAYER: tl.constexpr, C: tl.constexpr, D: tl.constexpr, DP: tl.constexpr, BLOCK: tl.constexpr):
    i = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    inside = i < total
    b = i // P
    p = i % P
    y = (p // W).to(tl.float64)
    x = (p % W).to(tl.float64)
    sx = tl.load(field_ptr + b * 2 * P + p, inside, other=0.)
    sy = tl.load(field_ptr + b * 2 * P + P + p, inside, other=0.)
    yf = y + sy
    xf = x + sx
    fy0 = tl.floor(yf)
    fx0 = tl.floor(xf)
    iy = fy0.to(tl.int64)
    ix = fx0.to(tl.int64)
    fy = yf - fy0
    fx = xf - fx0
    cols = tl.arange(0, DP)
    acc = tl.zeros((BLOCK, DP), dtype=tl.float64)
    for dy in tl.static_range(2):
        wy = fy if dy == 1 else 1. - fy
        for dx in tl.static_range(2):
            wx = fx if dx == 1 else 1. - fx
            yy = iy + dy
            xx = ix + dx
            keep = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
            weight = wy * wx * keep.to(tl.float64)
            yc = tl.minimum(tl.maximum(yy, 0), H - 1)
            xc = tl.minimum(tl.maximum(xx, 0), W - 1)
            site = b * P + yc * W + xc
            if BAYER:
                value = tl.load(image_ptr + site, inside, other=0).to(tl.float64)
                parity = (yc % 2) * 2 + (xc % 2)
                colour = tl.where(parity == 0, L00, tl.where(parity == 1, L01, tl.where(parity == 2, L10, L11)))
                signal = tl.where(cols[None, :] == colour[:, None], (value * weight)[:, None], 0.)
                sites = tl.where(cols[None, :] == colour[:, None] + 3, weight[:, None], 0.)
                acc += signal + sites
            else:
                value = tl.load(image_ptr + site[:, None] * C + cols[None, :],
                                inside[:, None] & (cols[None, :] < C), other=0).to(tl.float64)
                acc += tl.where(cols[None, :] < C, value * weight[:, None],
                                tl.where(cols[None, :] < D, weight[:, None], 0.))
    tl.store(out_ptr + i[:, None] * D + cols[None, :], acc, inside[:, None] & (cols[None, :] < D))


def pull(images, fields, labels=None):
    """Contributions (B, h, w, 2C) float64 of frames (B, h, w[, C]) of any numeric dtype pulled by fields
    (B, 2, h, w) (x, y). ``labels`` gives a Bayer frame's colour index (0 R, 1 G, 2 B) at its
    (row % 2, column % 2) sites.
    """
    images = images.contiguous()
    fields = fields.to(torch.float64).contiguous()
    B, h, w = images.shape[:3]
    C = 3 if labels is not None else (images.shape[3] if images.ndim == 4 else 1)
    D = 2 * C
    out = torch.empty((B, h, w, D), dtype=torch.float64, device=images.device)
    total = B * h * w
    (l00, l01), (l10, l11) = labels if labels is not None else ((0, 0), (0, 0))
    _pull[(triton.cdiv(total, BLOCK),)](images, fields, out, total, h * w, w, h, l00, l01, l10, l11,
                                        BAYER=labels is not None, C=C, D=D, DP=triton.next_power_of_2(D),
                                        BLOCK=BLOCK)
    return out


@triton.jit
def _accumulate(x_ptr, sums_ptr, wy_ptr, wx_ptr, scale_ptr, rest_ptr,
                pair_frame_ptr, pair_rest_ptr, pair_score_ptr, pair_first_ptr, points_ptr,
                slot_first_ptr, slot_ptr, P, W, K,
                D: tl.constexpr, DP: tl.constexpr, BLOCK: tl.constexpr):
    j = tl.program_id(0)
    p = (tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    inside = p < P
    y = p // W
    x = p % W
    cols = tl.arange(0, DP)
    mask = inside[:, None] & (cols[None, :] < D)
    scale = tl.load(scale_ptr + p, inside, other=0.)
    rest = tl.load(rest_ptr + p, inside, other=0.)
    acc = tl.zeros((BLOCK, DP), dtype=tl.float64)
    for q in range(tl.load(slot_first_ptr + j), tl.load(slot_first_ptr + j + 1)):
        first = tl.load(pair_first_ptr + q)
        last = tl.load(pair_first_ptr + q + 1)
        k = tl.load(points_ptr + first)
        footprint = tl.load(wy_ptr + y * K + k, inside, other=0.) * tl.load(wx_ptr + k * W + x, inside, other=0.)
        footprint = tl.where(first < last, footprint, 0.)
        for t in range(first + 1, last):
            k = tl.load(points_ptr + t)
            footprint += tl.load(wy_ptr + y * K + k, inside, other=0.) * tl.load(wx_ptr + k * W + x, inside, other=0.)
        weight = (footprint.to(tl.float64) * scale + tl.load(pair_rest_ptr + q) * rest) * tl.load(pair_score_ptr + q)
        f = tl.load(pair_frame_ptr + q).to(tl.int64)
        value = tl.load(x_ptr + (f * P + p)[:, None] * D + cols[None, :], mask, other=0.)
        acc += weight[:, None] * value
    slot = tl.load(slot_ptr + j).to(tl.int64)
    target = sums_ptr + (slot * P + p)[:, None] * D + cols[None, :]
    tl.store(target, tl.load(target, mask, other=0.) + acc, mask)


def accumulate(sums, contributions, wy, wx, scale, rest, pairs, upload=None):
    """Add contributions (G, P, D) into sums (slots, P, D) for ``pairs`` [(frame, slot, points, rest, score)].

    ``wy`` (h, K) and ``wx`` (K, w) are the footprints in blend precision; ``scale`` and ``rest`` (P,) float64.
    ``upload(key, array)`` optionally moves the pair tables to the device (e.g. from pinned memory).
    """
    if not pairs:
        return
    import numpy as np
    device = sums.device
    pairs = sorted(pairs, key=lambda pair: pair[1])
    slots = sorted({pair[1] for pair in pairs})
    slot_first, pair_first, points = [], [0], []
    for q, (_, slot, chosen, _, _) in enumerate(pairs):
        if not slot_first or pairs[q - 1][1] != slot:
            slot_first.append(q)
        points.extend(int(k) for k in chosen)
        pair_first.append(len(points))
    slot_first.append(len(pairs))
    tables = [[pair[0] for pair in pairs], pair_first, points + [0], slot_first, slots]
    offsets = np.cumsum([0] + [len(t) for t in tables])
    ints = np.concatenate([np.asarray(t, dtype=np.int32) for t in tables])
    floats = np.array([[pair[3] for pair in pairs], [pair[4] for pair in pairs]], dtype=np.float64)
    if upload is None:
        upload = lambda key, array: torch.as_tensor(array, device=device)
    ints, floats = upload('ints', ints), upload('floats', floats)
    frame, first, chosen, slot_first, slot_ids = (ints[offsets[k]:offsets[k + 1]] for k in range(5))
    P, D = sums.shape[1:]
    w = wx.shape[1]
    _accumulate[(len(slots), triton.cdiv(P, BLOCK))](
        contributions.contiguous(), sums, wy.contiguous(), wx.contiguous(), scale, rest,
        frame, floats[0], floats[1], first, chosen, slot_first, slot_ids, P, w, wy.shape[1],
        D=D, DP=triton.next_power_of_2(D), BLOCK=BLOCK)


@triton.jit
def _luminance(image_ptr, out_ptr, total, P, W, H,
               L00: tl.constexpr, L01: tl.constexpr, L10: tl.constexpr, L11: tl.constexpr,
               INTEGER: tl.constexpr, BLOCK: tl.constexpr):
    # 32-bit index arithmetic (``luminance`` checks the group fits); 64-bit integers are emulated.
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    inside = i < total
    b = i // P
    p = i - b * P
    y = p // W
    x = p - y * W
    parity = (y & 1) * 2 + (x & 1)
    own = tl.where(parity == 0, L00, tl.where(parity == 1, L01, tl.where(parity == 2, L10, L11)))
    # Integer samples sum exactly in int32 (as float64 sums of them would), so float64 is needed
    # only for the averages; consumer GPUs run float64 at a small fraction of integer speed.
    acc = tl.int32 if INTEGER else tl.float64
    centre = tl.load(image_ptr + i, inside, other=0).to(tl.float64)
    s0 = tl.zeros((BLOCK,), dtype=acc)
    s1 = tl.zeros((BLOCK,), dtype=acc)
    s2 = tl.zeros((BLOCK,), dtype=acc)
    n0 = tl.zeros((BLOCK,), dtype=tl.int32)
    n1 = tl.zeros((BLOCK,), dtype=tl.int32)
    n2 = tl.zeros((BLOCK,), dtype=tl.int32)
    for dy in tl.static_range(-1, 2):
        for dx in tl.static_range(-1, 2):
            if dy != 0 or dx != 0:
                yy = y + dy
                xx = x + dx
                keep = inside & (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
                value = tl.load(image_ptr + b * P + yy * W + xx, keep, other=0).to(acc)
                q = (yy & 1) * 2 + (xx & 1)
                colour = tl.where(q == 0, L00, tl.where(q == 1, L01, tl.where(q == 2, L10, L11)))
                s0 += tl.where(keep & (colour == 0), value, 0)
                s1 += tl.where(keep & (colour == 1), value, 0)
                s2 += tl.where(keep & (colour == 2), value, 0)
                n0 += (keep & (colour == 0)).to(tl.int32)
                n1 += (keep & (colour == 1)).to(tl.int32)
                n2 += (keep & (colour == 2)).to(tl.int32)
    r = tl.where(own == 0, centre, s0.to(tl.float64) / tl.maximum(n0, 1).to(tl.float64))
    g = tl.where(own == 1, centre, s1.to(tl.float64) / tl.maximum(n1, 1).to(tl.float64))
    bl = tl.where(own == 2, centre, s2.to(tl.float64) / tl.maximum(n2, 1).to(tl.float64))
    out = .25 * r
    out = out + .5 * g
    out = out + .25 * bl
    tl.store(out_ptr + i, out, inside)


def luminance(images, labels):
    """Bilinear-demosaic luminance (0.25 R + 0.5 G + 0.25 B) float64 of raw Bayer frames (B, h, w) of any
    numeric dtype; ``labels`` gives the colour index at each (row % 2, column % 2) site. Per pixel the
    same sums, in the same order, as ``TorchBackend.registration_plane``."""
    images = images.contiguous()
    B, h, w = images.shape
    out = torch.empty((B, h, w), dtype=torch.float64, device=images.device)
    total = B * h * w
    if total >= 2 ** 31:
        raise ValueError('frame group too large for 32-bit indexing')
    (l00, l01), (l10, l11) = labels
    integer = not images.dtype.is_floating_point and images.dtype.itemsize <= 2  # 8 neighbours fit in int32.
    _luminance[(triton.cdiv(total, BLOCK),)](images, out, total, h * w, w, h, l00, l01, l10, l11,
                                             INTEGER=integer, BLOCK=BLOCK)
    return out


@triton.jit
def _stencil(image_ptr, filtered_ptr, peak_ptr, out_ptr, P, W, H, blocks, BLOCK: tl.constexpr):
    b = tl.program_id(0)
    block = tl.program_id(1)
    p = block * BLOCK + tl.arange(0, BLOCK)
    inside = p < P
    y = p // W
    x = p - y * W
    reference = tl.load(filtered_ptr + p, inside, other=0.)
    py = tl.load(peak_ptr + 2 * b)
    px = tl.load(peak_ptr + 2 * b + 1)
    for dy in tl.static_range(-1, 2):
        for dx in tl.static_range(-1, 2):
            yy = (y + py + dy + 2 * H) % H
            xx = (x + px + dx + 2 * W) % W
            value = tl.load(image_ptr + b.to(tl.int64) * P + yy * W + xx, inside, other=0.)
            tl.store(out_ptr + (b * blocks + block) * 9 + (dy + 1) * 3 + (dx + 1), tl.sum(value * reference))


def stencil(images, filtered, peaks):
    """Correlations at the 3 x 3 lags around each frame's integer ``peaks`` (B, 2) (row, column): (B, 3, 3)
    float64, sum over pixels p of images[b, p + lag] x filtered[p] (indices wrap), summed per block then
    across blocks."""
    images = images.to(torch.float64).contiguous()
    B, h, w = images.shape
    P = h * w
    blocks = triton.cdiv(P, 1024)
    partial = torch.empty((B, blocks, 9), dtype=torch.float64, device=images.device)
    _stencil[(B, blocks)](images, filtered.contiguous(), peaks.to(torch.int32).contiguous(), partial,
                          P, w, h, blocks, BLOCK=1024)
    return partial.sum(1).reshape(B, 3, 3)
