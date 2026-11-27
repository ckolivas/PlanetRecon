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
                value = tl.load(image_ptr + site, inside, other=0.)
                parity = (yc % 2) * 2 + (xc % 2)
                colour = tl.where(parity == 0, L00, tl.where(parity == 1, L01, tl.where(parity == 2, L10, L11)))
                signal = tl.where(cols[None, :] == colour[:, None], (value * weight)[:, None], 0.)
                sites = tl.where(cols[None, :] == colour[:, None] + 3, weight[:, None], 0.)
                acc += signal + sites
            else:
                value = tl.load(image_ptr + site[:, None] * C + cols[None, :],
                                inside[:, None] & (cols[None, :] < C), other=0.)
                acc += tl.where(cols[None, :] < C, value * weight[:, None],
                                tl.where(cols[None, :] < D, weight[:, None], 0.))
    tl.store(out_ptr + i[:, None] * D + cols[None, :], acc, inside[:, None] & (cols[None, :] < D))


def pull(images, fields, labels=None):
    """Contributions (B, h, w, 2C) float64 of frames (B, h, w[, C]) pulled by fields (B, 2, h, w) (x, y).

    ``labels`` gives a Bayer frame's colour index (0 R, 1 G, 2 B) at its (row % 2, column % 2) sites.
    """
    images = images.to(torch.float64).contiguous()
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
