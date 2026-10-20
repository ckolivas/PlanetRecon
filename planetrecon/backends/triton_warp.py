"""Fused float64 bilinear pull for TorchGlobeWarp.render_tensor.

One kernel replaces the four corner index/weight tensors, their gathers and
the weighted sums. The arithmetic and its order follow ``TorchGlobeWarp.corners``
and ``render_tensor`` exactly (no fused multiply-adds), so results are identical.
"""
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _pull(x_ptr, y_ptr, valid_ptr, image_ptr, out_ptr, total, plane, frame_stride,
          H: tl.constexpr, W: tl.constexpr, C: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    inside = i < total
    x = tl.load(x_ptr + i, inside, other=0.) - .5
    y = tl.load(y_ptr + i, inside, other=0.) - .5
    valid = tl.load(valid_ptr + i, inside, other=0) != 0
    finite = (x == x) & (y == y) & (tl.abs(x) < float('inf')) & (tl.abs(y) < float('inf'))
    x = tl.where(finite, x, -2.)
    y = tl.where(finite, y, -2.)
    rx = libdevice.rint(x)
    ry = libdevice.rint(y)
    x = tl.where(tl.abs(x - rx) < 1e-10, rx, x)
    y = tl.where(tl.abs(y - ry) < 1e-10, ry, y)
    fx0 = libdevice.floor(x)
    fy0 = libdevice.floor(y)
    ix = fx0.to(tl.int64)
    iy = fy0.to(tl.int64)
    fx = x - fx0
    fy = y - fy0
    base = (i // plane) * frame_stride
    for c in tl.static_range(C):
        out = tl.zeros((BLOCK,), dtype=tl.float64)
        for dy in tl.static_range(2):
            wy = fy if dy == 1 else 1. - fy
            for dx in tl.static_range(2):
                wx = fx if dx == 1 else 1. - fx
                xx = ix + dx
                yy = iy + dy
                keep = valid & (xx >= 0) & (xx < W) & (yy >= 0) & (yy < H)
                index = tl.minimum(tl.maximum(yy, 0), H - 1) * W + tl.minimum(tl.maximum(xx, 0), W - 1)
                value = tl.load(image_ptr + base + index * C + c, inside, other=0.)
                weight = wy * wx * keep.to(tl.float64)
                part = value * weight
                out = part if (dy == 0 and dx == 0) else out + part
        tl.store(out_ptr + i * C + c, out, inside)


def pull(image, x, y, valid, per_pose):
    """Bilinear pull of (h, w[, C]) or per-pose (B, h, w[, C]) images at mapped detector coordinates."""
    h, w = x.shape[-2:]
    lead = x.shape[:-2]
    channels = image.shape[-1] if image.ndim == (len(lead) if per_pose else 0) + 3 else 1
    image = image.contiguous()
    x, y = x.contiguous(), y.contiguous()
    valid = valid.expand(x.shape).contiguous().to(torch.uint8)
    out = torch.empty((*lead, h, w, channels), device=x.device, dtype=torch.float64)
    total, block = x.numel(), 256
    _pull[(triton.cdiv(total, block),)](x, y, valid, image, out, total, h * w,
                                        h * w * channels if per_pose else 0,
                                        H=h, W=w, C=channels, BLOCK=block, enable_fp_fusion=False)
    return out


@triton.jit
def _scatter(x_ptr, y_ptr, valid_ptr, values_ptr, score_ptr, demo_ptr, cfa_ptr, labels_ptr,
             accum_ptr, weight_ptr, demo_accum_ptr, demo_weight_ptr, globe_ptr, ring_ptr, covered_ptr,
             total, plane, H: tl.constexpr, W: tl.constexpr, C: tl.constexpr, SCORE_MAP: tl.constexpr,
             CFA: tl.constexpr, DEMO: tl.constexpr, LAYERS: tl.constexpr, GLOBE: tl.constexpr,
             NEAR: tl.constexpr, FAR: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    inside = i < total
    frame = i // plane
    source = i % plane
    x = tl.load(x_ptr + i, inside, other=0.) - .5
    y = tl.load(y_ptr + i, inside, other=0.) - .5
    valid = (tl.load(valid_ptr + i, inside, other=0) != 0) & inside
    finite = (x == x) & (y == y) & (tl.abs(x) < float('inf')) & (tl.abs(y) < float('inf'))
    x = tl.where(finite, x, -2.)
    y = tl.where(finite, y, -2.)
    rx = libdevice.rint(x)
    ry = libdevice.rint(y)
    x = tl.where(tl.abs(x - rx) < 1e-10, rx, x)
    y = tl.where(tl.abs(y - ry) < 1e-10, ry, y)
    fx0 = libdevice.floor(x)
    fy0 = libdevice.floor(y)
    ix = fx0.to(tl.int64)
    iy = fy0.to(tl.int64)
    fx = x - fx0
    fy = y - fy0
    if not SCORE_MAP:
        scalar = tl.load(score_ptr + frame, inside, other=0.)
    if LAYERS:
        label = tl.load(labels_ptr + i, inside, other=0)
    covered = tl.zeros((BLOCK,), dtype=tl.float64)
    for dy in tl.static_range(2):
        wy = fy if dy == 1 else 1. - fy
        for dx in tl.static_range(2):
            wx = fx if dx == 1 else 1. - fx
            xx = ix + dx
            yy = iy + dy
            keep = valid & (xx >= 0) & (xx < W) & (yy >= 0) & (yy < H)
            index = tl.minimum(tl.maximum(yy, 0), H - 1) * W + tl.minimum(tl.maximum(xx, 0), W - 1)
            weight = wy * wx * keep.to(tl.float64)
            covered += weight
            if SCORE_MAP:
                weighted = weight * tl.load(score_ptr + frame * plane + index, keep, other=0.)
            else:
                weighted = weight * scalar
            add = keep & (weighted != 0.)
            if CFA:
                dest = index * 3 + tl.load(cfa_ptr + source, inside, other=0)
                tl.atomic_add(accum_ptr + dest, weighted * tl.load(values_ptr + i, inside, other=0.), add)
                tl.atomic_add(weight_ptr + dest, weighted, add)
            else:
                for c in tl.static_range(C):
                    value = tl.load(values_ptr + i * C + c, inside, other=0.)
                    tl.atomic_add(accum_ptr + index * C + c, weighted * value, add)
                    tl.atomic_add(weight_ptr + index * C + c, weighted, add)
            if DEMO:
                for c in tl.static_range(3):
                    value = tl.load(demo_ptr + i * 3 + c, inside, other=0.)
                    tl.atomic_add(demo_accum_ptr + index * 3 + c, weighted * value, add)
                    tl.atomic_add(demo_weight_ptr + index * 3 + c, weighted, add)
            if LAYERS:
                tl.atomic_add(globe_ptr + index, weighted, add & (label == GLOBE))
                tl.atomic_add(ring_ptr + index, weighted, add & ((label == NEAR) | (label == FAR)))
    tl.atomic_add(covered_ptr + frame, covered, inside)


def scatter(x, y, valid, values, score, sums, *, demo=None, cfa=None, labels=None, layer_sums=None,
            layer_codes=(0, 0, 0)):
    """Accumulate frames (B, h, w) at mapped coordinates into resident sums.

    ``values`` is (B, hw, C); ``score`` (B,) per frame or (B, hw) in destination
    geometry; ``sums`` = (accum, weight, demosaic_accum, demosaic_weight).
    Returns the bilinear coverage of each frame.
    """
    h, w = x.shape[-2:]
    n = values.shape[0]
    channels = values.shape[-1]
    x = x.expand(n, h, w).contiguous()
    y = y.expand(n, h, w).contiguous()
    valid = valid.expand(n, h, w).contiguous().to(torch.uint8)
    covered = torch.zeros(n, device=x.device, dtype=torch.float64)
    accum, weight, demo_accum, demo_weight = sums
    dummy = covered
    globe, ring = layer_sums if layer_sums is not None else (dummy, dummy)
    total, block = x.numel(), 256
    _scatter[(triton.cdiv(total, block),)](
        x, y, valid, values.contiguous(), score.contiguous(),
        dummy if demo is None else demo.contiguous(), dummy if cfa is None else cfa,
        dummy if labels is None else labels.contiguous(),
        accum, weight, dummy if demo_accum is None else demo_accum, dummy if demo_weight is None else demo_weight,
        globe, ring, covered, total, h * w, H=h, W=w, C=channels, SCORE_MAP=score.ndim == 2,
        CFA=cfa is not None, DEMO=demo is not None, LAYERS=labels is not None,
        GLOBE=layer_codes[0], NEAR=layer_codes[1], FAR=layer_codes[2], BLOCK=block, enable_fp_fusion=False)
    return covered


# Per-pose parameters of the rotating-globe mapping (TorchGlobeWarp.map), one row per pose.
SRC_CX, SRC_CY, SRC_COS, SRC_SIN, AA4, DENOM, REF_CX, REF_CY, REF_COS, REF_SIN = range(10)
RS, RR = 10, 19   # source and reference attitude matrices, row-major
PARAMETERS = 28


@triton.jit
def _globe_map(x_ptr, y_ptr, params_ptr, constants_ptr, out_x, out_y, out_valid, total, plane, W: tl.constexpr, EXPLICIT: tl.constexpr, FIELD: tl.constexpr,
               TAPER: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    inside = i < total
    p = params_ptr + (i // plane) * 28
    ia = tl.load(constants_ptr + 0)
    ic = tl.load(constants_ptr + 1)
    a2 = tl.load(constants_ptr + 2)
    c2 = tl.load(constants_ptr + 3)
    taper = tl.load(constants_ptr + 4)
    if EXPLICIT:
        x = tl.load(x_ptr + i, inside, other=0.)
        y = tl.load(y_ptr + i, inside, other=0.)
    else:
        pixel = i % plane
        x = (pixel % W).to(tl.float64) + .5
        y = (pixel // W).to(tl.float64) + .5
    sx = x - tl.load(p + 0, inside, other=0.)
    sy = tl.load(p + 1, inside, other=0.) - y
    if FIELD:
        c = tl.load(p + 2, inside, other=1.)
        s = tl.load(p + 3, inside, other=0.)
        sx, sy = c * sx - s * sy, s * sx + c * sy
    r00 = tl.load(p + 10, inside, other=0.)
    r01 = tl.load(p + 11, inside, other=0.)
    r02 = tl.load(p + 12, inside, other=0.)
    r10 = tl.load(p + 13, inside, other=0.)
    r11 = tl.load(p + 14, inside, other=0.)
    r12 = tl.load(p + 15, inside, other=0.)
    r20 = tl.load(p + 16, inside, other=0.)
    r21 = tl.load(p + 17, inside, other=0.)
    r22 = tl.load(p + 18, inside, other=0.)
    q0 = r00 * sx + r10 * sy
    q1 = r01 * sx + r11 * sy
    q2 = r02 * sx + r12 * sy
    bb = 2 * ((q0 * r20 + q1 * r21) * ia + q2 * r22 * ic)
    cc = (q0 * q0 + q1 * q1) * ia + q2 * q2 * ic - 1
    disc = bb * bb - tl.load(p + 4, inside, other=0.) * cc
    hit = disc >= 0
    t = tl.where(hit, (-bb + tl.sqrt(tl.maximum(disc, 0.))) / tl.load(p + 5, inside, other=1.), float('nan'))
    xb = q0 + t * r20
    yb = q1 + t * r21
    zb = q2 + t * r22
    nz = r20 * (xb * ia) + r21 * (yb * ia) + r22 * (zb * ic)
    mu_src = nz / tl.maximum(tl.sqrt((xb * ia) * (xb * ia) + (yb * ia) * (yb * ia) + (zb * ic) * (zb * ic)), 1e-30)
    on_globe = hit & (t == t) & (tl.abs(t) < float('inf')) & (nz >= -1e-12)
    # Radial projection onto the ellipsoid (the CPU longitude/latitude round trip).
    scale = tl.where(on_globe, tl.sqrt((xb * xb + yb * yb) / a2 + zb * zb / c2), float('nan'))
    xb = xb / scale
    yb = yb / scale
    zb = zb / scale
    rx = tl.load(p + 19, inside, other=0.) * xb + tl.load(p + 20, inside, other=0.) * yb \
        + tl.load(p + 21, inside, other=0.) * zb
    ry = tl.load(p + 22, inside, other=0.) * xb + tl.load(p + 23, inside, other=0.) * yb \
        + tl.load(p + 24, inside, other=0.) * zb
    nz = tl.load(p + 25, inside, other=0.) * (xb * ia) + tl.load(p + 26, inside, other=0.) * (yb * ia) \
        + tl.load(p + 27, inside, other=0.) * (zb * ic)
    nlen = tl.maximum(tl.sqrt((xb * ia) * (xb * ia) + (yb * ia) * (yb * ia) + (zb * ic) * (zb * ic)), 1e-30)
    mu_ref = nz / nlen
    visible = (rx == rx) & (tl.abs(rx) < float('inf')) & (mu_ref >= -1e-12)
    if FIELD:
        c = tl.load(p + 8, inside, other=1.)
        s = tl.load(p + 9, inside, other=0.)
        rx, ry = c * rx - s * ry, s * rx + c * ry
        sx, sy = c * sx - s * sy, s * sx + c * sy
    use = on_globe & visible
    cx = tl.load(p + 6, inside, other=0.)
    cy = tl.load(p + 7, inside, other=0.)
    if TAPER:
        u = tl.minimum(tl.maximum(tl.where(use, mu_src, 0.) / taper, 0.), 1.)
        u = u * u
        blend = u * u * (3. - 2. * u)
        u = tl.minimum(tl.maximum(tl.where(use, mu_ref, 0.) / taper, 0.), 1.)
        u = u * u
        blend = blend * (u * u * (3. - 2. * u))
        ox = sx + cx + blend * tl.where(blend > 0, rx - sx, 0.)
        oy = cy - sy - blend * tl.where(blend > 0, ry - sy, 0.)
        valid = (ox == ox) & (oy == oy) & (tl.abs(ox) < float('inf')) & (tl.abs(oy) < float('inf'))
    else:
        ox = tl.where(use, rx, sx) + cx
        oy = cy - tl.where(use, ry, sy)
        valid = (ox == ox) & (oy == oy) & (tl.abs(ox) < float('inf')) & (tl.abs(oy) < float('inf')) \
            & (~on_globe | visible)
    tl.store(out_x + i, ox, inside)
    tl.store(out_y + i, oy, inside)
    tl.store(out_valid + i, valid.to(tl.int8), inside)


def globe_map(params, shape, globe, *, x=None, y=None, field=True, limb_taper=None):
    """Rotating-globe mapping for B poses (``params`` is (B, PARAMETERS)): x, y, valid of (B, h, w)."""
    h, w = shape
    n = params.shape[0]
    out_x = torch.empty((n, h, w), device=params.device, dtype=torch.float64)
    out_y, out_valid = torch.empty_like(out_x), torch.empty((n, h, w), device=params.device, dtype=torch.int8)
    explicit = x is not None
    if explicit:
        x, y = x.expand(n, h, w).contiguous(), y.expand(n, h, w).contiguous()
    total, block = n * h * w, 256
    a, c = globe.equatorial_radius_px, globe.polar_radius_px
    # Float scalars would reach the kernel as fp32; constants travel as float64.
    constants = torch.tensor([1 / (a * a), 1 / (c * c), a * a, c * c, 1. if limb_taper is None else float(limb_taper)],
                             device=params.device, dtype=torch.float64)
    _globe_map[(triton.cdiv(total, block),)](
        x if explicit else out_x, y if explicit else out_y, params.contiguous(), constants, out_x, out_y, out_valid,
        total, h * w, W=w, EXPLICIT=explicit, FIELD=field, TAPER=limb_taper is not None, BLOCK=block, enable_fp_fusion=False)
    return out_x, out_y, out_valid.bool()


@triton.jit
def _clamped_bilinear(field, x, y, H: tl.constexpr, W: tl.constexpr, mask):
    """Order-one sample of one plane with edge clamping (map_coordinates mode 'nearest')."""
    x = tl.minimum(tl.maximum(x, 0.), W - 1.)
    y = tl.minimum(tl.maximum(y, 0.), H - 1.)
    x0 = libdevice.floor(x)
    y0 = libdevice.floor(y)
    fx = x - x0
    fy = y - y0
    ix = x0.to(tl.int64)
    iy = y0.to(tl.int64)
    ix1 = tl.minimum(ix + 1, W - 1)
    iy1 = tl.minimum(iy + 1, H - 1)
    f00 = tl.load(field + iy * W + ix, mask, other=0.)
    f01 = tl.load(field + iy * W + ix1, mask, other=0.)
    f10 = tl.load(field + iy1 * W + ix, mask, other=0.)
    f11 = tl.load(field + iy1 * W + ix1, mask, other=0.)
    return (1. - fy) * ((1. - fx) * f00 + fx * f01) + fy * ((1. - fx) * f10 + fx * f11)


@triton.jit
def _invert(field_ptr, qx_ptr, qy_ptr, error_ptr, total, H: tl.constexpr, W: tl.constexpr,
            STEPS: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    inside = i < total
    frame = i // (H * W)
    pixel = i % (H * W)
    x = (pixel % W).to(tl.float64)
    y = (pixel // W).to(tl.float64)
    ux = field_ptr + frame * (2 * H * W)
    uy = ux + H * W
    qx = x
    qy = y
    for _ in tl.static_range(STEPS):
        dx = _clamped_bilinear(ux, qx, qy, H, W, inside)
        dy = _clamped_bilinear(uy, qx, qy, H, W, inside)
        qx = x - dx
        qy = y - dy
    dx = _clamped_bilinear(ux, qx, qy, H, W, inside)
    dy = _clamped_bilinear(uy, qx, qy, H, W, inside)
    ex = tl.abs(qx + dx - x)
    ey = tl.abs(qy + dy - y)
    # Hardware min/max drop NaN: any non-finite residual must still fail the guard.
    finite = (ex == ex) & (ey == ey) & (ex < float('inf')) & (ey < float('inf'))
    error = tl.where(finite, tl.maximum(ex, ey), float('inf'))
    tl.store(qx_ptr + i, qx, inside)
    tl.store(qy_ptr + i, qy, inside)
    tl.store(error_ptr + i, error, inside)


def invert(field, steps=8):
    """Fixed-point inverse of pull fields (B, 2, h, w): positions qx, qy and residual errors (B, h, w)."""
    n, _, h, w = field.shape
    field = field.contiguous()
    qx = torch.empty((n, h, w), device=field.device, dtype=torch.float64)
    qy, error = torch.empty_like(qx), torch.empty_like(qx)
    total, block = n * h * w, 256
    _invert[(triton.cdiv(total, block),)](field, qx, qy, error, total, H=h, W=w, STEPS=steps, BLOCK=block,
                                          enable_fp_fusion=False)
    return qx, qy, error
