"""Optional fused float64 weighted correlations; no candidate tensor expansion."""
import torch
import triton
import triton.language as tl


@triton.jit
def _correlate(image, ys, xs, templates, strengths, weights, output,
               WIDTH: tl.constexpr, WINDOW: tl.constexpr, BLOCK: tl.constexpr):
    point = tl.program_id(0)
    offset = tl.program_id(1)
    k = tl.arange(0, BLOCK)
    valid = k < WINDOW*WINDOW
    y = tl.load(ys+point)+k//WINDOW-WINDOW//2+offset//7-3
    x = tl.load(xs+point)+k % WINDOW-WINDOW//2+offset % 7-3
    value = tl.load(image+y*WIDTH+x, valid, other=0.)
    weight = tl.load(weights+k, valid, other=0.)
    template = tl.load(templates+point*WINDOW*WINDOW+k, valid, other=0.)
    mean = tl.sum(value*weight, axis=0)
    centred = value-mean
    variance = tl.sum(centred*centred*weight, axis=0)
    numerator = tl.sum(centred*template*weight, axis=0)
    denominator = tl.maximum(tl.sqrt(variance)*tl.load(strengths+point), 1e-15)
    tl.store(output+point*49+offset, numerator/denominator)


def scores(image, ys, xs, templates, strengths, weights):
    window = weights.shape[0]
    result = torch.empty((len(ys), 7, 7), device=image.device, dtype=torch.float64)
    _correlate[(len(ys), 49)](image, ys, xs, templates, strengths, weights, result,
        WIDTH=image.shape[1], WINDOW=window, BLOCK=triton.next_power_of_2(window*window),
        num_warps=4 if window*window <= 2048 else (8 if window*window <= 8192 else 16),
        enable_fp_fusion=False)
    return result
