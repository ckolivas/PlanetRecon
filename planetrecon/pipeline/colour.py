"""Complete registered CFA planes into an RGB reconstruction."""
from __future__ import annotations

import numpy as np

from planetrecon.detector import is_bayer

SPARSE_FRACTION = .02  # Below this share of incomplete pixels, fill them individually.


def complete_bayer_rgb(result, regions=None):
    """Fill missing colours from nearest direct support within one pixel.

    Never propagate filled values, expand the observed footprint, or cross a
    supplied geometry region boundary. Coverage at an inferred channel is the
    weight of its source sample, not an additional independent observation.
    Direct channel weights remain available as named spatial coverage layers.
    Accumulator arrays are untouched, so checkpoints retain raw CFA sums.
    """
    color = result.provenance.get('color_mode')
    if not is_bayer(color) or 'rgb_completion' in result.provenance:
        return result
    image = result.image
    direct = result.coverage
    supported = result.validity & np.isfinite(image) & (direct > 0)
    footprint = np.any(supported, axis=2)
    result.coverage = direct.copy()
    result.validity = supported.copy()
    for c, name in enumerate('RGB'):
        result.layer_coverage[f'cfa_direct_{name}'] = direct[..., c].copy()
    h, w = image.shape[:2]
    offsets = sorted(((dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                      if dy or dx), key=lambda p: (p[0]**2 + p[1]**2, p))
    # Only footprint pixels missing a colour can change. When they are few, fill just them
    # (same offsets, order and sources); otherwise sweep whole shifted planes.
    py, px = np.nonzero(footprint & ~result.validity.all(axis=2))
    if len(py) <= SPARSE_FRACTION * h * w:
        valid = result.validity[py, px]
        values, weights = image[py, px], result.coverage[py, px]
        for dy, dx in offsets:
            sy, sx = py + dy, px + dx
            inside = (sy >= 0) & (sy < h) & (sx >= 0) & (sx < w)
            sy, sx = np.clip(sy, 0, h - 1), np.clip(sx, 0, w - 1)
            eligible = inside
            if regions is not None:
                eligible = eligible & (regions[py, px] == regions[sy, sx])
            # Only direct samples are eligible sources, including after earlier fills.
            take = ~valid & supported[sy, sx] & eligible[:, None]
            np.copyto(values, image[sy, sx], where=take)
            np.copyto(weights, direct[sy, sx], where=take)
            valid |= take
        image[py, px] = values
        result.coverage[py, px] = weights
        result.validity[py, px] = valid
    else:
        for dy, dx in offsets:
            target = (slice(max(0, -dy), min(h, h-dy)),
                      slice(max(0, -dx), min(w, w-dx)))
            source = (slice(max(0, dy), min(h, h+dy)),
                      slice(max(0, dx), min(w, w+dx)))
            eligible = footprint[target]
            if regions is not None:
                eligible = eligible & (regions[target] == regions[source])
            take = (~result.validity[target] & supported[source] & eligible[..., None])
            # Only direct samples are eligible sources, including after earlier fills.
            np.copyto(image[target], image[source], where=take)
            np.copyto(result.coverage[target], direct[source], where=take)
            result.validity[target] |= take
    result.provenance['rgb_completion'] = {
        'method': 'nearest supported channel within one pixel (including diagonals)',
        'coverage': 'source sample weight; direct weights in cfa_direct_R/G/B layers',
        'filled_channel_samples': int(np.count_nonzero(result.validity & ~supported)),
        'preserves_observed_footprint': True,
        'respects_geometry_regions': regions is not None,
    }
    return result
