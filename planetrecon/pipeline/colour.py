"""Complete registered CFA planes into an RGB reconstruction."""
from __future__ import annotations

import numpy as np

from planetrecon.detector import is_bayer


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
