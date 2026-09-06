"""Display-only sampling of reconstructed Bayer colour planes."""
import numpy as np

from planetrecon.detector import is_bayer


def display_result_preview(result, max_side=512):
    """Copy nearest supported RGB samples before reducing a Bayer display.

    Reconstructed channels need not retain the original CFA phase after shifts.
    Search each channel's actual support within one detector pixel, including
    diagonals. Larger unsupported regions stay masked. Never modify or export
    these display samples as the full-resolution scientific result.
    """
    preview = result.copy_preview(max_side)
    color = result.provenance.get('color_mode', result.provenance.get('source', {}).get('color_mode'))
    if not is_bayer(color) or result.image.ndim != 3:
        return preview
    step = preview.spatial_stride // result.spatial_stride
    h, w = result.image.shape[:2]
    y, x = np.arange(0, h, step), np.arange(0, w, step)
    filled = np.zeros(preview.image.shape, dtype=bool)
    # Increasing distance, deterministic ties. Read from the original lattice:
    # reducing first can discard every valid green site for an even stride.
    offsets = sorted(((dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)),
                     key=lambda offset: (offset[0]**2 + offset[1]**2, offset))
    for dy, dx in offsets:
        yy, xx = y+dy, x+dx
        inside = ((yy >= 0) & (yy < h))[:, None] & ((xx >= 0) & (xx < w))[None, :]
        yy, xx = np.clip(yy, 0, h-1), np.clip(xx, 0, w-1)
        samples = result.image[yy[:, None], xx[None, :]]
        weights = result.coverage[yy[:, None], xx[None, :]]
        valid = result.validity[yy[:, None], xx[None, :]]
        take = (~filled & inside[..., None] & valid & np.isfinite(samples)
                & np.isfinite(weights) & (weights > 0))
        preview.image[take] = samples[take]
        preview.coverage[take] = weights[take]
        filled |= take
    preview.validity = filled
    preview.provenance['display_sampling'] = 'nearest supported Bayer RGB within one pixel before reduction'
    return preview
