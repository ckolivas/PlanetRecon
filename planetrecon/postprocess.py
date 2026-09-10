"""Explicit result adjustments; reconstruction and detector data stay untouched."""
from copy import deepcopy

import numpy as np
from scipy.ndimage import shift


def align_rgb(result, red=(0., 0.), blue=(0., 0.)):
    """Move R/B content in detector pixels (right/down positive), keeping G fixed.

    Call against the original result for each preview, never a previous preview.
    Bilinear resampling adds no sharpening. A sample is valid only when its whole
    interpolation footprint was supported; weights are transported, not new data.
    """
    image = np.asarray(result.image)
    if (result.incomplete or result.spatial_stride != 1 or
            result.channel_order != 'RGB' or image.ndim != 3 or image.shape[-1] != 3):
        raise ValueError('RGB alignment requires a complete full-resolution RGB result')
    offsets = np.asarray([red, (0., 0.), blue], dtype=float)
    if offsets.shape != (3, 2) or not np.isfinite(offsets).all() or np.any(np.abs(offsets) > 16):
        raise ValueError('RGB offsets must be finite x/y pairs within 16 pixels')
    if result.provenance.get('rgb_alignment'):
        raise ValueError('Apply RGB alignment to the original result to avoid repeated resampling')
    adjusted = deepcopy(result)
    if not offsets.any():
        return adjusted

    def channels(array):
        value = np.asarray(array)
        if value.shape == image.shape[:2]:
            return np.broadcast_to(value[..., None], image.shape)
        if value.shape != image.shape:
            raise ValueError('RGB support arrays must match the image')
        return value

    valid = channels(result.validity) & np.isfinite(image)
    coverage = channels(result.coverage)
    valid &= np.isfinite(coverage) & (coverage > 0)
    adjusted.image = np.zeros_like(image, dtype=np.float64)
    adjusted.coverage = np.zeros_like(image, dtype=np.float64)
    adjusted.validity = np.zeros_like(image, dtype=bool)
    adjusted.layer_coverage = {}
    for c, (dx, dy) in enumerate(offsets):
        def move(array):
            return (array.copy() if dx == dy == 0 else
                    shift(array, (dy, dx), order=1, mode='grid-constant', prefilter=False))
        support = move(valid[..., c].astype(float))
        keep = support >= 1-1e-12
        adjusted.image[..., c] = np.where(keep, move(np.where(valid[..., c], image[..., c], 0.)), 0.)
        adjusted.coverage[..., c] = np.where(keep, move(np.where(valid[..., c], coverage[..., c], 0.)), 0.)
        adjusted.validity[..., c] = keep
    for name, original in result.layer_coverage.items():
        if original.shape != image.shape[:2]:
            raise ValueError('Named layer coverage must be a spatial map')
        if name in ('cfa_direct_R', 'cfa_direct_G', 'cfa_direct_B'):
            targets = [(name, 'RGB'.index(name[-1]))]
        else:
            # Shared spatial layers become different maps for each colour.
            # Keep the original name on fixed green for existing coverage views.
            targets = [(name, 1), ('rgb_alignment_R_'+name, 0), ('rgb_alignment_B_'+name, 2)]
        for target, c in targets:
            if target != name and target in result.layer_coverage:
                raise ValueError('RGB alignment coverage name conflicts with an existing layer')
            dx, dy = offsets[c]
            translated = (original.copy() if dx == dy == 0 else
                          shift(original, (dy, dx), order=1, mode='grid-constant', prefilter=False))
            adjusted.layer_coverage[target] = np.where(adjusted.validity[..., c], translated, 0.)
    adjusted.provenance['rgb_alignment'] = {
        'offsets_xy_px': {label: pair.tolist() for label, pair in zip('RGB', offsets)},
        'direction': 'positive x moves content right; positive y moves content down',
        'method': 'user-selected bilinear channel translation; green fixed; no sharpening',
        'coverage': 'transported reconstruction weights, not new observations or calibrated uncertainty',
        'layer_coverage': 'direct CFA names follow their colour; shared layer names follow green, with rgb_alignment_R_ and rgb_alignment_B_ maps for moved colours',
        'selection': 'manual; offsets do not establish atmospheric dispersion or physical geometry',
    }
    return adjusted
