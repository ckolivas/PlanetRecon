"""Arrows pointing to the moons a stack identified and stacked on their own positions.

Moon stacking records each moon's output position (pixel indices, y then x) as soon as
the moons are found, so progress previews and the final result can both show them.
Each arrow approaches its moon from the side away from the planet, stays inside the
frame, and stops short of the moon so its light stays visible.

Coordinates are detector pixel centres (x right, y down), as for the motion arrows.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Arrow length and minimum gap, as fractions of the image's shorter side.
LENGTH_FRACTION = .09
GAP_FRACTION = .02
MIN_LENGTH_PX = 12.
MIN_GAP_PX = 4.


@dataclass
class MoonMarker:
    points: np.ndarray  # (2, 2) detector x, y: tail then tip (the arrow head)
    position: tuple     # The moon's detector x, y
    label: str


def moon_positions(provenance):
    """[(x, y, fwhm or None)] in detector pixel centres from a result's moon provenance."""
    moons = (provenance or {}).get('moons') or {}
    entries = moons.get('moons') or moons.get('positions_px') or []
    out = []
    for entry in entries:
        position = entry.get('output_position_px')
        if position is None or len(position) != 2 or not np.all(np.isfinite(position)):
            continue
        y, x = (float(v) for v in position)
        out.append((x + .5, y + .5, entry.get('planet_registered_fwhm_px')))
    return out


def _planet_centre(image):
    """The brightness-weighted centre of what stands clearly above the sky (detector x, y)."""
    plane = np.asarray(image, dtype=np.float64)
    if plane.ndim == 3:
        plane = plane.mean(-1)
    finite = np.isfinite(plane)
    if not finite.any():
        return None
    values = plane[finite]
    sky, peak = np.percentile(values, 10), np.percentile(values, 99.5)
    weight = np.where(finite, plane - sky - .25 * (peak - sky), 0.).clip(0)
    total = weight.sum()
    if not total > 0:
        return None
    y, x = np.indices(plane.shape)
    return float((weight * (x + .5)).sum() / total), float((weight * (y + .5)).sum() / total)


def moon_markers(provenance, image, stride=1):
    """([MoonMarker], [notes]) for the moons recorded in ``provenance``.

    ``image`` is the displayed (possibly strided) image; ``stride`` maps its pixels to the
    detector's. Positions and arrows are in detector pixel centres.
    """
    moons = (provenance or {}).get('moons') or {}
    positions = moon_positions(provenance)
    notes = []
    if moons.get('skipped'):
        notes.append(f"Moons not stacked separately: {moons['skipped']}.")
    if not positions:
        return [], notes
    h, w = (np.asarray(image).shape[0] * stride, np.asarray(image).shape[1] * stride)
    side = min(h, w)
    length = max(LENGTH_FRACTION * side, MIN_LENGTH_PX)
    centre = _planet_centre(image)
    if centre is not None:
        centre = ((centre[0] - .5) * stride + .5, (centre[1] - .5) * stride + .5)
    markers = []
    for number, (x, y, fwhm) in enumerate(positions, 1):
        gap = max(GAP_FRACTION * side, MIN_GAP_PX, 1.5 * fwhm if fwhm else 0.)
        outward = (np.arctan2(y - centre[1], x - centre[0]) if centre is not None
                   and np.hypot(x - centre[0], y - centre[1]) > 1e-6 else -np.pi / 2)
        # Away from the planet first; turn progressively until the whole arrow fits.
        chosen = None
        for turn in np.radians([0, 30, -30, 60, -60, 90, -90, 120, -120, 150, -150, 180]):
            angle = outward + turn
            ux, uy = np.cos(angle), np.sin(angle)
            tail = (x + (gap + length) * ux, y + (gap + length) * uy)
            if 0 <= tail[0] <= w and 0 <= tail[1] <= h:
                chosen = (tail, (x + gap * ux, y + gap * uy))
                break
        if chosen is None:  # A tiny frame: point along the outward direction regardless.
            ux, uy = np.cos(outward), np.sin(outward)
            chosen = ((x + (gap + length) * ux, y + (gap + length) * uy), (x + gap * ux, y + gap * uy))
        markers.append(MoonMarker(np.array(chosen, dtype=np.float64), (x, y), f'moon {number}'))
    stacked = moons.get('stacked', len(positions))
    notes.append(f'Moons: {stacked} stacked on their own positions (green arrows).')
    return markers, notes
