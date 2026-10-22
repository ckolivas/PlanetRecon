"""Predicted motion overlay: curved arrows from the same geometry a stack would use.

After preprocessing the Geometry controls hold the estimated centre, radius
and orientation. This builds the motion model a stack with those settings
uses, and returns illustrative arcs whose direction is the model's own
prediction, for checking by eye that rotation was predicted the right way:

- surface rotation (Surface, Combined, Saturn): an arc along a latitude on the
  visible hemisphere at the reference epoch, pointing the way surface
  features move (``body_to_sky`` at a slightly later time, as the stack's
  globe model maps them); a nearly edge-on view is drawn tilted a little
  further so the arc stays curved;
- field rotation (Field, Combined, Saturn) with a known rate: an arc outside
  the disc pointing the way the field turns (``FieldOnlyModel``).

Coordinates are detector pixel centres (x right, y down).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from planetrecon.geometry.coords import sky_to_detector
from planetrecon.geometry.globe import body_to_sky, field_rotate_sky
from planetrecon.geometry.model import FieldOnlyModel
from planetrecon.geometry.pose import FramePose, globe_for_config

ARC_POINTS = 32
SURFACE_LATITUDE_RAD = np.radians(35.)  # Clear of the equator, where Saturn's rings lie.
SURFACE_SPAN_RAD = np.radians(55.)      # Longitudes either side of the central meridian.
# A latitude seen nearly edge-on projects almost straight; draw it as if seen
# from at least this far above the equator (same hemisphere, pole angle and
# spin sense) so the arrow stays visibly curved.
MIN_DRAWN_TILT_RAD = np.radians(17.)
FIELD_SPAN_RAD = np.radians(100.)


@dataclass
class MotionArrow:
    kind: str          # 'surface' or 'field'
    points: np.ndarray  # (n, 2) detector x, y; the arrow head is at the last point
    label: str


def _per_hour(rate_rad_s):
    return f'{np.degrees(rate_rad_s) * 3600:.3g}°/h'


def _orient(points, velocity):
    """Order ``points`` so the path runs along ``velocity`` (n, 2) on average."""
    tangent = np.gradient(points, axis=0)
    return points if np.sum(tangent * velocity) >= 0 else points[::-1]


def predicted_motion(config, shape=None):
    """([MotionArrow], [notes]) for the motion a stack with ``config`` predicts.

    ``shape`` (height, width) keeps the field arc inside the detector.
    """
    mode = config.geometry_mode
    if mode == 'none':
        return [], ['No motion model: frames are aligned without predicted rotation.']
    cx, cy, radius = config.field_center_x, config.field_center_y, config.equatorial_radius_px
    if cx is None or cy is None or radius is None:
        return [], ['Predicted motion needs the disc centre and radius (Geometry tab or preprocessing).']
    arrows, notes = [], []
    if mode in ('surface', 'combined', 'saturn'):
        rate = config.effective_surface_rate()
        if not rate:
            notes.append('Surface rotation rate is unknown or zero; no surface motion is predicted.')
        else:
            sub_lat = config.sub_obs_lat_rad if config.sub_obs_lat_rad is not None else 0.
            drawn_lat = np.copysign(max(abs(sub_lat), MIN_DRAWN_TILT_RAD), sub_lat if sub_lat else 1.)
            globe = globe_for_config(radius, config.flattening, config.pole_pa_rad, drawn_lat,
                                     config.sub_obs_lon0_rad, rate, config.reference_epoch_s)
            t0 = config.reference_epoch_s
            # The hemisphere tipped toward the observer shows the longer arc.
            lat = SURFACE_LATITUDE_RAD * (1. if sub_lat >= 0 else -1.)
            lon = globe.sub_obs_lon(t0) + np.linspace(-SURFACE_SPAN_RAD, SURFACE_SPAN_RAD, ARC_POINTS)
            dt = .01 / abs(rate)  # A small fraction of a radian of spin.
            sx, sy, visible, _ = body_to_sky(lon, lat, globe, t0)
            later_x, later_y, _, _ = body_to_sky(lon, lat, globe, t0 + dt)
            if mode in ('combined', 'saturn'):
                # The reference detector sees the sky through the field attitude.
                sx, sy = field_rotate_sky(sx, sy, config.field_angle0_rad)
                later_x, later_y = field_rotate_sky(later_x, later_y, config.field_angle0_rad)
            x, y = sky_to_detector(sx, sy, cx, cy)
            lx, ly = sky_to_detector(later_x, later_y, cx, cy)
            keep = visible
            if np.count_nonzero(keep) >= 3:
                points = np.column_stack([x[keep], y[keep]])
                velocity = np.column_stack([lx[keep] - x[keep], ly[keep] - y[keep]])
                arrows.append(MotionArrow('surface', _orient(points, velocity),
                                          f'surface {_per_hour(rate)}'))
    if mode in ('field', 'combined', 'saturn'):
        rate = config.field_rate_rad_s
        if rate is None:
            notes.append('Field rotation is estimated when stacking.')
        elif rate:
            outer = radius * 1.2
            if mode == 'saturn' and config.ring_outer_radius_px is not None:
                outer = max(outer, config.ring_outer_radius_px * 1.1)
            # Above the disc, clear of the surface arrow; below it, or closer
            # in, when the detector is too small.
            side = 1.
            if shape is not None:
                margin = 8.
                room_above, room_below = cy - margin, shape[0] - cy - margin
                side = 1. if room_above >= min(outer, room_below) else -1.
                outer = max(min(outer, room_above if side > 0 else room_below), radius * 1.05)
            angles = side * np.pi / 2 + np.linspace(-FIELD_SPAN_RAD / 2, FIELD_SPAN_RAD / 2, ARC_POINTS)
            x = cx + outer * np.cos(angles)
            y = cy - outer * np.sin(angles)
            dt = .01 / abs(rate)
            src = FramePose(0., config.field_angle0_rad, cx, cy)
            ref = FramePose(dt, config.field_angle0_rad + rate * dt, cx, cy)
            later_x, later_y, _ = FieldOnlyModel().src_to_ref(x, y, src, ref)
            points = np.column_stack([x, y])
            arrows.append(MotionArrow('field', _orient(points, np.column_stack([later_x - x, later_y - y])),
                                      f'field {_per_hour(rate)}'))
    return arrows, notes
