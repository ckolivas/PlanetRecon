"""Orthographic oblate globe: visibility, longitude and sky projection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from planetrecon.geometry.coords import rotate_sky


@dataclass(frozen=True)
class GlobeParams:
    equatorial_radius_px: float
    flattening: float = 0.0
    pole_pa_rad: float = 0.0
    sub_obs_lat_rad: float = 0.0
    sub_obs_lon0_rad: float = 0.0
    surface_rate_rad_s: float = 0.0
    reference_epoch_s: float = 0.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.equatorial_radius_px) or self.equatorial_radius_px <= 0:
            raise ValueError("equatorial_radius_px must be positive and finite")
        if not np.isfinite(self.flattening) or self.flattening < 0.0 or self.flattening >= 1.0:
            raise ValueError("flattening must be in [0, 1)")
        for name in (
            "pole_pa_rad",
            "sub_obs_lat_rad",
            "sub_obs_lon0_rad",
            "surface_rate_rad_s",
            "reference_epoch_s",
        ):
            if not np.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")

    @property
    def polar_radius_px(self) -> float:
        return float(self.equatorial_radius_px) * (1.0 - float(self.flattening))

    def sub_obs_lon(self, t_s: float) -> float:
        return float(self.sub_obs_lon0_rad) + float(self.surface_rate_rad_s) * (
            float(t_s) - float(self.reference_epoch_s)
        )


def body_to_obs_matrix(pole_pa_rad: float, sub_obs_lat_rad: float, sub_obs_lon_rad: float) -> np.ndarray:
    """Rotation ``p_obs = R @ p_body``. +z_obs toward the observer, +y_obs north at PA=0."""
    return (
        _rz(-float(pole_pa_rad) - 0.5 * np.pi)
        @ _ry(float(sub_obs_lat_rad) - 0.5 * np.pi)
        @ _rz(-float(sub_obs_lon_rad))
    )


def _rz(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _ry(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def _lon_lat(x: np.ndarray, y: np.ndarray, z: np.ndarray, a: float, c: float):
    lon = np.arctan2(y, x)
    lat = np.arctan2(z / c, np.hypot(x, y) / a)
    return lon, lat


def sky_to_body(
    sx,
    sy,
    globe: GlobeParams,
    t_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Intersect the near-side spheroid. Returns lon, lat, visible, emission cosine μ."""
    sx = np.asarray(sx, dtype=np.float64)
    sy = np.asarray(sy, dtype=np.float64)
    a = float(globe.equatorial_radius_px)
    c = float(globe.polar_radius_px)
    r = body_to_obs_matrix(globe.pole_pa_rad, globe.sub_obs_lat_rad, globe.sub_obs_lon(t_s))
    rt = r.T
    q = np.stack(
        [rt[0, 0] * sx + rt[0, 1] * sy, rt[1, 0] * sx + rt[1, 1] * sy, rt[2, 0] * sx + rt[2, 1] * sy],
        axis=0,
    )
    d = rt[:, 2]
    inv_a2 = 1.0 / (a * a)
    inv_c2 = 1.0 / (c * c)
    A = (d[0] * d[0] + d[1] * d[1]) * inv_a2 + d[2] * d[2] * inv_c2
    B = 2.0 * ((q[0] * d[0] + q[1] * d[1]) * inv_a2 + q[2] * d[2] * inv_c2)
    C = (q[0] * q[0] + q[1] * q[1]) * inv_a2 + q[2] * q[2] * inv_c2 - 1.0
    disc = B * B - 4.0 * A * C
    visible = disc >= 0.0
    sqrt_disc = np.sqrt(np.clip(disc, 0.0, None))
    # Near side is the larger observer-z root; p_obs_z = t.
    t_hit = np.where(visible, (-B + sqrt_disc) / np.maximum(2.0 * A, 1e-30), np.nan)
    xb = q[0] + t_hit * d[0]
    yb = q[1] + t_hit * d[1]
    zb = q[2] + t_hit * d[2]
    n_obs_z = r[2, 0] * (xb * inv_a2) + r[2, 1] * (yb * inv_a2) + r[2, 2] * (zb * inv_c2)
    visible = visible & np.isfinite(t_hit) & (n_obs_z >= -1e-12)
    lon, lat = _lon_lat(xb, yb, zb, a, c)
    mu = np.where(visible, np.clip(n_obs_z / np.maximum(np.abs(n_obs_z), 1e-30) * np.abs(n_obs_z), 0.0, None), 0.0)
    # Normalise μ by the local normal length so the limb is ~0 and the centre ~1.
    nlen = np.sqrt((xb * inv_a2) ** 2 + (yb * inv_a2) ** 2 + (zb * inv_c2) ** 2)
    mu = np.where(visible, np.clip(n_obs_z / np.maximum(nlen, 1e-30), 0.0, None), 0.0)
    lon = np.where(visible, lon, np.nan)
    lat = np.where(visible, lat, np.nan)
    return lon, lat, visible, mu


def body_to_sky(
    lon,
    lat,
    globe: GlobeParams,
    t_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Project body lon/lat to sky. Far-side points are marked not visible."""
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    a = float(globe.equatorial_radius_px)
    c = float(globe.polar_radius_px)
    clat = np.cos(lat)
    xb = a * clat * np.cos(lon)
    yb = a * clat * np.sin(lon)
    zb = c * np.sin(lat)
    r = body_to_obs_matrix(globe.pole_pa_rad, globe.sub_obs_lat_rad, globe.sub_obs_lon(t_s))
    xo = r[0, 0] * xb + r[0, 1] * yb + r[0, 2] * zb
    yo = r[1, 0] * xb + r[1, 1] * yb + r[1, 2] * zb
    zo = r[2, 0] * xb + r[2, 1] * yb + r[2, 2] * zb
    inv_a2 = 1.0 / (a * a)
    inv_c2 = 1.0 / (c * c)
    n_obs_z = r[2, 0] * (xb * inv_a2) + r[2, 1] * (yb * inv_a2) + r[2, 2] * (zb * inv_c2)
    nlen = np.sqrt((xb * inv_a2) ** 2 + (yb * inv_a2) ** 2 + (zb * inv_c2) ** 2)
    mu = n_obs_z / np.maximum(nlen, 1e-30)
    visible = np.isfinite(xo) & (mu >= -1e-12)
    mu = np.where(visible, np.clip(mu, 0.0, None), 0.0)
    return xo, yo, visible, mu


def field_rotate_sky(sx, sy, field_angle_rad: float):
    """Sky-to-detector attitude: inertial sky rotated onto the detector sky axes."""
    return rotate_sky(sx, sy, field_angle_rad)


def render_globe_texture(
    height: int,
    width: int,
    cx: float,
    cy: float,
    field_angle_rad: float,
    globe: GlobeParams,
    t_s: float,
    tex_fn,
    *,
    apply_field: bool = True,
    limb_weight: bool = True,
) -> np.ndarray:
    """Forward V: evaluate a body-fixed texture on the detector at time ``t_s``."""
    from planetrecon.geometry.coords import detector_to_sky, detector_xy_grids

    x, y = detector_xy_grids(height, width)
    sx, sy = detector_to_sky(x, y, cx, cy)
    if apply_field:
        sx, sy = field_rotate_sky(sx, sy, -float(field_angle_rad))
    lon, lat, visible, mu = sky_to_body(sx, sy, globe, t_s)
    img = np.zeros((int(height), int(width)), dtype=np.float64)
    if np.any(visible):
        val = np.asarray(tex_fn(lon[visible], lat[visible]), dtype=np.float64)
        if limb_weight:
            val = val * (0.35 + 0.65 * mu[visible])
        img[visible] = val
    return img
