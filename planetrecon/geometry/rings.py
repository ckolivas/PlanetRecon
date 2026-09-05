"""Equatorial ring annulus: plane intersection, depth, transmission and shadows."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from planetrecon.geometry.globe import GlobeParams, body_to_obs_matrix


EDGE_ON_MU = 0.04  # |n·los|; below this the ring plane is degenerate


@dataclass(frozen=True)
class RingParams:
    inner_radius_px: float
    outer_radius_px: float
    transmission: float = 0.35
    sun_lon_rad: float | None = None
    sun_lat_rad: float | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.inner_radius_px) or not np.isfinite(self.outer_radius_px):
            raise ValueError("ring radii must be finite")
        if self.inner_radius_px <= 0 or self.outer_radius_px <= self.inner_radius_px:
            raise ValueError("ring outer radius must exceed a positive inner radius")
        if not np.isfinite(self.transmission) or self.transmission < 0.0 or self.transmission > 1.0:
            raise ValueError("ring transmission must be in [0, 1]")
        for name in ("sun_lon_rad", "sun_lat_rad"):
            value = getattr(self, name)
            if value is not None and (not np.isfinite(value)):
                raise ValueError(f"{name} must be finite or null")
        if self.sun_lat_rad is not None and abs(self.sun_lat_rad) > np.pi / 2:
            raise ValueError("sun_lat_rad must be in [-pi/2, pi/2]")

    @property
    def optical_depth(self) -> float:
        t = min(max(float(self.transmission), 1e-12), 1.0)
        return float(-np.log(t))


def ring_normal_obs(globe: GlobeParams) -> np.ndarray:
    """Observer-frame normal of the equatorial plane. Independent of globe spin."""
    r = body_to_obs_matrix(globe.pole_pa_rad, globe.sub_obs_lat_rad, 0.0)
    return r[:, 2].copy()


def intersect_ring(
    sx,
    sy,
    globe: GlobeParams,
    rings: RingParams,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
    """Hit the equatorial annulus. Returns r, azimuth, observer-z, on_ring, edge_on."""
    sx = np.asarray(sx, dtype=np.float64)
    sy = np.asarray(sy, dtype=np.float64)
    n = ring_normal_obs(globe)
    edge_on = bool(abs(float(n[2])) < EDGE_ON_MU)
    t = np.full(sx.shape, np.nan, dtype=np.float64)
    radius = np.full(sx.shape, np.nan, dtype=np.float64)
    az = np.full(sx.shape, np.nan, dtype=np.float64)
    on = np.zeros(sx.shape, dtype=bool)
    if edge_on:
        return radius, az, t, on, True
    t = -(n[0] * sx + n[1] * sy) / n[2]
    rmat = body_to_obs_matrix(globe.pole_pa_rad, globe.sub_obs_lat_rad, 0.0)
    rt = rmat.T
    xb = rt[0, 0] * sx + rt[0, 1] * sy + rt[0, 2] * t
    yb = rt[1, 0] * sx + rt[1, 1] * sy + rt[1, 2] * t
    radius = np.hypot(xb, yb)
    az = np.arctan2(yb, xb)
    on = np.isfinite(t) & (radius >= rings.inner_radius_px) & (radius <= rings.outer_radius_px)
    radius = np.where(on, radius, np.nan)
    az = np.where(on, az, np.nan)
    t = np.where(on, t, np.nan)
    return radius, az, t, on, False


def sun_direction_obs(globe: GlobeParams, rings: RingParams) -> np.ndarray:
    """Unit vector from planet centre toward the sun, observer frame."""
    lat = globe.sub_obs_lat_rad if rings.sun_lat_rad is None else float(rings.sun_lat_rad)
    lon = 0.0 if rings.sun_lon_rad is None else float(rings.sun_lon_rad)
    body = np.array(
        [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)],
        dtype=np.float64,
    )
    r = body_to_obs_matrix(globe.pole_pa_rad, globe.sub_obs_lat_rad, 0.0)
    vec = r @ body
    nrm = float(np.linalg.norm(vec))
    if nrm <= 0:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return vec / nrm


def _spheroid_ray_u(origin_obs, direction_obs, globe: GlobeParams) -> np.ndarray:
    """Smallest u>0 where origin + u*direction hits the spheroid; nan if none."""
    a = float(globe.equatorial_radius_px)
    c = float(globe.polar_radius_px)
    r = body_to_obs_matrix(globe.pole_pa_rad, globe.sub_obs_lat_rad, 0.0)
    rt = r.T
    o = np.stack(origin_obs, axis=0)
    dobs = np.asarray(direction_obs, dtype=np.float64).reshape(3)
    pb = rt @ o.reshape(3, -1)
    db = rt @ dobs
    pb = pb.reshape(3, *o.shape[1:])
    inv_a2 = 1.0 / (a * a)
    inv_c2 = 1.0 / (c * c)
    A = (db[0] * db[0] + db[1] * db[1]) * inv_a2 + db[2] * db[2] * inv_c2
    B = 2.0 * ((pb[0] * db[0] + pb[1] * db[1]) * inv_a2 + pb[2] * db[2] * inv_c2)
    C = (pb[0] * pb[0] + pb[1] * pb[1]) * inv_a2 + pb[2] * pb[2] * inv_c2 - 1.0
    disc = B * B - 4.0 * A * C
    ok = disc >= 0.0
    sqrt_disc = np.sqrt(np.clip(disc, 0.0, None))
    den = np.maximum(2.0 * A, 1e-30)
    u1 = (-B - sqrt_disc) / den
    u2 = (-B + sqrt_disc) / den
    u = np.full(o.shape[1:], np.nan, dtype=np.float64)
    cand = np.stack([u1, u2], axis=0)
    cand = np.where(ok, cand, np.inf)
    cand = np.where(cand > 1e-4, cand, np.inf)
    u = np.min(cand, axis=0)
    return np.where(np.isfinite(u), u, np.nan)


def globe_shadow_on_ring(
    sx, sy, t_ring, on_ring, globe: GlobeParams, rings: RingParams
) -> np.ndarray:
    """True where a ring sample lies in the globe's shadow."""
    if not np.any(on_ring):
        return np.zeros(np.asarray(on_ring).shape, dtype=bool)
    sdir = sun_direction_obs(globe, rings)
    origin = (np.asarray(sx, dtype=np.float64), np.asarray(sy, dtype=np.float64), np.asarray(t_ring, dtype=np.float64))
    u = _spheroid_ray_u(origin, sdir, globe)
    return on_ring & np.isfinite(u)


def ring_shadow_on_globe(
    sx, sy, t_globe, on_globe, globe: GlobeParams, rings: RingParams
) -> np.ndarray:
    """True where a globe sample lies in the ring annulus shadow."""
    if not np.any(on_globe):
        return np.zeros(np.asarray(on_globe).shape, dtype=bool)
    sdir = sun_direction_obs(globe, rings)
    n = ring_normal_obs(globe)
    denom = float(n[0] * sdir[0] + n[1] * sdir[1] + n[2] * sdir[2])
    if abs(denom) < EDGE_ON_MU:
        return np.zeros(np.asarray(on_globe).shape, dtype=bool)
    ox = np.asarray(sx, dtype=np.float64)
    oy = np.asarray(sy, dtype=np.float64)
    oz = np.asarray(t_globe, dtype=np.float64)
    u = -(n[0] * ox + n[1] * oy + n[2] * oz) / denom
    hit = on_globe & (u > 1e-4)
    px = ox + u * sdir[0]
    py = oy + u * sdir[1]
    pz = oz + u * sdir[2]
    rmat = body_to_obs_matrix(globe.pole_pa_rad, globe.sub_obs_lat_rad, 0.0)
    rt = rmat.T
    xb = rt[0, 0] * px + rt[0, 1] * py + rt[0, 2] * pz
    yb = rt[1, 0] * px + rt[1, 1] * py + rt[1, 2] * pz
    rad = np.hypot(xb, yb)
    return hit & (rad >= rings.inner_radius_px) & (rad <= rings.outer_radius_px)


def ring_transmission(globe: GlobeParams, rings: RingParams) -> float:
    """Line-of-sight transmission, increasing toward edge-on until the degeneracy cut."""
    mu = abs(float(ring_normal_obs(globe)[2]))
    mu = max(mu, EDGE_ON_MU)
    return float(np.exp(-rings.optical_depth / mu))
