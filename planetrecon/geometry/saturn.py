"""Saturn layered scene: globe, near/far rings, moons and illumination masks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from planetrecon.geometry.coords import detector_to_sky, detector_xy_grids, sky_to_detector
from planetrecon.geometry.fit import fit_disc_ellipse
from planetrecon.geometry.globe import GlobeParams, field_rotate_sky, globe_hit, render_globe_texture
from planetrecon.geometry.model import FieldOnlyModel, OblateGlobeModel, SceneModel
from planetrecon.geometry.pose import FramePose
from planetrecon.geometry.rings import (
    EDGE_ON_MU,
    RingParams,
    globe_shadow_on_ring,
    intersect_ring,
    ring_normal_obs,
    ring_shadow_on_globe,
    ring_transmission,
)


LAYER_BACKGROUND = 0
LAYER_FAR_RING = 1
LAYER_GLOBE = 2
LAYER_NEAR_RING = 3
LAYER_MOON = 4


@dataclass(frozen=True)
class MoonTrack:
    x: float
    y: float
    radius_px: float
    vx_px_s: float = 0.0
    vy_px_s: float = 0.0

    def __post_init__(self) -> None:
        for name in ("x", "y", "radius_px", "vx_px_s", "vy_px_s"):
            if not np.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if self.radius_px <= 0:
            raise ValueError("moon radius must be positive")


def moon_mask(
    x, y, pose: FramePose, moon: MoonTrack, t_ref: float, field_ref_rad: float = 0.0
) -> np.ndarray:
    """Detector mask. ``moon.x/y`` are detector coordinates at the reference epoch."""
    dt = float(pose.t_s) - float(t_ref)
    sx0, sy0 = detector_to_sky(moon.x, moon.y, pose.cx, pose.cy)
    sx0, sy0 = field_rotate_sky(sx0, sy0, -float(field_ref_rad))
    sx = sx0 + moon.vx_px_s * dt
    sy = sy0 + moon.vy_px_s * dt
    rx, ry = field_rotate_sky(sx, sy, pose.field_angle_rad)
    mx, my = sky_to_detector(rx, ry, pose.cx, pose.cy)
    return np.hypot(np.asarray(x) - mx, np.asarray(y) - my) <= moon.radius_px


def classify_layers(
    sx,
    sy,
    globe: GlobeParams,
    rings: RingParams,
    t_s: float,
) -> dict:
    lon, lat, on_globe, mu, t_globe = globe_hit(sx, sy, globe, t_s)
    radius, az, t_ring, on_ring, edge_on = intersect_ring(sx, sy, globe, rings)
    near = np.zeros(np.asarray(sx).shape, dtype=bool)
    far = np.zeros_like(near)
    if not edge_on:
        both = on_globe & on_ring
        near = on_ring & (~on_globe | (both & (t_ring >= t_globe)))
        far = on_ring & ~near
        if np.any(both & ~np.isfinite(t_ring)):
            far = far | (both & ~np.isfinite(t_ring))
    labels = np.full(np.asarray(sx).shape, LAYER_BACKGROUND, dtype=np.int8)
    labels = np.where(far, LAYER_FAR_RING, labels)
    labels = np.where(on_globe, LAYER_GLOBE, labels)
    labels = np.where(near, LAYER_NEAR_RING, labels)
    return {
        "labels": labels,
        "lon": lon,
        "lat": lat,
        "mu": mu,
        "t_globe": t_globe,
        "t_ring": t_ring,
        "ring_radius": radius,
        "ring_azimuth": az,
        "on_globe": on_globe,
        "on_ring": on_ring,
        "near_ring": near,
        "far_ring": far,
        "edge_on": edge_on,
        "globe_shadow": globe_shadow_on_ring(sx, sy, t_ring, on_ring, globe, rings),
        "ring_shadow": ring_shadow_on_globe(sx, sy, t_globe, on_globe, globe, rings),
    }


class SaturnSceneModel(SceneModel):
    """Globe spin, static rings, field attitude on the composite, moons masked."""

    def __init__(
        self,
        globe: GlobeParams,
        rings: RingParams,
        *,
        moon: MoonTrack | None = None,
        apply_field: bool = True,
        apply_surface: bool = True,
        field_angle0_rad: float = 0.0,
    ):
        self.globe = globe
        self.rings = rings
        self.moon = moon
        self.apply_field = bool(apply_field)
        self.apply_surface = bool(apply_surface)
        self.field_angle0_rad = float(field_angle0_rad)
        self.globe_model = OblateGlobeModel(globe, apply_field=apply_field, apply_surface=apply_surface)
        self.field_model = FieldOnlyModel()
        n_z = abs(float(ring_normal_obs(globe)[2]))
        self.edge_on = n_z < EDGE_ON_MU
        self.low_opening = (not self.edge_on) and n_z < 0.2
        self.transmission = 0.0 if self.edge_on else ring_transmission(globe, rings)

    def classify_detector(self, x, y, pose: FramePose) -> dict:
        sx, sy = detector_to_sky(x, y, pose.cx, pose.cy)
        if self.apply_field:
            sx, sy = field_rotate_sky(sx, sy, -pose.field_angle_rad)
        t = pose.t_s if self.apply_surface else self.globe.reference_epoch_s
        info = classify_layers(sx, sy, self.globe, self.rings, t)
        if self.moon is not None:
            moon = moon_mask(
                x, y, pose, self.moon, self.globe.reference_epoch_s, self.field_angle0_rad
            )
            info["labels"] = np.where(moon, LAYER_MOON, info["labels"])
            info["moon"] = moon
        else:
            info["moon"] = np.zeros(np.asarray(x).shape, dtype=bool)
        return info

    def src_to_ref(self, x, y, src: FramePose, ref: FramePose):
        info = self.classify_detector(x, y, src)
        gx, gy, gv = self.globe_model.src_to_ref(x, y, src, ref)
        fx, fy, fv = self.field_model.src_to_ref(x, y, src, ref)
        globe = info["labels"] == LAYER_GLOBE
        moon = info["labels"] == LAYER_MOON
        dx = np.where(globe, gx, fx)
        dy = np.where(globe, gy, fy)
        valid = np.where(moon, False, np.where(globe, gv, fv))
        if self.edge_on:
            ring = (info["labels"] == LAYER_NEAR_RING) | (info["labels"] == LAYER_FAR_RING)
            valid = valid & ~ring
        elif self.low_opening:
            overlap = info["on_globe"] & info["on_ring"] & ~moon
            valid = valid & ~overlap
        return dx, dy, valid


def fit_saturn_geometry(image: np.ndarray) -> dict:
    """Globe from the inner disc; rings from the outer bright annulus. Ansae do not set the globe radius."""
    base = fit_disc_ellipse(image)
    if not base["ok"]:
        return {**base, "opening_rad": 0.0, "ring_inner": None, "ring_outer": None}
    img = np.asarray(image, dtype=np.float64)
    if img.ndim == 3:
        img = 0.5 * img[..., 1] + 0.25 * img[..., 0] + 0.25 * img[..., 2]
    finite = img[np.isfinite(img)]
    sky = float(np.percentile(finite, 10))
    peak = float(np.percentile(finite, 99))
    mask = np.isfinite(img) & (img > sky + 0.25 * (peak - sky))
    yy, xx = np.indices(img.shape, dtype=np.float64)
    cx, cy = float(base["cx"]), float(base["cy"])
    rr = np.hypot(xx + 0.5 - cx, yy + 0.5 - cy)
    r_bright = rr[mask]
    if r_bright.size < 9:
        return {**base, "opening_rad": 0.0, "ring_inner": None, "ring_outer": None}
    r95 = float(np.percentile(r_bright, 95))
    core = mask & (rr < 0.5 * r95)
    if int(core.sum()) < 9:
        core = mask & (rr < 0.7 * r95)
    globe = fit_disc_ellipse(np.where(core, img, sky))
    if globe["ok"]:
        out = dict(globe)
    else:
        out = dict(base)
    band = mask & (rr > 0.65 * r95)
    degeneracy = list(out.get("degeneracy") or ())
    opening = 0.0
    inner = None
    outer = None
    if int(band.sum()) >= 20:
        ring = fit_disc_ellipse(np.where(band, img, sky))
        if ring["ok"] and ring["semi_major"] > 1e-6:
            ratio = float(np.clip(ring["semi_minor"] / ring["semi_major"], 0.0, 1.0))
            opening = float(np.arcsin(ratio))
            inner = float(out["radius"]) * 1.12
            outer = float(ring["semi_major"])
            if ratio < EDGE_ON_MU:
                degeneracy.append("edge_on_rings")
            elif ratio < 0.2:
                degeneracy.append("low_opening")
    out["degeneracy"] = tuple(dict.fromkeys(degeneracy))
    out["opening_rad"] = opening
    out["ring_inner"] = inner
    out["ring_outer"] = outer
    out["outer_bright_radius"] = r95
    return out


def default_ring_profile(radius, azimuth, inner: float, outer: float):
    x = (np.asarray(radius, dtype=np.float64) - inner) / max(outer - inner, 1e-6)
    return 0.25 + 0.85 * np.exp(-((x - 0.35) / 0.22) ** 2)


def render_saturn(
    height: int,
    width: int,
    pose: FramePose,
    globe: GlobeParams,
    rings: RingParams,
    globe_tex,
    *,
    ring_tex=None,
    moon: MoonTrack | None = None,
    apply_field: bool = True,
) -> np.ndarray:
    """Composite far ring, globe, near ring. Globe spin does not move the rings."""
    x, y = detector_xy_grids(height, width)
    sx, sy = detector_to_sky(x, y, pose.cx, pose.cy)
    if apply_field:
        sx, sy = field_rotate_sky(sx, sy, -float(pose.field_angle_rad))
    info = classify_layers(sx, sy, globe, rings, pose.t_s)
    img = np.zeros((int(height), int(width)), dtype=np.float64)
    tex_r = ring_tex if ring_tex is not None else (
        lambda radius, azimuth: default_ring_profile(radius, azimuth, rings.inner_radius_px, rings.outer_radius_px)
    )
    trans = 0.0 if info["edge_on"] else ring_transmission(globe, rings)
    if np.any(info["far_ring"]):
        val = np.asarray(
            tex_r(info["ring_radius"][info["far_ring"]], info["ring_azimuth"][info["far_ring"]]),
            dtype=np.float64,
        )
        val = np.where(info["globe_shadow"][info["far_ring"]], 0.05 * val, val)
        img[info["far_ring"]] = val
    globe_img = render_globe_texture(
        height, width, pose.cx, pose.cy, pose.field_angle_rad, globe, pose.t_s, globe_tex,
        apply_field=apply_field, limb_weight=True,
    )
    if np.any(info["ring_shadow"]):
        globe_img = globe_img.copy()
        globe_img[info["ring_shadow"]] *= 0.15
    img = np.where(info["on_globe"], globe_img, img)
    if np.any(info["near_ring"]):
        near_vals = np.zeros_like(img)
        val = np.asarray(
            tex_r(info["ring_radius"][info["near_ring"]], info["ring_azimuth"][info["near_ring"]]),
            dtype=np.float64,
        )
        val = np.where(info["globe_shadow"][info["near_ring"]], 0.05 * val, val)
        near_vals[info["near_ring"]] = val
        overlap = info["near_ring"] & info["on_globe"]
        near_only = info["near_ring"] & ~info["on_globe"]
        img = np.where(near_only, near_vals, img)
        img = np.where(overlap, near_vals + trans * img, img)
    if moon is not None:
        m = moon_mask(x, y, pose, moon, globe.reference_epoch_s)
        img = np.where(m, 1.35, img)
    return img
