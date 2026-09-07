"""Composable scene maps: field attitude and globe visibility as separate operators."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from planetrecon.geometry.coords import detector_to_sky, detector_xy_grids, sky_to_detector
from planetrecon.geometry.globe import GlobeParams, body_to_sky, field_rotate_sky, sky_to_body
from planetrecon.geometry.pose import FramePose
from planetrecon.geometry.warp import bilinear_sample
from planetrecon.operators import adjoint_relative_error


class SceneModel(ABC):
    @abstractmethod
    def src_to_ref(
        self, x, y, src: FramePose, ref: FramePose
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Map source detector pixel centres to the reference detector. Third array is valid."""
        raise NotImplementedError

    def ref_to_src(
        self, x, y, src: FramePose, ref: FramePose
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Inverse map: reference pixels to the source detector."""
        return self.src_to_ref(x, y, ref, src)


class FieldOnlyModel(SceneModel):
    """Sky-to-detector attitude. No surface motion."""

    def src_to_ref(self, x, y, src: FramePose, ref: FramePose):
        sx, sy = detector_to_sky(x, y, src.cx, src.cy)
        rx, ry = field_rotate_sky(sx, sy, ref.field_angle_rad - src.field_angle_rad)
        dx, dy = sky_to_detector(rx, ry, ref.cx, ref.cy)
        valid = np.isfinite(dx) & np.isfinite(dy)
        return dx, dy, valid


class OblateGlobeModel(SceneModel):
    """Rigid oblate globe. Surface rotation changes visible longitude; field is optional."""

    def __init__(self, globe: GlobeParams, *, apply_field: bool = True, apply_surface: bool = True):
        self.globe = globe
        self.apply_field = bool(apply_field)
        self.apply_surface = bool(apply_surface)

    def src_to_ref(self, x, y, src: FramePose, ref: FramePose):
        sx, sy = detector_to_sky(x, y, src.cx, src.cy)
        if not self.apply_surface or self.globe.surface_rate_rad_s * (src.t_s - ref.t_s) == 0:
            # No longitude change: avoid a lossy sphere round trip, especially
            # the tiny fractional CFA weights it creates at unsampled colours.
            angle = ref.field_angle_rad - src.field_angle_rad if self.apply_field else 0.
            rx, ry = field_rotate_sky(sx, sy, angle)
            dx, dy = sky_to_detector(rx, ry, ref.cx, ref.cy)
            return dx, dy, np.isfinite(dx) & np.isfinite(dy)
        if self.apply_field:
            sky_x, sky_y = field_rotate_sky(sx, sy, -src.field_angle_rad)
        else:
            sky_x, sky_y = sx, sy
        t_src = src.t_s if self.apply_surface else self.globe.reference_epoch_s
        t_ref = ref.t_s if self.apply_surface else self.globe.reference_epoch_s
        lon, lat, on_globe, _mu = sky_to_body(sky_x, sky_y, self.globe, t_src)
        rx_g, ry_g, vis_ref, _ = body_to_sky(lon, lat, self.globe, t_ref)
        if self.apply_field:
            det_sx, det_sy = field_rotate_sky(rx_g, ry_g, ref.field_angle_rad)
            bg_sx, bg_sy = field_rotate_sky(sky_x, sky_y, ref.field_angle_rad)
        else:
            det_sx, det_sy = rx_g, ry_g
            bg_sx, bg_sy = sky_x, sky_y
        use_globe = on_globe & vis_ref
        out_sx = np.where(use_globe, det_sx, bg_sx)
        out_sy = np.where(use_globe, det_sy, bg_sy)
        dx, dy = sky_to_detector(out_sx, out_sy, ref.cx, ref.cy)
        valid = np.isfinite(dx) & np.isfinite(dy) & np.where(on_globe, vis_ref, True)
        return dx, dy, valid


def select_scene_model(
    mode: str, globe: GlobeParams | None, rings=None, moon=None, field_angle0_rad: float = 0.0
) -> SceneModel:
    if mode == "none":
        raise ValueError("scene model is not used when geometry_mode is none")
    if mode == "field":
        return FieldOnlyModel()
    if globe is None:
        raise ValueError("surface/combined/saturn geometry requires globe parameters")
    if mode == "surface":
        return OblateGlobeModel(globe, apply_field=False, apply_surface=True)
    if mode == "combined":
        return OblateGlobeModel(globe, apply_field=True, apply_surface=True)
    if mode == "saturn":
        from planetrecon.geometry.saturn import SaturnSceneModel

        if rings is None:
            raise ValueError("saturn geometry requires ring inner/outer radii")
        return SaturnSceneModel(
            globe, rings, moon=moon, apply_field=True, apply_surface=True,
            field_angle0_rad=field_angle0_rad,
        )
    raise ValueError(f"unknown geometry_mode {mode!r}")


def compose_src_to_ref(
    mode: str,
    globe: GlobeParams | None,
    x,
    y,
    src: FramePose,
    ref: FramePose,
):
    return select_scene_model(mode, globe).src_to_ref(x, y, src, ref)


def render_observed(
    reference: np.ndarray,
    model: SceneModel,
    src: FramePose,
    ref: FramePose,
) -> np.ndarray:
    """Pull the reference-epoch scene into a source-time detector image."""
    h, w = reference.shape[:2]
    x, y = detector_xy_grids(h, w)
    sx, sy, valid = model.src_to_ref(x, y, src, ref)
    sampled = bilinear_sample(reference, sy - 0.5, sx - 0.5, fill=0.0)
    if sampled.ndim == 3:
        return np.where(valid[..., None], sampled, 0.0)
    return np.where(valid, sampled, 0.0)


def warp_adjoint_error(
    model: SceneModel,
    src: FramePose,
    ref: FramePose,
    shape: tuple[int, int],
    rng: np.random.Generator,
) -> float:
    """Relative adjoint error of dest-driven sampling through the composed map."""
    from planetrecon.geometry.warp import bilinear_sample_adjoint

    h, w = shape
    x, y = detector_xy_grids(h, w)
    sx, sy, valid = model.src_to_ref(x, y, src, ref)

    def forward(img):
        out = bilinear_sample(img, sy - 0.5, sx - 0.5, fill=0.0)
        return np.where(valid, out, 0.0)

    def adjoint(img):
        masked = np.where(valid, img, 0.0)
        return bilinear_sample_adjoint(masked, sy - 0.5, sx - 0.5, (h, w))

    a = rng.normal(size=(h, w))
    b = rng.normal(size=(h, w))
    return adjoint_relative_error(forward, adjoint, a, b)
