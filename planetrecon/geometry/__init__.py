"""Field rotation, rigid oblate-globe and Saturn ring-layer operators (W09–W10).

Coordinate frames: body-fixed surface, projected sky (x right, y up) and
detector pixels (x column, y row, origin top-left). Field rotation is sky
attitude; surface rotation is globe longitude. They are not a single 2-D warp.
"""

from planetrecon.geometry.coords import (
    detector_to_sky,
    detector_xy_grids,
    rotate_sky,
    sky_to_detector,
)
from planetrecon.geometry.fit import (
    estimate_field_angle,
    fit_disc_ellipse,
    sequence_degeneracy,
)
from planetrecon.geometry.globe import GlobeParams, body_to_sky, render_globe_texture, sky_to_body
from planetrecon.geometry.rings import RingParams
from planetrecon.geometry.saturn import SaturnSceneModel, render_saturn
from planetrecon.geometry.model import (
    SceneModel,
    compose_src_to_ref,
    render_observed,
)
from planetrecon.geometry.pose import FramePose, build_frame_poses, source_times_s, unwrap_angles
from planetrecon.geometry.warp import bilinear_push, bilinear_sample, bilinear_sample_adjoint

__all__ = [
    "GlobeParams",
    "RingParams",
    "FramePose",
    "SaturnSceneModel",
    "SceneModel",
    "bilinear_push",
    "bilinear_sample",
    "bilinear_sample_adjoint",
    "body_to_sky",
    "build_frame_poses",
    "compose_src_to_ref",
    "detector_to_sky",
    "detector_xy_grids",
    "estimate_field_angle",
    "fit_disc_ellipse",
    "render_globe_texture",
    "render_observed",
    "render_saturn",
    "rotate_sky",
    "sequence_degeneracy",
    "sky_to_body",
    "sky_to_detector",
    "source_times_s",
    "unwrap_angles",
]
