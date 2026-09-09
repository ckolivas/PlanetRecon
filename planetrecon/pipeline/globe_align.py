"""Camera drift from shared visible surface, excluding missing predictions."""

import numpy as np
from scipy.ndimage import binary_erosion

from planetrecon.geometry.model import render_observed
from planetrecon.geometry.globe import sky_to_body, field_rotate_sky
from planetrecon.pipeline.masked_align import MaskedRegistration


def surface_displacement(reference, frame, model, pose, reference_pose):
    """Match only where the prediction and its filter footprint are observed.

    A newly visible hemisphere has no reference data. Its zero-filled prediction
    must not pull the translation toward an artificial dark limb.
    """
    predicted = render_observed(reference, model, pose, reference_pose)
    support = render_observed(np.ones(reference.shape), model, pose, reference_pose)
    y, x = np.indices(reference.shape)
    sx, sy = x+.5-pose.cx, pose.cy-y-.5
    if model.apply_field:
        sx, sy = field_rotate_sky(sx, sy, -pose.field_angle_rad)
    _, _, on_globe, _ = sky_to_body(sx, sy, model.globe, pose.t_s)
    # Exclude the interpolated limb as well as newly visible longitudes. Its
    # pixelated silhouette changes under rotation without any camera drift.
    mask = binary_erosion((support >= 1.-1e-12) & on_globe,
                          structure=np.ones((15, 15), bool))
    return MaskedRegistration(predicted, mask).displacement(frame)
