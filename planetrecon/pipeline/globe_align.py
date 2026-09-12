"""Camera drift from shared visible surface, excluding missing predictions."""

import numpy as np
from scipy.ndimage import minimum_filter

from planetrecon.geometry.model import render_observed
from planetrecon.geometry.globe import sky_to_body, field_rotate_sky
from planetrecon.pipeline.masked_align import MaskedRegistration


def surface_displacement(reference, frame, model, pose, reference_pose, *, renderer=None):
    """Match only where the prediction and its filter footprint are observed.

    A newly visible hemisphere has no reference data. Its zero-filled prediction
    must not pull the translation toward an artificial dark limb.
    """
    def render(image):
        if renderer is not None:
            return renderer(image, pose, reference_pose)
        return render_observed(image, model, pose, reference_pose)
    predicted = render(reference)
    y, x = np.indices(reference.shape)
    rx, ry = x+.5-reference_pose.cx, reference_pose.cy-y-.5
    if model.apply_field:
        rx, ry = field_rotate_sky(rx, ry, -reference_pose.field_angle_rad)
    _, _, reference_globe, _ = sky_to_body(rx, ry, model.globe, reference_pose.t_s)
    # Reference limb pixels can contain interpolated sky (especially in a CFA
    # proxy). Foreshortening can stretch that artificial edge deep into the new
    # disc. Require a complete reference neighbourhood before warping support.
    reference_support = minimum_filter(reference_globe, size=3, mode='constant', cval=0)
    support = render(reference_support.astype(float))
    sx, sy = x+.5-pose.cx, pose.cy-y-.5
    if model.apply_field:
        sx, sy = field_rotate_sky(sx, sy, -pose.field_angle_rad)
    _, _, on_globe, _ = sky_to_body(sx, sy, model.globe, pose.t_s)
    # Exclude the interpolated limb as well as newly visible longitudes. Its
    # pixelated silhouette changes under rotation without any camera drift.
    mask = minimum_filter((support >= 1.-1e-12) & on_globe,
                          size=15, mode='constant', cval=0)
    return MaskedRegistration(predicted, mask).displacement(frame, distinct_peaks=True)
