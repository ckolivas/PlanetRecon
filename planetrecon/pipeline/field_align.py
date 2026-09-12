"""Field-only translation from shared observed detector support."""
import numpy as np
from scipy.ndimage import minimum_filter

from planetrecon.geometry.model import FieldOnlyModel, render_observed
from planetrecon.pipeline.masked_align import MaskedRegistration


def field_displacement(reference, frame, pose, reference_pose, radius):
    """Exclude rotated padding and its smoothing footprint from drift fitting."""
    model = FieldOnlyModel()
    predicted = render_observed(reference, model, pose, reference_pose)
    support = render_observed(np.ones(reference.shape), model, pose, reference_pose)
    # The sampled limb changes under interpolation even with no true drift.
    # Keep its smoothing footprint out of the translation evidence as well.
    y, x = np.indices(reference.shape)
    radial = np.hypot(x+.5-pose.cx, y+.5-pose.cy)
    mask = minimum_filter((support >= 1.-1e-12) & (radial < radius),
                          size=15, mode='constant', cval=0)
    return MaskedRegistration(predicted, mask).displacement(frame)
