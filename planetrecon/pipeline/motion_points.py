"""Alignment points combined with a motion model: the derotated reference.

The reference is a mean of many frames predicted into the anchor geometry by
the model, refined once with alignment points. Frames are then matched in
their own detector coordinates against that reference predicted at each frame
time (motion_local.local_coordinates), so the measured seeing field is smooth
and needs no composition through the model's limb discontinuities.
"""
import numpy as np
from planetrecon.pipeline.local_align import pull


def refine_motion_template(template, indices, read_aligned, *, size, use_cuda=False, should_cancel=None,
                           progress=None):
    """Average the anchor-geometry frames again after alignment points align them to ``template``."""
    from planetrecon.pipeline.ap_align import AlignmentPoints
    matcher = AlignmentPoints(template, size=size, use_cuda=use_cuda, should_cancel=should_cancel)
    total = np.zeros_like(template)
    weight = np.zeros_like(template)
    for step, index in enumerate(indices, 1):
        if should_cancel is not None and should_cancel():
            raise InterruptedError('cancelled while refining the motion alignment reference')
        if progress is not None:
            progress(step)
        aligned = read_aligned(int(index))
        if aligned is None:
            continue
        image, support = aligned
        field = matcher.displacement(image, (0., 0.)) if len(matcher.centres) else (0., 0.)
        # Rendered planes are already weighted by their bilinear support.
        total += pull(image, field)
        weight += pull(support, field)
    refined = np.divide(total, weight, out=template.copy(), where=weight > 1e-12)
    return refined, weight >= 4. - 1e-9
