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
    if use_cuda and len(matcher.centres):
        return _refine_on_device(template, matcher, indices, read_aligned, should_cancel, progress)
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


def _refine_on_device(template, matcher, indices, read_aligned, should_cancel, progress):
    """Same refinement with fields and resampling kept on the GPU."""
    import torch
    from planetrecon.backends.torch_circular import sample
    h, w = template.shape
    yy, xx = torch.meshgrid(torch.arange(h, device='cuda:0', dtype=torch.float64),
                            torch.arange(w, device='cuda:0', dtype=torch.float64), indexing='ij')
    total = torch.zeros((h, w), device='cuda:0', dtype=torch.float64)
    weight = torch.zeros_like(total)
    for step, index in enumerate(indices, 1):
        if should_cancel is not None and should_cancel():
            raise InterruptedError('cancelled while refining the motion alignment reference')
        if progress is not None:
            progress(step)
        aligned = read_aligned(int(index))
        if aligned is None:
            continue
        image, support = (torch.as_tensor(a, device='cuda:0', dtype=torch.float64) for a in aligned)
        field = matcher.displacement_tensor(image, (0., 0.))
        total.add_(sample(image, yy + field[1], xx + field[0]))
        weight.add_(sample(support, yy + field[1], xx + field[0]))
    total, weight = total.cpu().numpy(), weight.cpu().numpy()
    refined = np.divide(total, weight, out=template.copy(), where=weight > 1e-12)
    return refined, weight >= 4. - 1e-9
