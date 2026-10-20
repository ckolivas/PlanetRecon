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
                           progress=None, read_group=None, group=1):
    """Average the anchor-geometry frames again after alignment points align them to ``template``.

    ``read_group`` reads several indices at once (device planes); frames are
    then matched and pulled ``group`` at a time.
    """
    from planetrecon.pipeline.ap_align import AlignmentPoints
    matcher = AlignmentPoints(template, size=size, use_cuda=use_cuda, should_cancel=should_cancel)
    if use_cuda and len(matcher.centres) and read_group is not None:
        return _refine_groups(template, matcher, indices, read_group, group, should_cancel, progress)
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


def _refine_groups(template, matcher, indices, read_group, group, should_cancel, progress):
    """Device refinement with frames matched and pulled together."""
    import torch
    from planetrecon.backends.torch_ap import PointGrid, gaussian_proxy, sample_frames
    grid = PointGrid(template.shape, matcher.size, search=matcher.search)
    points = templates = None
    total = torch.zeros(template.shape, device='cuda:0', dtype=torch.float64)
    weight = torch.zeros_like(total)
    for begin in range(0, len(indices), group):
        if should_cancel is not None and should_cancel():
            raise InterruptedError('cancelled while refining the motion alignment reference')
        part = indices[begin:begin + group]
        if progress is not None:
            progress(begin + len(part))
        aligned = [a for a in read_group([int(i) for i in part]) if a is not None]
        if not aligned:
            continue
        images = torch.stack([a[0] for a in aligned])
        supports = torch.stack([a[1] for a in aligned])
        if templates is None:
            # Matching runs in the precision of the planes read; sums stay float64.
            points, templates = grid.fixed(matcher, images.dtype)
        field, _ = grid.fixed_fields(gaussian_proxy(images), points, templates)
        total += sample_frames(images, field).sum(0)
        weight += sample_frames(supports, field).sum(0)
    total, weight = total.cpu().numpy(), weight.cpu().numpy()
    refined = np.divide(total, weight, out=template.copy(), where=weight > 1e-12)
    return refined, weight >= 4. - 1e-9
