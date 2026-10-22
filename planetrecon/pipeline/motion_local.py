"""Seeing residuals against a motion-predicted template, without resampling raw CFA."""
import numpy as np
from scipy.ndimage import map_coordinates


def build_motion_template(reference, indices, read_aligned, should_cancel=None, *, read_group=None, group=1):
    """Average observed proxies in the selected anchor geometry and detector origin.

    ``read_group`` reads ``group`` indices at once; its planes may be device tensors.
    """
    total, weight = np.zeros_like(reference), np.zeros_like(reference)
    used = 0
    if read_group is not None:
        import torch
        total, weight = (torch.as_tensor(a, device='cuda:0') for a in (total, weight))
    for begin in range(0, len(indices), group):
        if should_cancel is not None and should_cancel():
            raise InterruptedError('cancelled while building motion alignment reference')
        part = [int(i) for i in indices[begin:begin + group]]
        for aligned in (read_group(part) if read_group is not None else map(read_aligned, part)):
            if aligned is None:
                continue
            image, support = aligned
            total += image
            weight += support
            used += 1
    if read_group is not None:
        total, weight = total.cpu().numpy(), weight.cpu().numpy()
    if used < 4:
        raise ValueError('Local motion alignment requires four usable reference frames; use global alignment.')
    template = np.divide(total,weight,out=reference.copy(),where=weight > 1e-12)
    return template, weight >= 4.-1e-9


def inverse_local_coordinates(shift, *, use_cuda=False):
    """Invert the observed pull field, so each raw sample is scattered once.

    Alignment points measure predicted -> observed offsets. The geometry
    operator needs observed -> predicted positions; simply subtracting the
    offset at the observed pixel is biased for spatially varying distortion.
    """
    if use_cuda:
        return cuda_inverse_local_coordinates(shift)
    ux,uy = shift
    y,x = np.indices(ux.shape,dtype=float)
    qx,qy = x.copy(),y.copy()
    for _ in range(8):
        dx = map_coordinates(ux,[qy,qx],order=1,mode='nearest',prefilter=False)
        dy = map_coordinates(uy,[qy,qx],order=1,mode='nearest',prefilter=False)
        qx,qy = x-dx,y-dy
    error = np.maximum(np.abs(qx+map_coordinates(ux,[qy,qx],order=1,mode='nearest',prefilter=False)-x),
                       np.abs(qy+map_coordinates(uy,[qy,qx],order=1,mode='nearest',prefilter=False)-y))
    if not np.isfinite(error).all() or error.max() > .01:
        return x+.5,y+.5  # Unreliable deformation retains global motion.
    return qx+.5,qy+.5


def cuda_inverse_local_coordinates(shift, *, tensors=False):
    """Same eight fixed-point steps and residual guard on the selected GPU.

    ``shift`` is (2, h, w), or (B, 2, h, w) for a batch with a guard per frame.
    """
    import torch
    import torch.nn.functional as functional
    if torch.is_tensor(shift):
        # Tensors keep their precision (float32 registration fields stay float32).
        field = shift.to(device='cuda:0')
    else:
        field = torch.as_tensor(np.stack(shift), dtype=torch.float64, device='cuda:0')
    batch = field.ndim == 4
    field = field if batch else field[None]
    h,w = field.shape[-2:]
    y,x = torch.meshgrid(torch.arange(h, dtype=field.dtype, device='cuda:0'),
                         torch.arange(w, dtype=field.dtype, device='cuda:0'), indexing='ij')
    try:
        from planetrecon.backends.triton_warp import invert
    except ImportError:
        invert = None
    if invert is not None:
        # All eight steps per pixel in one kernel, sampling at pixel coordinates as the CPU does.
        qx,qy,error = invert(field)
        error = error.flatten(1)
    else:
        qx,qy = x.expand(len(field),h,w),y.expand(len(field),h,w)
        def sample():
            grid = torch.stack((2*qx/max(w-1,1)-1, 2*qy/max(h-1,1)-1), dim=-1)
            return functional.grid_sample(field, grid, mode='bilinear', padding_mode='border',
                                          align_corners=True).unbind(1)
        for _ in range(8):
            dx,dy = sample()
            qx,qy = x-dx,y-dy
        dx,dy = sample()
        error = torch.maximum((qx+dx-x).abs(), (qy+dy-y).abs()).flatten(1)
    # Unreliable deformation retains global motion.
    bad = (~torch.isfinite(error).all(1) | (error.amax(1) > .01))[:, None, None]
    qx,qy = torch.where(bad,x,qx),torch.where(bad,y,qy)
    if not batch:
        qx,qy = qx[0],qy[0]
    if tensors:
        return qx+.5, qy+.5
    output = torch.stack((qx+.5,qy+.5)).cpu().numpy()
    return output[0],output[1]


def local_coordinates(template_with_support, frame, pose, reference_pose, render, use_cuda,
                      *, should_cancel=None, ap_size=64):
    """Predict a fixed, prepared reference; only pose and observations vary."""
    from planetrecon.pipeline.ap_align import AlignmentPoints
    projected = render(template_with_support,pose,reference_pose,gain_channels=1)
    predicted,support = projected[...,0],projected[...,1]
    if should_cancel and should_cancel():
        raise InterruptedError('cancelled during local motion alignment')
    matcher = AlignmentPoints(predicted, size=ap_size, use_cuda=use_cuda, valid_mask=support >= 1.-1e-9,
                              should_cancel=should_cancel)
    shift = matcher.displacement(frame,(0.,0.))
    return inverse_local_coordinates(shift, use_cuda=use_cuda)
