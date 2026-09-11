"""Seeing residuals against a motion-predicted template, without resampling raw CFA."""
import numpy as np
from scipy.ndimage import map_coordinates

from planetrecon.pipeline.local_align import LocalRegistration


def build_motion_template(reference, indices, read_aligned, should_cancel=None):
    """Average observed proxies in the best frame's geometry and detector origin."""
    total, weight = np.zeros_like(reference), np.zeros_like(reference)
    used = 0
    for index in indices:
        if should_cancel is not None and should_cancel():
            raise InterruptedError('cancelled while building motion alignment reference')
        aligned = read_aligned(int(index))
        if aligned is None:
            continue
        image, support = aligned
        total += image
        weight += support
        used += 1
    if used < 4:
        raise ValueError('Local motion alignment requires four usable reference frames; disable Local patch alignment.')
    template = np.divide(total,weight,out=reference.copy(),where=weight > 1e-12)
    return template, weight >= 4.-1e-9


def inverse_local_coordinates(shift):
    """Invert the observed pull field, so each raw sample is scattered once.

    LocalRegistration measures predicted -> observed offsets. The geometry
    operator needs observed -> predicted positions; simply subtracting the
    offset at the observed pixel is biased for spatially varying distortion.
    """
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


def local_coordinates(template, template_support, frame, pose, reference_pose, render, window, use_cuda):
    predicted = render(template,pose,reference_pose)
    support = render(template_support.astype(float),pose,reference_pose)
    matcher = LocalRegistration(predicted,window=window,step=window//2,use_cuda=use_cuda,
                                valid_mask=support >= 1.-1e-9)
    shift = matcher.displacement(frame,(0.,0.))
    return inverse_local_coordinates(shift)
