"""Diagnostic joint fitter with differentiable prefiltered cubic B-splines.

The original joint fitter is deliberately retained for reproducibility. Only
its forward reference representation and sampler change here. Coefficients are
not image pixels: prefiltering is required to interpolate the reference itself.
Observed frames and the final stacking sampler remain unchanged.
"""
import numpy as np
from scipy.ndimage import spline_filter
import torch

from tools.joint_registration import JointRegistration


def spline_coefficients(images):
    """Convert each 2-D image to reflect-boundary cubic spline coefficients."""
    return np.stack([spline_filter(image, order=3, mode='reflect') for image in images])


def spline_sample(coefficients, x, y):
    """Evaluate a cardinal cubic spline; differentiate coordinates and values."""
    def axis(coordinates, length):
        base = coordinates.floor().long()
        t = coordinates-base
        weights = torch.stack(((1-t)**3, 3*t**3-6*t*t+4,
                               -3*t**3+3*t*t+3*t+1, t**3))/6
        offsets = torch.arange(-1, 3, device=coordinates.device)
        indices = base[None]+offsets.reshape(4, *([1]*coordinates.ndim))
        indices = indices.remainder(2*length)
        indices = torch.where(indices < length, indices, 2*length-1-indices)
        return indices, weights

    h, w = coefficients.shape
    ix, wx = axis(x, w)
    iy, wy = axis(y, h)
    values = coefficients[iy[:, None], ix[None, :]]
    return (values*wy[:, None]*wx[None, :]).sum(dim=(0, 1))


class SplineJointRegistration(JointRegistration):
    def __init__(self, reference, matcher, **kwargs):
        super().__init__(reference, matcher, **kwargs)
        self.bank = torch.as_tensor(spline_coefficients(self.bank.cpu().numpy()),
                                    device=self.device, dtype=torch.float64)

    def render(self, reference, x, y):
        return spline_sample(reference, x, y)
