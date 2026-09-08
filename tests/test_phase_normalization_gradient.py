"""Normalized PSFs cannot fit a constant brightness residual through phase."""
import numpy as np
import pytest

from planetrecon.config import make_config
from planetrecon.mfbd import PupilForward, frame_loss_and_grad


@pytest.mark.parametrize('m', [3, 15, 35, 60])
def test_constant_residual_does_not_change_normalized_psf_phase_gradient(m):
    fwd = PupilForward.from_config(make_config(1001, 4., n_diam=16, pupil_pad_factor=8., eval_size=32))
    y, x = np.indices((32, 32))
    obj = 40.+200.*np.exp(-((x-15.)**2+(y-17.)**2)/24.)
    obj_f = np.fft.fft2(obj)
    alpha = np.random.default_rng(91).normal(0., .3, size=m)
    alpha[:2] = [33.8, -32.8]
    image = np.fft.ifft2(fwd.otf(np.zeros(m))*obj_f).real
    _, grad = frame_loss_and_grad(alpha, fwd, obj_f, image, 2.)
    _, offset_grad = frame_loss_and_grad(alpha, fwd, obj_f, image+1000., 2.)
    np.testing.assert_allclose(offset_grad, grad, rtol=1e-8, atol=1e-7)


def test_phase_gradient_with_large_shift_and_brightness_mismatch():
    fwd = PupilForward.from_config(make_config(1001, 4., n_diam=16, pupil_pad_factor=8., eval_size=32))
    y, x = np.indices((32, 32))
    obj_f = np.fft.fft2(40.+200.*np.exp(-((x-15.)**2+(y-17.)**2)/24.))
    alpha = np.random.default_rng(91).normal(0., .3, size=15)
    alpha[:2] = [33.8, -32.8]
    image = np.fft.ifft2(fwd.otf(np.zeros(15))*obj_f).real+500.
    _, gradient = frame_loss_and_grad(alpha, fwd, obj_f, image, 2.)
    eps = .001
    fd = []
    for i in range(15):
        delta = np.eye(15)[i]*eps
        up = frame_loss_and_grad(alpha+delta, fwd, obj_f, image, 2.)[0]
        down = frame_loss_and_grad(alpha-delta, fwd, obj_f, image, 2.)[0]
        fd.append((up-down)/(2*eps))
    assert np.linalg.norm(gradient-fd)/np.linalg.norm(fd) < 1e-5
