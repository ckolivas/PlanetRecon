"""Independent analytic rotations test estimation and the resulting stack."""

from dataclasses import replace

import numpy as np
import pytest

from planetrecon.geometry.fit import estimate_field_angle
from planetrecon.io.source import ArraySource
from planetrecon.pipeline.baseline import stack_source
from planetrecon.reconstruction import ReconstructionConfig


def scene(angle):
    # Sample the rotated continuous scene directly, without the application
    # warp or an image interpolation that could share its coordinate errors.
    y, x = np.indices((192, 192), dtype=float)
    radius = np.hypot(x + .5 - 96, 96 - y - .5)
    theta = np.arctan2(96 - y - .5, x + .5 - 96) - angle
    image = ((radius < 70) * (1 + .3*np.cos(2*theta) + .15*np.sin(3*theta))
             * (1 + .1*np.cos(radius/4)))
    return image, radius


@pytest.mark.parametrize('degrees', [-4., -2., -1., -.5, 0., .5, 1., 2., 4.])
def test_fractional_polar_bins_recover_signed_rotation(degrees):
    reference, _ = scene(0)
    image, _ = scene(np.deg2rad(degrees))
    estimate = estimate_field_angle(reference, image, 96, 96, 70)
    assert not estimate['degeneracy']
    assert np.rad2deg(estimate['angle_rad']) == pytest.approx(degrees, abs=.08)


@pytest.mark.parametrize('direction', [-1., 1.])
def test_small_automatic_rotation_improves_unsharpened_stack(direction):
    angles = np.linspace(0, direction*np.deg2rad(1), 9)
    frames = np.array([scene(angle)[0] for angle in angles])
    source = ArraySource(frames, bit_depth=32, timestamps=np.arange(9, dtype=float))
    config = ReconstructionConfig(device='cpu', threads=2, frame_preselection=False,
        geometry_mode='field', field_center_x=96, field_center_y=96,
        equatorial_radius_px=70)
    inferred = stack_source(source, config)
    # The old whole-bin estimator returned zero throughout this sequence.
    uncorrected = stack_source(source, replace(config, field_rate_rad_s=0))
    oracle = stack_source(source, replace(config, field_rate_rad_s=angles[-1]/8))
    radius = scene(0)[1]
    interior = (radius > 10) & (radius < 60)
    def error(result):
        assert result.n_used == 9 and result.n_rejected == 0
        assert result.validity[interior].all()
        return np.sqrt(np.mean((result.image[interior] - frames[0][interior])**2))
    assert error(inferred) < .1*error(uncorrected)
    assert error(inferred) < 1.1*error(oracle)
    assert inferred.provenance['geometry']['field_rate_rad_s'] == pytest.approx(
        angles[-1]/8, rel=.02)


def test_featureless_disc_still_leaves_roll_unconstrained():
    _, radius = scene(0)
    disc = (radius < 70).astype(float)
    estimate = estimate_field_angle(disc, disc, 96, 96, 70)
    assert estimate['angle_rad'] == 0
    assert 'roll_unconstrained' in estimate['degeneracy']
