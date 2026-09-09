"""Independent analytic rotations test estimation and the resulting stack."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.ndimage import shift

from planetrecon.geometry.fit import estimate_field_angle
from planetrecon.io.source import ArraySource
from planetrecon.pipeline.baseline import stack_source
from planetrecon.pipeline.geometry_stack import prepare_geometry
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


@pytest.mark.parametrize('degrees', [0., 1., -1.])
def test_camera_drift_does_not_become_field_rotation(degrees):
    offsets = [(0, 0), (4, -6), (-7, 2), (2, 3), (4, -6),
               (-3, 2), (1, -4), (3, 1), (-7, 2)]
    frames = np.array([shift(scene(np.deg2rad(degrees)*i/8)[0], (dy, dx),
                             order=1, mode='constant')
                       for i, (dx, dy) in enumerate(offsets)])
    src = ArraySource(frames, bit_depth=32, timestamps=np.arange(9.))
    cfg = ReconstructionConfig(device='cpu', threads=2, frame_preselection=False,
        geometry_mode='field', field_center_x=96, field_center_y=96,
        equatorial_radius_px=70)
    corrected = stack_source(src, cfg)
    # Reproduce the former angle fit on its same three unregistered samples.
    angles = [estimate_field_angle(frames[0], frames[i], 96, 96, 70)['angle_rad']
              for i in (0, 4, 8)]
    old_rate = float(np.median(np.diff(np.unwrap(angles))/4))
    old = stack_source(src, replace(cfg, field_rate_rad_s=old_rate))
    reference, radius = scene(0)
    mask = (radius > 10) & (radius < 60)
    def error(result):
        assert result.n_used == 9 and result.n_rejected == 0
        return np.sqrt(np.mean((result.image[mask]-reference[mask])**2))
    assert error(corrected) < .02*error(old)
    assert corrected.provenance['geometry']['field_rate_rad_s'] == pytest.approx(
        np.deg2rad(degrees)/8, abs=3e-5)


def test_out_of_range_estimation_samples_cannot_set_a_rotation_rate():
    image, _ = scene(0)
    src = ArraySource(np.array([image, np.roll(image, 5, axis=1),
                                np.roll(image, 8, axis=1)]),
                      bit_depth=32, timestamps=np.arange(3.))
    cfg = ReconstructionConfig(geometry_mode='field', field_center_x=96,
        field_center_y=96, equatorial_radius_px=70, max_shift_px=2)
    _, _, diagnostic, warnings = prepare_geometry(src, cfg)
    assert diagnostic['field_sample_shifts_px'] == [[0., 0.], None, None]
    assert diagnostic['field_rate_rad_s'] == 0
    assert any('roll_unconstrained:' in warning for warning in warnings)


def test_rejected_middle_sample_does_not_discard_valid_endpoint_rate():
    angles = np.linspace(0, np.deg2rad(1), 9)
    frames = np.array([scene(angle)[0] for angle in angles])
    frames[4] = shift(frames[4], (0, 15), order=1, mode='constant')
    src = ArraySource(frames, bit_depth=32, timestamps=np.arange(9.))
    cfg = ReconstructionConfig(device='cpu', threads=2, frame_preselection=False,
        geometry_mode='field', field_center_x=96, field_center_y=96,
        equatorial_radius_px=70, max_shift_px=6)
    result = stack_source(src, cfg)
    assert result.n_used == 8 and result.n_rejected == 1
    diagnostics = result.provenance['geometry']
    assert diagnostics['field_sample_shifts_px'][1] is None
    assert diagnostics['field_rate_rad_s'] == pytest.approx(angles[-1]/8, abs=3e-5)
    reference, radius = scene(0)
    mask = (radius > 10) & (radius < 60)
    old_zero = stack_source(src, replace(cfg, field_rate_rad_s=0.))
    error = lambda r: np.sqrt(np.mean((r.image[mask]-reference[mask])**2))
    assert error(result) < .1*error(old_zero)


def test_rejected_angle_cannot_change_unwrap_branch(monkeypatch):
    import planetrecon.pipeline.geometry_stack as module
    image, _ = scene(0)
    frames = np.stack([image, shift(image, (0, 15), order=1, mode='constant'), image])
    src = ArraySource(frames, bit_depth=32, timestamps=np.array([0., 2., 8.]))
    cfg = ReconstructionConfig(geometry_mode='field', field_center_x=96,
        field_center_y=96, equatorial_radius_px=70, max_shift_px=6)
    angles = iter([0., 3.2, .1])
    monkeypatch.setattr(module, 'estimate_field_angle',
        lambda *args: {'angle_rad': next(angles), 'degeneracy': []})
    _, _, diagnostics, _ = prepare_geometry(src, cfg)
    assert diagnostics['field_sample_shifts_px'][1] is None
    assert diagnostics['field_rate_rad_s'] == pytest.approx(.1/8)
