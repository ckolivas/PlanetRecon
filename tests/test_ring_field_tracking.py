"""Known field rotation and camera drift must preserve Saturn layer separation."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.ndimage import shift

from planetrecon.geometry.pose import FramePose
from planetrecon.io.source import ArraySource
from planetrecon.pipeline.baseline import stack_source
from planetrecon.pipeline.ring_align import RingRegistration
from test_ring_registration import config, source


@pytest.mark.parametrize('colour', ['mono', 'RGGB', 'GRBG', 'GBRG', 'BGGR'])
@pytest.mark.parametrize('rate', [-.03, .03, 1e-9])
def test_field_and_spin_drift_preserves_stationary_stack(colour, rate):
    cfg = replace(config(), field_rate_rad_s=rate)
    offsets = [(0, 0), (2, -4), (-4, 2), (4, 4), (-2, -2)]
    corrected = stack_source(source(colour, offsets, rate), cfg)
    stationary = stack_source(source(colour, [(0, 0)]*5, rate), cfg)
    mask = corrected.validity & stationary.validity & (stationary.image > .05)
    assert corrected.n_used == 5 and corrected.n_rejected == 0
    np.testing.assert_allclose(corrected.image[mask], stationary.image[mask], atol=1e-7)
    for layer in ('globe', 'ring'):
        np.testing.assert_array_equal(corrected.layer_coverage[layer] > 0,
                                      stationary.layer_coverage[layer] > 0)


def test_noisy_asymmetric_rings_improve_with_field_and_spin(monkeypatch):
    rate = .04
    texture = lambda radius, az: .6+.25*np.cos(3*az)+.2*np.sin(radius/2)
    clean = source('mono', [(0, 0)]*5, rate, texture)
    offsets = [(0, 0), (1.3, -2.4), (-3.7, 2.2), (.3, .4), (2.5, -1.2)]
    rng = np.random.default_rng(811)
    frames = np.array([shift(clean.read_raw(i), (dy, dx), order=3, mode='constant')
                       + rng.normal(0, .01, (112, 112))
                       for i, (dx, dy) in enumerate(offsets)])
    drifting = ArraySource(frames, bit_depth=32, timestamps=np.arange(5.))
    cfg = replace(config(), field_rate_rad_s=rate)
    stationary = stack_source(clean, cfg)
    corrected = stack_source(drifting, cfg)
    monkeypatch.setattr(RingRegistration, 'displacement_at_pose', lambda *args: (0., 0.))
    fixed = stack_source(drifting, cfg)
    mask = stationary.validity & corrected.validity & fixed.validity & (stationary.image > .05)
    def error(result):
        assert result.n_used == 5 and result.n_rejected == 0
        return np.sqrt(np.mean((result.image[mask]-stationary.image[mask])**2))
    assert error(corrected) < .3*error(fixed)


def test_rotated_template_excludes_missing_capture_and_filter_footprint(monkeypatch):
    reference = source('mono', [(0, 0)]).read_raw(0)[16:96]
    matcher = RingRegistration(reference, 56, 40, 20, 42)
    masks = []
    def inspect(self, frame, **kwargs):
        masks.append(self.mask.copy())
        return None
    monkeypatch.setattr(RingRegistration, 'displacement', inspect)
    matcher.displacement_at_pose(reference, FramePose(0, .4, 56, 40), FramePose(0, 0, 56, 40))
    assert len(masks) == 1 and masks[0].any()
    y, x = np.nonzero(masks[0])
    # Independently map the corners of every smoothing footprint back into
    # the original capture; bilinear interpolation needs real neighbours.
    for dy in (-7, 7):
        for dx in (-7, 7):
            sx, sy = x+.5+dx-56, 40-(y+.5+dy)
            rx = np.cos(.4)*sx + np.sin(.4)*sy + 56
            ry = 40-(-np.sin(.4)*sx + np.cos(.4)*sy)
            assert np.all((rx >= .5) & (rx <= 111.5))
            assert np.all((ry >= .5) & (ry <= 79.5))


@pytest.mark.parametrize('index', [1, 2, 4])
def test_interpolation_alone_does_not_invent_camera_drift(index):
    from planetrecon.geometry.model import FieldOnlyModel, render_observed
    texture = lambda radius, az: .6+.25*np.cos(3*az)+.2*np.sin(radius/2)
    src = source('mono', [(0, 0)]*5, .04, texture)
    reference = src.read_raw(0)
    anchor = FramePose(0, 0, 56, 56)
    pose = FramePose(index, .04*index, 56, 56)
    predicted = render_observed(reference, FieldOnlyModel(), pose, anchor)
    ungated = RingRegistration(predicted, 56, 56, 20, 42).displacement(src.read_raw(index))
    assert np.linalg.norm(ungated) > 1e-4
    matcher = RingRegistration(reference, 56, 56, 20, 42)
    assert matcher.displacement_at_pose(src.read_raw(index), pose, anchor) == (0., 0.)


def test_field_tracking_keeps_nonzero_reference_anchor():
    rate = .03
    cfg = replace(config(), field_rate_rad_s=rate, reference_index=2)
    offsets = [(2, -4), (-4, 2), (0, 0), (4, 4), (-2, -2)]
    corrected = stack_source(source('mono', offsets, rate), cfg)
    stationary = stack_source(source('mono', [(0, 0)]*5, rate), cfg)
    assert corrected.provenance['reference_index'] == 2
    assert corrected.n_used == stationary.n_used == 5
    mask = corrected.validity & stationary.validity & (stationary.image > .05)
    np.testing.assert_allclose(corrected.image[mask], stationary.image[mask], atol=1e-7)


def test_field_tracking_respects_shift_limit():
    cfg = replace(config(), field_rate_rad_s=.03, max_shift_px=2)
    result = stack_source(source('mono', [(0, 0), (4, 4)], .03), cfg)
    assert result.n_used == 1 and result.n_rejected == 1
