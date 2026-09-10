"""Exposed rings constrain camera motion independently of globe spin."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.ndimage import shift

from planetrecon.detector import cfa_labels
from planetrecon.geometry.globe import GlobeParams
from planetrecon.geometry.pose import FramePose
from planetrecon.geometry.rings import RingParams
from planetrecon.geometry.saturn import render_saturn
from planetrecon.io.source import ArraySource
from planetrecon.pipeline.baseline import stack_source
from planetrecon.pipeline.ring_align import RingRegistration
from planetrecon.reconstruction import ReconstructionConfig


def source(colour, offsets, field_rate=0., ring_texture=None):
    globe = GlobeParams(20, flattening=.1, sub_obs_lat_rad=.4, surface_rate_rad_s=.3)
    rings = RingParams(26, 42, transmission=.35)
    frames = []
    for i, (dx, dy) in enumerate(offsets):
        image = render_saturn(112, 112, FramePose(i, field_rate*i, 56+dx, 56+dy), globe, rings,
            lambda lon, lat: 1+.2*np.cos(5*lon)*np.cos(3*lat), ring_tex=ring_texture)
        if colour != 'mono':
            labels = cfa_labels(112, 112, colour)
            image *= sum((labels == name)*value for name, value in zip('RGB', (.8, 1., .6)))
        frames.append(image)
    return ArraySource(np.array(frames), color_mode=colour, bit_depth=32,
                       timestamps=np.arange(len(frames), dtype=float))


def config():
    return ReconstructionConfig(device='cpu', threads=2, frame_preselection=False,
        geometry_mode='saturn', field_center_x=56, field_center_y=56,
        equatorial_radius_px=20, flattening=.1, sub_obs_lat_rad=.4,
        ring_inner_radius_px=26, ring_outer_radius_px=42, ring_transmission=.35,
        surface_rate_rad_s=.3, field_rate_rad_s=0)


@pytest.mark.parametrize('colour', ['mono', 'RGGB', 'GRBG', 'GBRG', 'BGGR'])
def test_rotating_saturn_drift_recovers_stationary_layers(colour):
    offsets = [(0, 0), (2, -4), (-4, 2), (4, 4), (-2, -2)]
    result = stack_source(source(colour, offsets), config())
    stationary = stack_source(source(colour, [(0, 0)]*5), config())
    mask = stationary.validity & (stationary.image > .05)
    np.testing.assert_allclose(result.image[mask], stationary.image[mask], atol=1e-10)
    assert result.n_used == 5 and result.n_rejected == 0
    for layer in ('globe', 'ring'):
        np.testing.assert_allclose(result.layer_coverage[layer],
                                   stationary.layer_coverage[layer], atol=1e-10)


def test_ring_match_ignores_changing_globe_and_brightness():
    frames = source('mono', [(0, 0), (2, -4)])
    reference, shifted = (frames.read_raw(i).copy() for i in range(2))
    y, x = np.indices(reference.shape)
    reference[np.hypot(x+.5-56, y+.5-56) < 20] = 1000
    shifted[np.hypot(x+.5-58, y+.5-52) < 20] = -1000
    matcher = RingRegistration(reference, 56, 56, 20, 42)
    np.testing.assert_allclose(matcher.displacement(shifted*.7+12), (2, -4), atol=1e-8)


def test_flat_ring_region_is_unconstrained():
    matcher = RingRegistration(np.ones((112, 112)), 56, 56, 20, 42)
    assert matcher.displacement(np.ones((112, 112))) is None


def test_unconstrained_ring_match_refuses_run():
    flat = ArraySource(np.ones((2, 112, 112)), bit_depth=32, timestamps=np.arange(2.))
    with pytest.raises(ValueError, match='exposed rings cannot constrain camera drift'):
        stack_source(flat, config())


def test_large_camera_drift_uses_existing_shift_limit():
    result = stack_source(source('mono', [(0, 0), (4, 4)]),
                          replace(config(), max_shift_px=2))
    assert result.n_used == 1 and result.n_rejected == 1


@pytest.mark.parametrize('offset', [(1.3, -2.4), (-3.7, 2.2), (.3, .4)])
def test_noisy_fractional_translation(offset):
    reference = source('mono', [(0, 0)]).read_raw(0)
    image = shift(reference, offset[::-1], order=3, mode='constant')
    image += np.random.default_rng(672).normal(0, .01, image.shape)
    estimate = RingRegistration(reference, 56, 56, 20, 42).displacement(image)
    np.testing.assert_allclose(estimate, offset, atol=.02)


def test_repeating_texture_does_not_invent_a_translation():
    _, x = np.indices((112, 112))
    stripes = 1 + np.cos(x*2*np.pi/8)
    matcher = RingRegistration(stripes, 56, 56, 20, 42)
    assert matcher.displacement(stripes) is None


def test_broad_resolved_ring_peak_is_not_a_competing_translation():
    from scipy.ndimage import zoom, gaussian_filter
    from planetrecon.pipeline.masked_align import MaskedRegistration
    reference = zoom(source('mono', [(0, 0)]).read_raw(0), 3., order=1)
    reference = gaussian_filter(reference, 4.)
    offset = (1.3, -2.4)
    moved = shift(reference, offset[::-1], order=3, mode='constant')
    moved += np.random.default_rng(11).normal(0, .001, moved.shape)
    matcher = RingRegistration(reference, 168, 168, 60, 126)
    # The old test treats the same peak's shoulders as alternative matches.
    assert MaskedRegistration.displacement(matcher, moved) is None
    np.testing.assert_allclose(matcher.displacement(moved), offset, atol=.03)


def test_noisy_drifting_spin_stack_improves_image(monkeypatch):
    offsets = [(0, 0), (1.3, -2.4), (-3.7, 2.2), (.3, .4), (2.5, -1.2)]
    clean = source('mono', [(0, 0)]*5)
    rng = np.random.default_rng(672)
    frames = np.array([shift(clean.read_raw(i), (dy, dx), order=3, mode='constant')
                       + rng.normal(0, .01, (112, 112))
                       for i, (dx, dy) in enumerate(offsets)])
    drifting = ArraySource(frames, bit_depth=32, timestamps=np.arange(5.))
    stationary = stack_source(clean, config())
    corrected = stack_source(drifting, config())
    # Compare the old fixed-centre behavior on exactly the same observations.
    monkeypatch.setattr(RingRegistration, 'displacement', lambda self, frame: (0., 0.))
    fixed = stack_source(drifting, config())
    mask = stationary.validity & corrected.validity & fixed.validity & (stationary.image > .05)
    def error(result):
        assert result.n_used == 5 and result.n_rejected == 0
        return np.sqrt(np.mean((result.image[mask]-stationary.image[mask])**2))
    assert error(corrected) < .3*error(fixed)


@pytest.mark.parametrize('colour', ['mono', 'RGGB'])
@pytest.mark.parametrize('field_rate', [0., .03])
def test_ring_registration_resume_is_exact(tmp_path, colour, field_rate):
    from planetrecon.io.ser import SERSource, write_ser, NAME_TO_COLOR
    capture = source(colour, [(0, 0), (2, -4), (-4, 2), (4, 4), (-2, -2)], field_rate)
    path = write_ser(tmp_path/'spin.ser', np.array([
        capture.read_raw(i)*1000 for i in range(5)]).astype('u2'),
        color_id=NAME_TO_COLOR[colour])
    cfg = replace(config(), cadence_s=1., batch_frames=1, field_rate_rad_s=field_rate)
    state = tmp_path/'state.npz'
    cancelled = False
    def event(result, info):
        nonlocal cancelled
        if info['n_processed'] >= 2:
            cancelled = True
    with SERSource(path) as src:
        whole = stack_source(src, cfg)
        partial = stack_source(src, cfg, state_checkpoint=state, on_event=event,
                               should_cancel=lambda: cancelled)
        resumed = stack_source(src, cfg, resume_from=state)
    assert partial.incomplete and partial.n_used == 2
    assert resumed.n_used == whole.n_used == 5
    for key in ('image', 'coverage', 'validity'):
        np.testing.assert_array_equal(getattr(resumed, key), getattr(whole, key))
    for layer in ('globe', 'ring'):
        np.testing.assert_array_equal(resumed.layer_coverage[layer], whole.layer_coverage[layer])
