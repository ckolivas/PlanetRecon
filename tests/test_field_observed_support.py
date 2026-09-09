"""Field rotation must use observed texture, never off-detector padding."""
from dataclasses import replace

import numpy as np
import pytest

from planetrecon.geometry.fit import estimate_field_angle
from planetrecon.detector import cfa_labels
from planetrecon.geometry.pose import FramePose
from planetrecon.io.source import ArraySource
from planetrecon.pipeline.baseline import stack_source
from planetrecon.reconstruction import ReconstructionConfig


def cropped_scene(angle, cx=44., dx=0., dy=0.):
    y, x = np.indices((96, 128), dtype=float)
    radius = np.hypot(x+.5-cx-dx, 48+dy-y-.5)
    theta = np.arctan2(48+dy-y-.5, x+.5-cx-dx)-angle
    image = (radius < 70)*(1+.3*np.cos(2*theta)+.15*np.sin(3*theta))
    return image, radius


@pytest.mark.parametrize('cx', [44., 84., 2., -5.])
def test_detector_boundary_cannot_supply_rotation_texture(cx):
    image = np.ones((96, 128))
    result = estimate_field_angle(image, image, cx, 48, 70)
    assert 'roll_unconstrained' in result['degeneracy']


@pytest.mark.parametrize('cx', [44., 84.])
@pytest.mark.parametrize('degrees', [-4., 4.])
def test_cropped_disc_uses_its_observed_inner_texture(cx, degrees):
    reference, _ = cropped_scene(0., cx)
    image, _ = cropped_scene(np.deg2rad(degrees), cx)
    result = estimate_field_angle(reference, image, cx, 48, 70)
    assert not result['degeneracy']
    assert np.rad2deg(result['angle_rad']) == pytest.approx(degrees, abs=.08)


@pytest.mark.parametrize('colour', ['mono', 'RGB', 'RGGB', 'GRBG', 'GBRG', 'BGGR'])
@pytest.mark.parametrize('direction', [-1., 1.])
def test_cropped_disc_rotation_improves_unsharpened_stack(direction, colour):
    angles = np.linspace(0, direction*np.deg2rad(4), 9)
    frames = np.array([cropped_scene(angle)[0] for angle in angles])
    truth = frames[0].copy()
    if colour != 'mono':
        truth = truth[..., None]*[.8, 1., .6]
        if colour == 'RGB':
            frames = frames[..., None]*[.8, 1., .6]
        else:
            labels = cfa_labels(96, 128, colour)
            frames *= np.where(labels == 'R', .8, np.where(labels == 'B', .6, 1.))
    src = ArraySource(frames, color_mode=colour, bit_depth=32, timestamps=np.arange(9.))
    cfg = ReconstructionConfig(device='cpu', threads=2, frame_preselection=False,
        geometry_mode='field', field_center_x=44, field_center_y=48,
        equatorial_radius_px=70)
    corrected = stack_source(src, cfg)
    uncorrected = stack_source(src, replace(cfg, field_rate_rad_s=0.))
    oracle = stack_source(src, replace(cfg, field_rate_rad_s=angles[-1]/8))
    _, radius = cropped_scene(0.)
    mask = (radius > 10) & (radius < 32)
    def error(result):
        assert result.n_used == 9 and result.n_rejected == 0
        assert result.validity[mask].all()
        return np.sqrt(np.mean((result.image[mask]-truth[mask])**2))
    # CFA reconstruction has a separate interpolation floor shared with the
    # known-rate control; judge estimation against that same colour sampling.
    assert error(corrected) < (.2 if colour in ('mono', 'RGB') else .5)*error(uncorrected)
    assert error(corrected) < 1.1*error(oracle)


def test_unobserved_padding_cannot_authorize_a_motion_run():
    src = ArraySource(np.ones((3, 96, 128)), bit_depth=32, timestamps=np.arange(3.))
    cfg = ReconstructionConfig(device='cpu', threads=2, frame_preselection=False,
        geometry_mode='field', field_center_x=44, field_center_y=48,
        equatorial_radius_px=70)
    with pytest.raises(ValueError, match='field rotation could not be estimated'):
        stack_source(src, cfg)
    explicit_zero = stack_source(src, replace(cfg, field_rate_rad_s=0.))
    assert explicit_zero.n_used == 3
    np.testing.assert_array_equal(explicit_zero.image, 1.)


@pytest.mark.parametrize('angle', [-.07, .07])
@pytest.mark.parametrize('offset', [(0., 0.), (3., -2.), (-2.4, 1.3)])
def test_cropped_rotating_scene_recovers_camera_drift(angle, offset):
    from planetrecon.pipeline.field_align import field_displacement
    reference, _ = cropped_scene(0.)
    frame, _ = cropped_scene(angle, dx=offset[0], dy=offset[1])
    fitted = field_displacement(reference, frame, FramePose(1., angle, 44, 48),
                                FramePose(0., 0., 44, 48), 70)
    assert fitted is not None
    np.testing.assert_allclose(fitted, offset, atol=.1)


def test_field_matching_excludes_full_unobserved_filter_footprint(monkeypatch):
    from planetrecon.pipeline.field_align import field_displacement
    from planetrecon.pipeline.masked_align import MaskedRegistration
    masks = []
    def inspect(self, frame, **kwargs):
        masks.append(self.mask)
        return None
    monkeypatch.setattr(MaskedRegistration, 'displacement', inspect)
    reference, _ = cropped_scene(0.)
    field_displacement(reference, reference, FramePose(1., .4, 44, 48),
                       FramePose(0., 0., 44, 48), 70)
    assert len(masks) == 1 and masks[0].any()
    y, x = np.nonzero(masks[0])
    for dx in (-7, 7):
        for dy in (-7, 7):
            sx, sy = x+.5+dx-44, 48-(y+.5+dy)
            assert np.all(sx*sx+sy*sy < 70**2)
            rx = np.cos(.4)*sx+np.sin(.4)*sy+44
            ry = 48-(-np.sin(.4)*sx+np.cos(.4)*sy)
            assert np.all((rx >= .5) & (rx <= 127.5))
            assert np.all((ry >= .5) & (ry <= 95.5))


@pytest.mark.parametrize('colour', ['mono', 'RGGB'])
def test_cropped_field_resume_preserves_image_and_support(tmp_path, colour):
    from planetrecon.io.ser import write_ser, SERSource, COLOR_RGGB
    frames = np.array([cropped_scene(.01*i)[0] for i in range(7)])*10000
    if colour == 'RGGB':
        labels = cfa_labels(96, 128, colour)
        frames *= np.where(labels == 'R', .8, np.where(labels == 'B', .6, 1.))
    path = write_ser(tmp_path/'in.ser', frames.astype('u2'),
                     color_id=COLOR_RGGB if colour == 'RGGB' else 0)
    cfg = ReconstructionConfig(device='cpu', threads=2, frame_preselection=False,
        geometry_mode='field', field_center_x=44, field_center_y=48,
        equatorial_radius_px=70, cadence_s=1., reference_index=3, batch_frames=2)
    state = tmp_path/'state.npz'
    with SERSource(path) as src:
        whole = stack_source(src, cfg)
    cancel = False
    def event(result, info):
        nonlocal cancel
        if info['n_processed'] >= 4:
            cancel = True
    with SERSource(path) as src:
        partial = stack_source(src, cfg, state_checkpoint=state, on_event=event,
                               should_cancel=lambda: cancel)
    assert partial.incomplete and partial.n_used == 4
    with SERSource(path) as src:
        continued = stack_source(src, cfg, resume_from=state)
    assert continued.n_used == whole.n_used == 7
    assert continued.provenance['reference_index'] == 3
    for key in ('image', 'coverage', 'validity'):
        np.testing.assert_array_equal(getattr(continued, key), getattr(whole, key))
