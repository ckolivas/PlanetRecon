"""Repeated angular detail cannot identify a unique physical field rotation."""
import json

import numpy as np
import pytest

from planetrecon.geometry.fit import estimate_field_angle
from planetrecon.io.source import ArraySource
from planetrecon.pipeline.baseline import stack_source
from planetrecon.reconstruction import ReconstructionConfig


def scene(angle, repeats=2, asymmetry=0.):
    y, x = np.indices((192, 192), dtype=float)
    radius = np.hypot(x+.5-96, 96-y-.5)
    theta = np.arctan2(96-y-.5, x+.5-96)-angle
    envelope = np.exp(-((radius-48)/17)**2)
    image = (radius < 80)*(1+envelope*(.3*np.cos(repeats*theta)
                                      + asymmetry*np.sin(theta+.4)))
    return image, radius


def config(**kwargs):
    return ReconstructionConfig(device='cpu', threads=2, frame_preselection=False,
        geometry_mode='field', field_center_x=96, field_center_y=96,
        equatorial_radius_px=80, **kwargs)


@pytest.mark.parametrize('repeats', [2, 3, 4])
@pytest.mark.parametrize('angle', [-.07, .07])
def test_equal_rotation_branches_are_unconstrained(repeats, angle):
    estimate = estimate_field_angle(scene(0., repeats)[0], scene(angle, repeats)[0], 96, 96, 80)
    assert 'roll_ambiguous' in estimate['degeneracy']
    assert 'roll_unconstrained' in estimate['degeneracy']


@pytest.mark.parametrize('repeats', [2, 3, 4])
@pytest.mark.parametrize('angle', [-.07, .07])
def test_observed_asymmetry_resolves_the_correct_branch(repeats, angle):
    reference = scene(0., repeats, .08)[0]
    estimate = estimate_field_angle(reference, scene(angle, repeats, .08)[0], 96, 96, 80)
    assert not estimate['degeneracy']
    assert estimate['angle_rad'] == pytest.approx(angle, abs=.001)


@pytest.mark.parametrize('colour', ['mono', 'RGB', 'RGGB', 'GRBG', 'GBRG', 'BGGR'])
def test_ambiguous_automatic_run_refuses_before_output_but_known_rate_works(tmp_path, colour):
    from planetrecon.detector import cfa_labels
    rate = -.07/8
    frames = np.array([scene(rate*i)[0] for i in range(9)])
    truth, radius = scene(0.)
    if colour != 'mono':
        truth = truth[..., None]*[.8, 1., .6]
        if colour == 'RGB':
            frames = frames[..., None]*[.8, 1., .6]
        else:
            labels = cfa_labels(192, 192, colour)
            frames *= np.where(labels == 'R', .8, np.where(labels == 'B', .6, 1.))
    source = ArraySource(frames, color_mode=colour, bit_depth=32, timestamps=np.arange(9.))
    checkpoint = tmp_path/'state.npz'
    checkpoint.write_bytes(b'existing checkpoint must survive failed preflight')
    events = []
    with pytest.raises(ValueError, match='ambiguous'):
        stack_source(source, config(), state_checkpoint=checkpoint,
                     on_event=lambda *args: events.append(args))
    assert not events
    assert checkpoint.read_bytes() == b'existing checkpoint must survive failed preflight'
    corrected = stack_source(source, config(field_rate_rad_s=rate))
    zero = stack_source(source, config(field_rate_rad_s=0.))
    roi = (radius > 30) & (radius < 65)
    assert corrected.n_used == zero.n_used == 9
    assert corrected.n_rejected == zero.n_rejected == 0
    assert corrected.validity[roi].all()
    assert np.linalg.norm((corrected.image-truth)[roi]) < .5*np.linalg.norm((zero.image-truth)[roi])


def test_inferred_rate_resume_binds_peak_uniqueness_policy(tmp_path):
    from planetrecon.io.ser import write_ser, SERSource
    frames = np.array([scene(.01*i, asymmetry=.08)[0] for i in range(5)])
    path = write_ser(tmp_path/'in.ser', (frames*10000).astype('u2'))
    checkpoint = tmp_path/'state.npz'
    cfg = config(cadence_s=1.)
    with SERSource(path) as source:
        whole = stack_source(source, cfg, state_checkpoint=checkpoint)
        restored = stack_source(source, cfg, resume_from=checkpoint)
        np.testing.assert_array_equal(restored.image, whole.image)
        np.testing.assert_array_equal(restored.coverage, whole.coverage)
        with np.load(checkpoint) as data:
            payload = {key: data[key].copy() for key in data.files}
        metadata = json.loads(str(payload['metadata']))
        assert metadata['identity']['geometry'].pop('field_peak_uniqueness')
        payload['metadata'] = json.dumps(metadata)
        np.savez_compressed(tmp_path/'old.npz', **payload)
        with pytest.raises(ValueError, match='identity/configuration'):
            stack_source(source, cfg, resume_from=tmp_path/'old.npz')
