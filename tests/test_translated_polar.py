"""Independently sampled moving crops qualify joint angle/translation fitting."""
from dataclasses import replace

import numpy as np
import pytest

from planetrecon.detector import cfa_labels
from planetrecon.geometry.fit import estimate_field_angle
from planetrecon.io.source import ArraySource
from planetrecon.pipeline.baseline import stack_source
from planetrecon.reconstruction import ReconstructionConfig
from test_field_observed_support import cropped_scene


OFFSETS = [(0, 0), (3.4, -2.3), (-4.3, 3.4), (2.4, 3.3), (-5.3, -3.4),
           (3.3, 2.4), (-3.4, -2.3), (2.3, -4.4), (-5.4, 3.3)]


def config():
    return ReconstructionConfig(device='cpu', threads=2, frame_preselection=False,
        geometry_mode='field', field_center_x=44, field_center_y=48,
        equatorial_radius_px=70)


@pytest.mark.parametrize('offset', [(-25.4, 3.3), (35.3, -23.4), (-5.4, 3.3)])
@pytest.mark.parametrize('direction', [-1., 1.])
def test_translated_polar_samples_recover_signed_angle(offset, direction):
    dx, dy = offset
    reference, _ = cropped_scene(0.)
    frame, _ = cropped_scene(direction*.07, dx=dx, dy=dy)
    result = estimate_field_angle(reference, frame, 44, 48, 70,
                                  frame_center=(44+dx, 48+dy))
    assert not result['degeneracy']
    assert result['angle_rad'] == pytest.approx(direction*.07, abs=.002)


@pytest.mark.parametrize('frame_center', [(2., 48.), (130., 48.), (44., -2.)])
def test_unobserved_translated_interior_cannot_supply_angle(frame_center):
    image, _ = cropped_scene(0.)
    result = estimate_field_angle(image, image, 44, 48, 70, frame_center=frame_center)
    assert 'roll_unconstrained' in result['degeneracy']


@pytest.mark.parametrize('colour', ['mono', 'RGB', 'RGGB', 'GRBG', 'GBRG', 'BGGR'])
@pytest.mark.parametrize('direction', [-1., 1.])
def test_drifting_crop_recovers_automatic_field_stack(colour, direction):
    rate = direction*np.deg2rad(4)/8
    frames = np.array([cropped_scene(rate*i, dx=dx, dy=dy)[0]
                       for i, (dx, dy) in enumerate(OFFSETS)])
    truth = frames[0].copy()
    if colour != 'mono':
        truth = truth[..., None]*[.8, 1., .6]
        if colour == 'RGB':
            frames = frames[..., None]*[.8, 1., .6]
        else:
            labels = cfa_labels(96, 128, colour)
            frames *= np.where(labels == 'R', .8, np.where(labels == 'B', .6, 1.))
    src = ArraySource(frames, color_mode=colour, bit_depth=32, timestamps=np.arange(9.))
    cfg = config()
    corrected = stack_source(src, cfg)
    oracle = stack_source(src, replace(cfg, field_rate_rad_s=rate))
    zero = stack_source(src, replace(cfg, field_rate_rad_s=0.))
    _, radius = cropped_scene(0.)
    mask = (radius > 10) & (radius < 32)
    def error(result):
        assert result.n_used == 9 and result.n_rejected == 0
        assert result.validity[mask].all()
        return np.sqrt(np.mean((result.image[mask]-truth[mask])**2))
    assert corrected.provenance['geometry']['field_rate_rad_s'] == pytest.approx(rate, rel=.04)
    # CFA sampling has its own error floor. Compare it tightly to the same
    # known-rate colour reconstruction, rather than demanding mono accuracy.
    if colour in ('mono', 'RGB'):
        assert error(corrected) < .1*error(zero)
        assert error(corrected) < 2*error(oracle)
    else:
        assert error(corrected) < error(zero)
        assert error(corrected) < 1.1*error(oracle)


def test_nonconverging_joint_fit_cannot_authorize_motion(monkeypatch):
    from planetrecon.pipeline.geometry_stack import prepare_geometry
    calls = []
    def oscillating(*args):
        calls.append(1)
        return (2. if len(calls) % 2 else -2., 0.)
    monkeypatch.setattr('planetrecon.pipeline.field_align.field_displacement', oscillating)
    src = ArraySource(np.array([cropped_scene(i*.02)[0] for i in range(3)]),
                      bit_depth=32, timestamps=np.arange(3.))
    _, _, diagnostic, _ = prepare_geometry(src, config())
    assert len(calls) == 128  # Two bounded, nonconvergent sample fits.
    assert diagnostic['field_sample_shifts_px'] == [[0., 0.], None, None]
    assert diagnostic['unavailable_motion']
    with pytest.raises(ValueError, match='field rotation could not be estimated'):
        stack_source(src, config())


def test_old_field_estimator_checkpoint_cannot_resume(tmp_path):
    import json
    from planetrecon.io.ser import write_ser, SERSource
    frames = np.array([cropped_scene(.02*i, dx=dx, dy=dy)[0]
                       for i, (dx, dy) in enumerate(OFFSETS[:3])])*10000
    path = write_ser(tmp_path/'in.ser', frames.astype('u2'))
    cfg = replace(config(), cadence_s=1.)
    checkpoint = tmp_path/'state.npz'
    with SERSource(path) as src:
        stack_source(src, cfg, state_checkpoint=checkpoint)
    with np.load(checkpoint, allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    metadata = json.loads(str(arrays['metadata']))
    assert metadata['identity']['geometry'].pop('field_estimation')
    arrays['metadata'] = json.dumps(metadata)
    np.savez(checkpoint, **arrays)
    with SERSource(path) as src:
        with pytest.raises(ValueError, match='identity/configuration mismatch'):
            stack_source(src, cfg, resume_from=checkpoint)
