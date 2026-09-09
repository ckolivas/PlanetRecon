"""Tilted structure needs a coupled subpixel translation estimate."""
import numpy as np
import pytest

from planetrecon.io.source import ArraySource
from planetrecon.pipeline.baseline import stack_source
from planetrecon.pipeline.masked_align import MaskedRegistration
from planetrecon.reconstruction import ReconstructionConfig


def tilted_feature(angle, dx=0., dy=0.):
    # Sample the translated continuous Gaussian directly, without image warps.
    y, x = np.indices((96, 96), dtype=float)
    u = np.cos(angle)*(x-47.5-dx)+np.sin(angle)*(y-47.5-dy)
    v = -np.sin(angle)*(x-47.5-dx)+np.cos(angle)*(y-47.5-dy)
    return np.exp(-.5*((u/6)**2+(v/2)**2))


@pytest.mark.parametrize('angle', [-np.pi/4, -np.pi/6, 0., np.pi/6, np.pi/4])
@pytest.mark.parametrize('offset', [(.4, -.3), (.3, .4), (2.4, -1.3)])
def test_tilted_texture_recovers_both_displacement_components(angle, offset):
    mask = np.zeros((96, 96), bool)
    mask[8:88, 8:88] = True
    matcher = MaskedRegistration(tilted_feature(angle), mask)
    displacement = matcher.displacement(tilted_feature(angle, *offset))
    assert displacement is not None
    np.testing.assert_allclose(displacement, offset, atol=.05)


@pytest.mark.parametrize('offset', [(0., 0.), (3., -2.)])
def test_exact_tilted_matches_keep_integer_displacements(offset):
    mask = np.zeros((96, 96), bool)
    mask[8:88, 8:88] = True
    matcher = MaskedRegistration(tilted_feature(np.pi/4), mask)
    np.testing.assert_array_equal(matcher.displacement(tilted_feature(np.pi/4, *offset)), offset)


@pytest.mark.parametrize('direction', [-1., 1.])
def test_coupled_motion_stack_matches_known_camera_shifts(monkeypatch, direction):
    offsets = [(0, 0), (.4, -.3), (-.3, .4), (.3, .4), (-.4, -.3),
               (.4, .3), (-.3, -.4), (.3, -.4), (-.4, .3)]
    rate = direction*.02
    frames = np.array([tilted_feature(np.pi/4-rate*t, dx, dy)
                       for t, (dx, dy) in enumerate(offsets)])
    src = ArraySource(frames, bit_depth=32, timestamps=np.arange(9.))
    cfg = ReconstructionConfig(device='cpu', threads=2, frame_preselection=False,
        geometry_mode='field', field_center_x=48, field_center_y=48,
        equatorial_radius_px=80, field_rate_rad_s=rate)
    corrected = stack_source(src, cfg)
    monkeypatch.setattr('planetrecon.pipeline.field_align.field_displacement',
        lambda reference, frame, pose, reference_pose, radius: offsets[round(pose.t_s)])
    known = stack_source(src, cfg)
    assert corrected.n_used == known.n_used == 9
    assert corrected.n_rejected == known.n_rejected == 0
    assert corrected.provenance['geometry']['registration_peak'] == 'joint two-dimensional quadratic'
    y, x = np.indices((96, 96))
    mask = (x-47.5)**2+(y-47.5)**2 < 15**2
    assert np.sqrt(np.mean((corrected.image[mask]-known.image[mask])**2)) < .001


def test_old_axis_fit_checkpoint_cannot_resume_with_joint_peak(tmp_path):
    import json
    from planetrecon.io.ser import write_ser, SERSource
    frames = np.array([tilted_feature(np.pi/4-.02*t, .4*t, -.3*t)
                       for t in range(3)])*30000
    path = write_ser(tmp_path/'in.ser', frames.astype('u2'))
    cfg = ReconstructionConfig(device='cpu', threads=2, frame_preselection=False,
        geometry_mode='field', field_center_x=48, field_center_y=48,
        equatorial_radius_px=80, field_rate_rad_s=.02, cadence_s=1.)
    checkpoint = tmp_path/'state.npz'
    with SERSource(path) as src:
        stack_source(src, cfg, state_checkpoint=checkpoint)
    with np.load(checkpoint, allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    metadata = json.loads(str(arrays['metadata']))
    assert metadata['identity']['geometry'].pop('registration_peak') == 'joint two-dimensional quadratic'
    arrays['metadata'] = json.dumps(metadata)
    np.savez(checkpoint, **arrays)
    with SERSource(path) as src:
        with pytest.raises(ValueError, match='identity/configuration mismatch'):
            stack_source(src, cfg, resume_from=checkpoint)
