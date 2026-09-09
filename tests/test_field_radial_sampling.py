"""Narrow observed detail must not disappear between polar sampling radii."""
from dataclasses import replace

import numpy as np
import pytest

from planetrecon.detector import cfa_labels
from planetrecon.geometry.fit import estimate_field_angle
from planetrecon.io.source import ArraySource
from planetrecon.pipeline.baseline import stack_source
from planetrecon.reconstruction import ReconstructionConfig


def narrow_scene(angle, scale=1., angular=True):
    # Independent continuous scene: a resolved, roughly three-pixel-wide band
    # contains asymmetric detail. Its detector width stays fixed as size changes.
    size = int(384*scale)
    y, x = np.indices((size, size), dtype=float)
    radius = np.hypot(x+.5-size/2, size/2-y-.5)
    theta = np.arctan2(size/2-y-.5, x+.5-size/2)-angle
    detail = np.cos(2*theta)+.5*np.sin(3*theta) if angular else 1.
    image = (radius < 160*scale)*(1+.4*np.exp(-.5*((radius-115*scale)/1.2)**2)*detail)
    return image, radius


@pytest.mark.parametrize('scale', [1., 1.5])
@pytest.mark.parametrize('angle', [-.07, -.017, .017, .07])
def test_previously_unresolved_narrow_detail_recovers_rotation(scale, angle):
    reference, _ = narrow_scene(0., scale)
    frame, _ = narrow_scene(angle, scale)
    result = estimate_field_angle(reference, frame, 192*scale, 192*scale, 160*scale)
    assert not result['degeneracy']
    assert result['angle_rad'] == pytest.approx(angle, abs=.001)


@pytest.mark.parametrize('scale', [.5, 1., 1.5])
def test_radial_brightness_alone_still_cannot_constrain_roll(scale):
    image, _ = narrow_scene(0., scale, angular=False)
    result = estimate_field_angle(image, image, 192*scale, 192*scale, 160*scale)
    assert 'roll_unconstrained' in result['degeneracy']


@pytest.mark.parametrize('colour', ['mono', 'RGB', 'RGGB'])
@pytest.mark.parametrize('direction', [-1., 1.])
def test_narrow_detail_enables_correct_automatic_stack(colour, direction):
    rate = direction*.07/8
    frames = np.array([narrow_scene(rate*i)[0] for i in range(9)])
    truth, radius = narrow_scene(0.)
    if colour != 'mono':
        truth = truth[..., None]*[.8, 1., .6]
        if colour == 'RGB':
            frames = frames[..., None]*[.8, 1., .6]
        else:
            labels = cfa_labels(384, 384, colour)
            frames *= np.where(labels == 'R', .8, np.where(labels == 'B', .6, 1.))
    src = ArraySource(frames, color_mode=colour, bit_depth=32, timestamps=np.arange(9.))
    cfg = ReconstructionConfig(device='cpu', threads=2, frame_preselection=False,
        geometry_mode='field', field_center_x=192, field_center_y=192,
        equatorial_radius_px=160)
    corrected = stack_source(src, cfg)
    oracle = stack_source(src, replace(cfg, field_rate_rad_s=rate))
    zero = stack_source(src, replace(cfg, field_rate_rad_s=0.))
    mask = (radius > 110) & (radius < 120)
    def error(result):
        assert result.n_used == 9 and result.n_rejected == 0
        assert result.validity[mask].all()
        return np.sqrt(np.mean((result.image[mask]-truth[mask])**2))
    assert corrected.provenance['geometry']['field_rate_rad_s'] == pytest.approx(rate, rel=.02)
    assert error(corrected) < error(zero)
    assert error(corrected) < 1.05*error(oracle)


@pytest.mark.parametrize('seed', [732, 818])
def test_uncorrelated_noise_does_not_supply_a_retry_angle(seed):
    rng = np.random.default_rng(seed)
    frames = 1.+rng.normal(0., .02, (2, 384, 384))
    result = estimate_field_angle(frames[0], frames[1], 192, 192, 160)
    assert 'roll_unconstrained' in result['degeneracy']


def test_sparse_only_checkpoint_cannot_resume_with_retry(tmp_path):
    import json
    from planetrecon.io.ser import write_ser, SERSource
    frames = np.array([narrow_scene(.02*i)[0] for i in range(3)])*10000
    path = write_ser(tmp_path/'in.ser', frames.astype('u2'))
    cfg = ReconstructionConfig(device='cpu', threads=2, frame_preselection=False,
        geometry_mode='field', field_center_x=192, field_center_y=192,
        equatorial_radius_px=160, cadence_s=1.)
    checkpoint = tmp_path/'state.npz'
    with SERSource(path) as src:
        stack_source(src, cfg, state_checkpoint=checkpoint)
    with np.load(checkpoint, allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    metadata = json.loads(str(arrays['metadata']))
    assert metadata['identity']['geometry'].pop('field_polar_sampling')
    arrays['metadata'] = json.dumps(metadata)
    np.savez(checkpoint, **arrays)
    with SERSource(path) as src:
        with pytest.raises(ValueError, match='identity/configuration mismatch'):
            stack_source(src, cfg, resume_from=checkpoint)
