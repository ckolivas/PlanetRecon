"""Detector padding must not supply evidence for a local deformation."""
import json

import numpy as np
import pytest

from planetrecon.pipeline.local_align import LocalRegistration, pull


def continuous_scene(y, x):
    # Evaluate directly at each detector coordinate, without resampling a frame.
    return (300 + 30*np.cos(x/3) + 35*np.sin(y/4)
            + 20*np.sin((x+y)/7) + 25*np.cos((2*x-y)/9))


@pytest.mark.parametrize('dx,dy', [(50, 0), (-50, 0), (0, 50), (0, -50), (50, 30)])
def test_cropped_translation_does_not_create_local_warp(dx, dy):
    y, x = np.indices((192, 224), dtype=float)
    reference = continuous_scene(y, x)
    frame = continuous_scene(y-dy, x-dx)
    displacement = LocalRegistration(reference).displacement(frame, (dx, dy))
    observed = (x+dx > 2) & (x+dx < 221) & (y+dy > 2) & (y+dy < 189)
    np.testing.assert_array_equal(pull(frame, (dx, dy))[observed], reference[observed])
    assert np.hypot(displacement[0]-dx, displacement[1]-dy).max() < .03
    assert np.sqrt(np.mean((pull(frame, displacement)[observed]-reference[observed])**2)) < .04


def test_observed_interior_still_corrects_spatial_motion_after_large_drift():
    y, x = np.indices((256, 288), dtype=float)
    reference = continuous_scene(y, x)
    # Horizontal distortion depending only on y has an exact independent inverse.
    dx, dy = 50., 20.
    flow = lambda yy: 1.2*np.sin(yy/55)
    frame = continuous_scene(y-dy, x-dx-flow(y-dy))
    displacement = LocalRegistration(reference).displacement(frame, (dx, dy))
    roi = np.s_[75:150, 75:155]
    corrected = pull(frame, displacement)
    global_only = pull(frame, (dx, dy))
    assert np.linalg.norm((corrected-reference)[roi]) < .4*np.linalg.norm((global_only-reference)[roi])


@pytest.mark.parametrize('color_id', [0, 8])
@pytest.mark.parametrize('policy,value', [
    ('local_patch_support', 'complete observed search footprint'),
    ('local_patch_boundary', 'one grid interval taper to global'),
    ('local_patch_peak', 'joint two-dimensional quadratic'),
    ('local_template_boundary', 'normalised support; best-frame fill'),
])
def test_old_local_checkpoint_policy_is_refused(tmp_path, color_id, policy, value):
    from test_local_alignment import capture, config
    from planetrecon.io.ser import SERSource
    from planetrecon.pipeline.baseline import stack_source
    from planetrecon.pipeline.preprocess_cache import preprocess_source

    with SERSource(capture(tmp_path, color_id)) as source:
        cfg = config()
        preprocess_source(source, cfg)
        checkpoint = tmp_path/'state.npz'
        whole = stack_source(source, cfg, state_checkpoint=checkpoint)
        restored = stack_source(source, cfg, resume_from=checkpoint)
        np.testing.assert_array_equal(restored.image, whole.image)
        np.testing.assert_array_equal(restored.coverage, whole.coverage)
        with np.load(checkpoint) as data:
            payload = {name: data[name].copy() for name in data.files}
        metadata = json.loads(str(payload['metadata']))
        assert metadata['identity'].pop(policy) == value
        payload['metadata'] = json.dumps(metadata)
        np.savez_compressed(tmp_path/'old.npz', **payload)
        with pytest.raises(ValueError, match='identity'):
            stack_source(source, cfg, resume_from=tmp_path/'old.npz')


@pytest.mark.hardware
def test_cropped_local_support_matches_cuda():
    import torch
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    y, x = np.indices((192, 224), dtype=float)
    reference = continuous_scene(y, x)
    frame = continuous_scene(y-49.75, x-50.25)
    cpu = LocalRegistration(reference).displacement(frame, (50.25, 49.75))
    gpu = LocalRegistration(reference, use_cuda=True).displacement(frame, (50.25, 49.75))
    np.testing.assert_allclose(cpu, gpu, atol=1e-10, rtol=1e-10)
