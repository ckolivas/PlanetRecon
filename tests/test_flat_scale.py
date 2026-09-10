"""Flat-field illumination units must not change recovered brightness or detail."""
from copy import deepcopy
from dataclasses import replace
import json

import numpy as np
import pytest

from planetrecon.calibration import Calibration, apply_calibration


@pytest.mark.parametrize('scale', [1e-200, 1e-8, 1e-6, 1., 1e200])
@pytest.mark.parametrize('rgb', [False, True])
def test_flat_recovers_known_scene_independently_of_illumination_units(scale, rgb):
    y, x = np.indices((32, 40), dtype=float)
    sensitivity = .5 + x/39  # Median sensitivity is one.
    truth = 100 + 20*np.cos(x/3) + 30*np.sin(y/4)
    if rgb:
        truth = truth[..., None] * [.8, 1., .6]
        sensitivity = np.broadcast_to(sensitivity[..., None], truth.shape).copy()
    bias, dark, gain = 10., 3., 2.
    raw = truth*sensitivity + bias + dark
    original = raw.copy()
    flat = sensitivity*scale
    actual, info = apply_calibration(raw, Calibration(
        bias=np.full_like(raw, bias), dark=np.full_like(raw, dark),
        flat=flat, gain_e_per_adu=gain))
    np.testing.assert_allclose(actual, truth*gain, rtol=1e-14, atol=1e-12)
    np.testing.assert_array_equal(raw, original)
    np.testing.assert_array_equal(flat, sensitivity*scale)
    assert info['units'] == 'e-'


def test_flat_does_not_clip_negative_bias_subtracted_signal():
    flat = np.array([[.5, 1., 1.5], [.5, 1., 1.5]])*1e-8
    actual, _ = apply_calibration(np.ones((2, 3)),
                                 Calibration(bias=np.full((2, 3), 2.), flat=flat))
    np.testing.assert_allclose(actual, -1 / np.array([[.5, 1., 1.5]]*2))


@pytest.mark.parametrize('flat', [np.zeros((5, 6)), -np.ones((5, 6)),
                                  np.full((5, 6), np.nan), np.ones((5, 1))])
def test_explicit_calibration_refuses_invalid_flat_instead_of_clamping_or_broadcasting(flat):
    with pytest.raises(ValueError, match='flat must be finite, positive'):
        apply_calibration(np.ones((5, 6)), Calibration(flat=flat))


@pytest.mark.parametrize('color_id', [0, 8, 100])
@pytest.mark.parametrize('device', ['cpu', pytest.param('gpu', marks=pytest.mark.hardware)])
def test_local_stack_and_screening_are_invariant_to_flat_scale(tmp_path, color_id, device):
    if device == 'gpu':
        import torch
        if not torch.cuda.is_available():
            pytest.skip('CUDA unavailable')
    from test_local_alignment import capture, config
    from planetrecon.io.ser import SERSource, write_ser
    from planetrecon.pipeline.baseline import stack_source
    from planetrecon.pipeline.preprocess_cache import preprocess_source

    # Keep at least four copies of every quality level so the upper-half cut
    # still has enough observations to construct a local template.
    with SERSource(capture(tmp_path, color_id)) as original:
        frames = np.stack([original.read_raw(i) for i in range(original.n_frames())])
    path = write_ser(tmp_path/'flat.ser', np.repeat(frames, 4, axis=0), color_id=color_id)
    with SERSource(path) as source:
        cfg = replace(config(stack_percent=50), device=device)
        y, x = np.indices(source.frame_shape()[:2], dtype=float)
        flat = .75 + x/256 + y/512
        if color_id == 100:
            flat = np.broadcast_to(flat[..., None], source.frame_shape()).copy()
        selections, results = [], []
        for scale in (1., 2.**-30):
            cal = Calibration(flat=flat*scale)
            selections.append(preprocess_source(source, cfg, calibration=cal))
            results.append(stack_source(source, cfg, calibration=cal))
        np.testing.assert_array_equal(selections[0].measurements, selections[1].measurements)
        np.testing.assert_array_equal(selections[0].accepted, selections[1].accepted)
        assert results[0].n_used == results[1].n_used >= 4
        assert results[0].backend == ('cuda' if device == 'gpu' else 'cpu')
        for name in ('image', 'coverage', 'validity'):
            np.testing.assert_array_equal(getattr(results[0], name), getattr(results[1], name))


def test_old_flat_cache_and_checkpoint_are_refused_but_current_resume_is_exact(tmp_path):
    from test_local_alignment import capture, config
    from planetrecon.io.ser import SERSource
    from planetrecon.pipeline.baseline import stack_source
    from planetrecon.pipeline.preprocess_cache import (
        load_cache, preprocess_source, save_cache, selection_digest,
    )

    with SERSource(capture(tmp_path)) as source:
        cfg = config()
        cal = Calibration(flat=np.full(source.frame_shape(), 1e-8))
        selection = preprocess_source(source, cfg, calibration=cal)
        checkpoint = tmp_path/'state.npz'
        whole = stack_source(source, cfg, calibration=cal, state_checkpoint=checkpoint)
        resumed = stack_source(source, cfg, calibration=cal, resume_from=checkpoint)
        np.testing.assert_array_equal(resumed.image, whole.image)
        np.testing.assert_array_equal(resumed.coverage, whole.coverage)

        old = replace(selection, identity=deepcopy(selection.identity))
        old.identity['calibration']['flat'].pop('normalisation')
        old.digest = selection_digest(old)
        old_cache = tmp_path/'old-cache.npz'
        save_cache(old_cache, old, source, cfg)
        loaded, status = load_cache(source, cfg, cal, path=old_cache)
        assert loaded is None and status['status'] == 'stale'
        loaded, status = load_cache(source, cfg, cal)
        assert loaded is not None and status['status'] == 'ready'

        with np.load(checkpoint) as data:
            payload = {key: data[key].copy() for key in data.files}
        meta = json.loads(str(payload['metadata']))
        meta['identity']['capture']['calibration']['flat'].pop('normalisation')
        payload['metadata'] = json.dumps(meta)
        old_checkpoint = tmp_path/'old-state.npz'
        np.savez_compressed(old_checkpoint, **payload)
        with pytest.raises(ValueError, match='identity/configuration mismatch'):
            stack_source(source, cfg, calibration=cal, resume_from=old_checkpoint)
