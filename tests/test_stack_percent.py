from dataclasses import replace

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from planetrecon.io.ser import SERSource, write_ser
from planetrecon.pipeline.baseline import stack_source
from planetrecon.pipeline.preprocess import FrameSelection, best_frame_mask
from planetrecon.pipeline.preprocess_cache import preprocess_source, default_cache_path
from planetrecon.reconstruction import ReconstructionConfig


def config(**kwargs):
    return ReconstructionConfig(device='cpu', threads=2, batch_frames=2, **kwargs)


def capture(tmp_path):
    y, x = np.indices((64, 80))
    disc = ((x-40)/22)**2 + ((y-32)/19)**2 < 1
    truth = gaussian_filter(disc * (5000 + 1000*np.cos(y/2) + 500*np.cos(x/3)), .45)
    sharp = truth.astype('u2')
    soft = gaussian_filter(truth, 1.6).astype('u2')
    # Best frames occur late, exercising selection across batch boundaries.
    return write_ser(tmp_path/'seeing.ser', np.stack([soft]*9 + [sharp]*3)), sharp


def test_selection_keeps_only_screened_frames_and_stable_ties():
    measurements = np.zeros((6, 4))
    measurements[:, 0] = [1, 8, 99, 8, 2, 3]
    selection = FrameSelection(np.array([True, True, False, True, True, True]), measurements, {})
    before = selection.accepted.copy()
    np.testing.assert_array_equal(np.flatnonzero(best_frame_mask(selection, 21)), [1, 3])
    np.testing.assert_array_equal(best_frame_mask(selection, 100), before)
    np.testing.assert_array_equal(selection.accepted, before)
    assert best_frame_mask(selection, 1).sum() == 1


@pytest.mark.parametrize('value', [0, 101, -1, True, 25.5, None])
def test_invalid_percentage_rejected(value):
    with pytest.raises(ValueError, match='stack_percent'):
        config(stack_percent=value)


def test_best_percent_reduces_seeing_blur_without_sharpening_and_reuses_cache(tmp_path):
    path, truth = capture(tmp_path)
    with SERSource(path) as source:
        cached = preprocess_source(source, config())
        original_mask = cached.accepted.copy()
        assert original_mask.all()
        cache_path = default_cache_path(source)
        cache_bytes = cache_path.read_bytes()
        all_frames = stack_source(source, config(), preprocessing=cached)
        best = stack_source(source, config(stack_percent=25), preprocessing=cached)
        half = stack_source(source, config(stack_percent=50), preprocessing=cached)
        assert (all_frames.n_used, half.n_used, best.n_used) == (12, 6, 3)
        assert best.n_rejected == 9 and best.provenance['reference_index'] == 9
        mask = truth > 500
        assert np.linalg.norm((best.image-truth)[mask]) < .1*np.linalg.norm((all_frames.image-truth)[mask])
        assert best.provenance['preprocessing']['best_frame_selection']['additional_exclusions'] == 9
        assert not best.provenance.get('sharpening')
        np.testing.assert_array_equal(cached.accepted, original_mask)
        assert cache_path.read_bytes() == cache_bytes
        repeated = stack_source(source, config(stack_percent=25))
        np.testing.assert_array_equal(repeated.image, best.image)


def test_percentage_requires_cache_and_disabled_cache_uses_all_frames(tmp_path):
    path, _ = capture(tmp_path)
    with SERSource(path) as source:
        with pytest.raises(ValueError, match='Run Preprocess first'):
            stack_source(source, config(stack_percent=25))
        assert stack_source(source, config(stack_percent=25, frame_preselection=False)).n_used == 12


@pytest.mark.parametrize('geometry', ['none', 'field', 'surface', 'combined', 'saturn'])
def test_selection_applies_to_geometry_and_resume(tmp_path, geometry):
    path, _ = capture(tmp_path)
    cfg = config(stack_percent=50, geometry_mode=geometry, cadence_s=.1,
                 field_rate_rad_s=0., surface_rate_rad_s=0., field_center_x=40., field_center_y=32.,
                 equatorial_radius_px=22., sub_obs_lat_rad=.4,
                 ring_inner_radius_px=26. if geometry == 'saturn' else None,
                 ring_outer_radius_px=30. if geometry == 'saturn' else None)
    with SERSource(path) as source:
        preprocess_source(source, cfg)
        whole = stack_source(source, cfg)
        assert whole.n_used == 6 and whole.n_rejected == 6
        stop = False
        def event(result, info):
            nonlocal stop
            if info['n_processed'] >= 4: stop = True
        checkpoint = tmp_path/'resume.npz'
        partial = stack_source(source, cfg, should_cancel=lambda: stop, on_event=event,
                               state_checkpoint=checkpoint)
        assert partial.incomplete
        resumed = stack_source(source, cfg, resume_from=checkpoint)
        np.testing.assert_array_equal(resumed.image, whole.image)
        np.testing.assert_array_equal(resumed.coverage, whole.coverage)
        with pytest.raises(ValueError, match='identity|configuration|mismatch'):
            stack_source(source, replace(cfg, stack_percent=25), resume_from=checkpoint)


def test_explicit_reference_outside_selection_is_not_silently_replaced(tmp_path):
    path, _ = capture(tmp_path)
    with SERSource(path) as source:
        preprocess_source(source, config())
        with pytest.raises(ValueError, match='reference frame was rejected'):
            stack_source(source, config(stack_percent=25, reference_index=1))


def test_cli_and_config_roundtrip(tmp_path):
    from planetrecon.cli import main
    from planetrecon.result import load_snapshot
    path, _ = capture(tmp_path)
    with SERSource(path) as source: preprocess_source(source, config())
    cfg = config(stack_percent=25)
    assert ReconstructionConfig.from_dict(cfg.to_dict()) == cfg
    assert main(['--threads', '2', 'stack', '--path', str(path), '--device', 'cpu',
                 '--stack-percent', '25', '--out', str(tmp_path/'stack')]) == 0
    assert load_snapshot(tmp_path/'stack/stack.npz').n_used == 3


def test_gui_commits_typed_percentage_for_each_new_run():
    from PySide6.QtWidgets import QApplication
    from planetrecon.gui.controls import ConfigControls
    app = QApplication.instance() or QApplication([])
    controls = ConfigControls(config())
    edit = controls.fields['stack_percent']
    edit.lineEdit().setText('25')
    assert controls.configuration().stack_percent == 25
    edit.lineEdit().setText('50')
    assert controls.configuration().stack_percent == 50
    controls.fields['frame_preselection'].setChecked(False)
    assert not edit.isEnabled()
    controls.fields['frame_preselection'].setChecked(True)
    assert edit.isEnabled() and controls.configuration().stack_percent == 50
    assert 'blur' in edit.toolTip()
    controls.close()


def bayer_capture(tmp_path):
    path, _ = capture(tmp_path)
    with SERSource(path) as source:
        frames = np.stack([source.read_raw(i) for i in range(source.n_frames())]).astype(float)
    frames[:, ::2, ::2] *= .8
    frames[:, 1::2, 1::2] *= .6
    return write_ser(tmp_path/'colour.ser', frames.astype('u2'), color_id=8)


def test_best_bayer_frames_produce_rgb_with_valid_channels(tmp_path):
    with SERSource(bayer_capture(tmp_path)) as source:
        selected = preprocess_source(source, config())
        result = stack_source(source, config(stack_percent=25), preprocessing=selected)
    assert result.n_used == 3 and result.channel_order == 'RGB'
    assert result.validity[20:44, 28:52].all()
    assert np.isfinite(result.image).all()
    averages = result.image[20:44, 28:52].mean(axis=(0, 1))
    np.testing.assert_allclose(averages / averages[1], [.8, 1., .6], atol=.03)


@pytest.mark.hardware
def test_gpu_uses_same_best_bayer_frames_as_cpu(tmp_path):
    import torch
    if not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    with SERSource(bayer_capture(tmp_path)) as source:
        selected = preprocess_source(source, config())
        cpu = stack_source(source, config(stack_percent=25), preprocessing=selected)
        gpu = stack_source(source, replace(config(stack_percent=25), device='gpu'), preprocessing=selected)
    assert gpu.backend == 'cuda' and gpu.n_used == cpu.n_used == 3
    np.testing.assert_array_equal(gpu.validity, cpu.validity)
    np.testing.assert_allclose(gpu.image, cpu.image, rtol=1e-12, atol=1e-10)
    np.testing.assert_allclose(gpu.coverage, cpu.coverage, rtol=1e-12, atol=1e-10)
