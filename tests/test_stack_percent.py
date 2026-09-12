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
    return ReconstructionConfig(device='cpu', threads=2, batch_frames=2,
                                **{'frame_selection_mode': 'frame_count', **kwargs})


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


@pytest.mark.parametrize('mode,expected', [('frame_count', 6), ('quality_range', 3)])
@pytest.mark.parametrize('geometry', ['none', 'field', 'surface', 'combined', 'saturn'])
def test_selection_applies_to_geometry_and_resume(tmp_path, geometry, mode, expected):
    path, _ = capture(tmp_path)
    cfg = config(stack_percent=50, frame_selection_mode=mode, geometry_mode=geometry, cadence_s=.1,
                 field_rate_rad_s=0., surface_rate_rad_s=0., field_center_x=40., field_center_y=32.,
                 equatorial_radius_px=22., sub_obs_lat_rad=.4,
                 ring_inner_radius_px=26. if geometry == 'saturn' else None,
                 ring_outer_radius_px=30. if geometry == 'saturn' else None)
    with SERSource(path) as source:
        preprocess_source(source, cfg)
        whole = stack_source(source, cfg)
        assert whole.n_used == expected and whole.n_rejected == 12 - expected
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
        with pytest.raises(ValueError, match='identity|configuration|mismatch'):
            stack_source(source, replace(cfg, frame_selection_mode=(
                'quality_range' if mode == 'frame_count' else 'frame_count')), resume_from=checkpoint)


def test_explicit_reference_outside_selection_is_not_silently_replaced(tmp_path):
    path, _ = capture(tmp_path)
    with SERSource(path) as source:
        preprocess_source(source, config())
        with pytest.raises(ValueError, match='reference frame was rejected'):
            stack_source(source, config(stack_percent=25, reference_index=1))


@pytest.mark.parametrize('mode,expected', [('frame_count', 6), ('quality_range', 3)])
def test_cli_and_config_roundtrip(tmp_path, mode, expected):
    from planetrecon.cli import main
    from planetrecon.result import load_snapshot
    path, _ = capture(tmp_path)
    with SERSource(path) as source: preprocess_source(source, config())
    cfg = config(stack_percent=50, frame_selection_mode=mode)
    assert ReconstructionConfig.from_dict(cfg.to_dict()) == cfg
    assert main(['--threads', '2', 'stack', '--path', str(path), '--device', 'cpu',
                 '--no-local-alignment', '--selection-mode', mode, '--stack-percent', '50',
                 '--out', str(tmp_path/'stack')]) == 0
    assert load_snapshot(tmp_path/'stack/stack.npz').n_used == expected


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
@pytest.mark.parametrize('mode', ['frame_count', 'quality_range'])
def test_gpu_uses_same_best_bayer_frames_as_cpu(tmp_path, mode):
    import torch
    if not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    with SERSource(bayer_capture(tmp_path)) as source:
        selected = preprocess_source(source, config())
        cpu = stack_source(source, config(stack_percent=25, frame_selection_mode=mode), preprocessing=selected)
        gpu = stack_source(source, replace(config(stack_percent=25, frame_selection_mode=mode), device='gpu'), preprocessing=selected)
    assert gpu.backend == 'cuda' and gpu.n_used == cpu.n_used == 3
    np.testing.assert_array_equal(gpu.validity, cpu.validity)
    np.testing.assert_allclose(gpu.image, cpu.image, rtol=1e-12, atol=1e-10)
    np.testing.assert_allclose(gpu.coverage, cpu.coverage, rtol=1e-12, atol=1e-10)


def quality_selection(scores, accepted=None):
    scores = np.asarray(scores, dtype=float)
    measurements = np.zeros((len(scores), 4))
    measurements[:, 0] = scores
    return FrameSelection(np.isfinite(scores) if accepted is None else np.asarray(accepted, dtype=bool),
                          measurements, {})


def test_quality_range_midpoint_is_not_half_the_frame_count():
    selection = quality_selection([0, 10, 60, 70, 80, 90, 95, 100])
    assert best_frame_mask(selection, 50, 'quality_range').sum() == 6
    assert best_frame_mask(selection, 50, 'frame_count').sum() == 4
    selection = quality_selection([0, 50, 100])
    np.testing.assert_array_equal(best_frame_mask(selection, 50, 'quality_range'), [False, False, True])


def test_quality_range_uses_capture_extrema_then_applies_screening():
    selection = quality_selection([0, 40, 60, 80, 100, np.nan],
                                  [False, True, True, True, False, False])
    before = selection.accepted.copy()
    np.testing.assert_array_equal(best_frame_mask(selection, 50, 'quality_range'),
                                  [False, False, True, True, False, False])
    np.testing.assert_array_equal(selection.accepted, before)


@pytest.mark.parametrize('scores', [[4, 4, 4], [np.nan, np.nan], [0, 50, 100], [0, 10, 60, 70, 100]])
def test_cached_range_counts_match_masks_including_flat_and_empty_range(scores):
    from planetrecon.pipeline.preprocess import quality_range_counts
    selection = quality_selection(scores)
    report = quality_range_counts(selection)
    assert report['retained_counts'] == [
        int(best_frame_mask(selection, percent, 'quality_range').sum()) for percent in range(1, 101)]
    np.testing.assert_array_equal(best_frame_mask(selection, 100, 'quality_range'), selection.accepted)


def test_range_configuration_is_optional_and_old_percentages_keep_count_meaning():
    assert ReconstructionConfig().stack_percent == 100
    assert ReconstructionConfig().frame_selection_mode == 'quality_range'
    old = config(stack_percent=50).to_dict()
    old.pop('frame_selection_mode')
    assert ReconstructionConfig.from_dict(old).frame_selection_mode == 'frame_count'
    cfg = config(stack_percent=50, frame_selection_mode='quality_range')
    assert ReconstructionConfig.from_dict(cfg.to_dict()) == cfg
    with pytest.raises(ValueError, match='frame_selection_mode'):
        config(frame_selection_mode='unknown')


def test_quality_range_new_runs_reuse_cache_and_record_actual_cutoff(tmp_path):
    from planetrecon.pipeline.preprocess_cache import cache_report
    path, _ = capture(tmp_path)
    with SERSource(path) as source:
        cached = preprocess_source(source, config())
        cache_path = default_cache_path(source)
        before = cache_path.read_bytes()
        report = cache_report(cached)
        result = stack_source(source, config(stack_percent=50, frame_selection_mode='quality_range'))
        detail = result.provenance['preprocessing']['best_frame_selection']
        assert result.n_used == report['quality_range']['retained_counts'][49] == 3
        assert detail['quality_cutoff'] == pytest.approx(
            (cached.measurements[:, 0].min() + cached.measurements[:, 0].max()) / 2)
        assert stack_source(source, config(stack_percent=100)).n_used == 12
        assert cache_path.read_bytes() == before


def test_gui_range_counts_and_mode_changes_apply_to_next_run():
    from planetrecon.pipeline.preprocess import quality_range_counts
    from planetrecon.gui.app import MainWindow, create_app
    app = create_app([])
    win = MainWindow()
    try:
        selection = quality_selection([0, 10, 60, 70, 80, 90, 95, 100])
        win.preprocessing_info = dict(status='ready', accepted=8, n_total=8, excluded=0,
                                      quality=0, shape=0, quality_shape_overlap=0, other=0,
                                      quality_range=quality_range_counts(selection))
        percent = win.controls.fields['stack_percent']
        percent.lineEdit().setText('50')
        assert win.controls.configuration().stack_percent == 50
        win._refresh_preprocessing()
        assert '6 frames (75.0% of capture)' in win.preprocessing_label.text()
        mode = win.controls.fields['frame_selection_mode']
        mode.setCurrentIndex(mode.findData('frame_count'))
        assert '4 screened frames' in win.preprocessing_label.text()
        assert win.controls.configuration().frame_selection_mode == 'frame_count'
        assert win.controls.stack_percent_label.text() == 'Best retained frames (%)'
        mode.setCurrentIndex(mode.findData('quality_range'))
        assert win.controls.configuration().frame_selection_mode == 'quality_range'
        assert win.controls.stack_percent_label.text() == 'Upper quality range (%)'
        percent.setValue(100)
        assert 'all 8 screened frames' in win.preprocessing_label.text()
        win.controls.fields['frame_preselection'].setChecked(False)
        assert not mode.isEnabled() and not percent.isEnabled()
    finally:
        win._shutdown()
        win.window.close()
        app.processEvents()
