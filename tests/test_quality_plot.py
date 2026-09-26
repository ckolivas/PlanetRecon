"""The plot must describe the actual next-run mask, including cache lifecycle."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
import pytest

from planetrecon.pipeline.preprocess import FrameSelection
from planetrecon.pipeline.preprocess_cache import cache_report, quality_plot_data
from test_w14 import gui, pump


def report(scores=(0, 50, 75, 100, 90, np.nan), accepted=(True, True, True, True, False, False)):
    selection = FrameSelection(np.array(accepted), np.array([[q, 1, 1, 0] for q in scores]),
        {'rejected_indices_by_reason': {'low_quality': [], 'width_outlier': [4],
         'height_outlier': [], 'clipped_target': [], 'no_target': [5], 'invalid': [], 'saturated': []}},
        digest='plot-test')
    return {**cache_report(selection), 'frame_quality': quality_plot_data(selection)}


def test_plot_tracks_threshold_masks_order_and_cache_identity(gui):
    app, win = gui
    fields = win.controls.fields
    fields['frame_preselection'].setChecked(True)
    fields['frame_selection_mode'].setCurrentIndex(0)
    fields['stack_percent'].setValue(50)
    info = report()
    win._set_preprocessing(info)
    plot = win.quality_plot
    assert win.preview_tabs.currentWidget() is plot
    assert plot.cutoff == 50
    # Equality at the quality cutoff is excluded, and a sharp shape outlier stays excluded.
    assert plot.selected.tolist() == [False, False, True, True, False, False]
    assert 'width outlier' in plot.frame_description(4)
    assert 'Rank 2/6 (top 33.33% of all frames)' in plot.frame_description(4)
    assert 'unavailable' in plot.frame_description(5)
    plot.order_control.setCurrentIndex(1)
    assert plot.order.tolist() == [3, 4, 2, 1, 0, 5]
    assert plot.quality_ranks.tolist() == [5, 4, 3, 1, 2, 6]
    fields['stack_percent'].setValue(100)
    assert plot.cutoff == 0
    assert plot.selected.tolist() == [True, True, True, True, False, False]
    fields['frame_preselection'].setChecked(False)
    assert plot.cutoff is None and plot.selected.all()
    assert 'disabled' in plot.summary.text()
    fields['frame_preselection'].setChecked(True)
    compact = {key: value for key, value in info.items() if key != 'frame_quality'}
    win._set_preprocessing(compact)
    assert plot.data is info['frame_quality']
    fields['gain_e_per_adu'].setText('2')
    assert plot.selection is None
    assert not plot.quality_ranks.size
    win._set_preprocessing({**compact, 'digest': 'different-capture'})
    assert plot.selection is None
    app.processEvents()


def test_flat_scores_count_ties_and_single_frame(gui):
    app, win = gui
    fields = win.controls.fields
    fields['frame_preselection'].setChecked(True)
    fields['stack_percent'].setValue(50)
    fields['frame_selection_mode'].setCurrentIndex(0)
    win._set_preprocessing(report((7, 7, 7, 7), (True, True, True, True)))
    plot = win.quality_plot
    assert plot.selected.all() and plot.cutoff is None
    assert plot.quality_ranks.tolist() == [1, 2, 3, 4]
    assert 'Rank 2/4 (top 50.00% of all frames)' in plot.frame_description(1)
    fields['frame_selection_mode'].setCurrentIndex(1)
    assert plot.selected.tolist() == [True, True, False, False]
    assert plot.cutoff is None  # A horizontal line cannot represent tied count selection.
    win._set_preprocessing(report((7,), (True,)))
    app.processEvents()
    assert plot.selected.tolist() == [True]
    assert 'Rank 1/1 (top 100.00% of all frames)' in plot.frame_description(0)
    assert not plot.canvas.grab().isNull()
    win._set_preprocessing(report((np.nan,), (False,)))
    assert not plot.selected.any() and plot.cutoff is None
    assert 'unavailable' in plot.frame_description(0)
    assert 'Rank 1/1' in plot.frame_description(0)
    assert not plot.canvas.grab().isNull()


@pytest.mark.parametrize('absolute', [False, True])
@pytest.mark.parametrize('logarithmic', [False, True])
def test_axis_modes_preserve_selection_and_transform_cutoff(gui, absolute, logarithmic):
    app, win = gui
    plot = win.quality_plot
    plot.set_report(report((10, 20, 40, 100, 80, np.nan)), 50, 'quality_range', True)
    selected = plot.selected.copy()
    plot.absolute_control.setChecked(absolute)
    plot.log_control.setChecked(logarithmic)
    assert plot.cutoff == 55
    np.testing.assert_array_equal(plot.selected, selected)
    scores = plot.selection.measurements[:, 0]
    positions = plot.quality_position(scores)
    assert positions[3] == 1
    assert np.isnan(positions[5])
    assert np.all(np.diff(positions[:4]) > 0)
    assert (positions[0] > 0) == absolute
    fraction = .55 if absolute else .5
    expected_cutoff = (1 + np.log10(fraction * 1000)) / 4 if logarithmic else fraction
    assert plot.quality_position(plot.cutoff) == pytest.approx(expected_cutoff)
    assert plot.axis_fraction(0) == 0
    assert plot.axis_ticks()[0] == 0
    assert not plot.canvas.grab().isNull()


@pytest.mark.parametrize('score', [0., 7., 1e-20, np.nan])
def test_log_absolute_flat_or_unavailable_quality(gui, score):
    app, win = gui
    plot = win.quality_plot
    plot.absolute_control.setChecked(True)
    plot.log_control.setChecked(True)
    plot.set_report(report((score,), (np.isfinite(score),)), 50, 'quality_range', True)
    assert plot.cutoff is None
    assert np.isfinite(plot.axis_fraction(plot.axis_ticks())).all()
    if np.isfinite(score):
        assert plot.quality_position(score) == (1 if score > 0 else 0)
    assert not plot.canvas.grab().isNull()


def test_worker_preprocess_and_inspect_deliver_plot_without_export_bloat(gui, tmp_path):
    from test_preprocess_cache import capture
    app, win = gui
    win.path = capture(tmp_path)
    win.controls.fields['frame_preselection'].setChecked(True)
    win._preprocess()
    pump(app, lambda: win.job is None, timeout=20)
    assert not win.error.text()
    assert win.input_image is not None
    assert win.view.currentText() == 'Input'
    assert not win.cancel_btn.isEnabled()
    plot = win.quality_plot
    assert len(plot.selected) == 34
    assert np.flatnonzero(~plot.selection.accepted).tolist() == [32, 33]
    original = plot.selection.measurements.copy()
    win.preprocessing_info = {}
    win._refresh_preprocessing()
    win._capture_ready()
    pump(app, lambda: win.job is None, timeout=20)
    np.testing.assert_array_equal(plot.selection.measurements, original)
    win._run()
    pump(app, lambda: win.job is None, timeout=20)
    assert not win.error.text()
    assert len(plot.selected) == 34
    assert 'frame_quality' not in win.last_result.provenance['preprocessing_cache']


def test_graph_clicks_load_original_frames_in_both_orders(gui, tmp_path):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from planetrecon.io.ser import write_ser
    app, win = gui
    frames = np.arange(6*12*16, dtype='u2').reshape(6, 12, 16) + 100
    win.path = write_ser(tmp_path/'frames.ser', frames)
    win._start(inspect_only=True)
    pump(app, lambda: win.job is None)
    plot = win.quality_plot
    assert plot.preview_index == 0 and not plot.frame_image.image.isNull()
    np.testing.assert_array_equal(win.quality_frame_image, frames[0])
    original_input = win.input_image.copy()
    win._set_preprocessing(report())
    app.processEvents()
    levels = win.quality_levels

    def click(rank):
        rect = plot.canvas.plot_rect()
        point = rect.center()
        point.setX(rect.left() + rank / 5 * rect.width())
        QTest.mouseClick(plot.canvas, Qt.MouseButton.LeftButton, pos=point.toPoint())
        pump(app, lambda: win.frame_job is None)

    click(1)
    assert plot.preview_index == 1
    assert 'Frame 2 · Rank 4/6 (top 66.67% of all frames)' in plot.frame_label.text()
    np.testing.assert_array_equal(win.quality_frame_image, frames[1])
    plot.order_control.setCurrentIndex(1)
    plot.absolute_control.setChecked(True)
    plot.log_control.setChecked(True)
    click(1)  # Quality rank 2 is original frame 5, a screened shape outlier.
    assert plot.preview_index == 4
    assert 'Frame 5 · Rank 2/6 (top 33.33% of all frames)' in plot.frame_label.text()
    np.testing.assert_array_equal(win.quality_frame_image, frames[4])
    assert 'width outlier' in plot.frame_label.text()
    click(5)  # Frames without a quality measurement are still inspectable.
    assert plot.preview_index == 5
    np.testing.assert_array_equal(win.quality_frame_image, frames[5])
    assert win.quality_levels == levels
    np.testing.assert_array_equal(win.input_image, original_input)
    assert win.last_result is None and not win.cancel_btn.isEnabled()


def test_latest_click_wins_and_new_capture_clears_preview(gui, tmp_path, monkeypatch):
    from planetrecon.io.ser import write_ser
    from planetrecon.gui import app as module
    app, win = gui
    frames = np.arange(6*12*16, dtype='u2').reshape(6, 12, 16)
    win.path = write_ser(tmp_path/'frames.ser', frames)
    win._start(inspect_only=True)
    pump(app, lambda: win.job is None)
    win._set_preprocessing(report())
    calls = []
    start_job = module.start_stack_job
    def start(*args, **kwargs):
        calls.append(kwargs['preview_frame'])
        return start_job(*args, **kwargs)
    monkeypatch.setattr(module, 'start_stack_job', start)
    for index in (1, 2, 5):
        win._request_quality_frame(index)
    pump(app, lambda: win.frame_job is None)
    assert calls == [1, 5]
    assert win.quality_plot.preview_index == 5
    np.testing.assert_array_equal(win.quality_frame_image, frames[5])
    win._request_quality_frame(3)
    old_job = win.frame_job
    win.path = tmp_path/'different.ser'
    win._capture_ready()
    assert win.frame_job is None and not old_job.process.is_alive()
    assert win.quality_plot.preview_index is None
    assert win.quality_plot.frame_image.image.isNull()
    assert win.quality_frame_image is None and not win.frame_timer.isActive()


@pytest.mark.parametrize('color', ['mono', 'BGR', 'RGGB'])
def test_preview_reads_only_requested_frame_with_correct_colour(color):
    from planetrecon.jobs import input_frame_preview
    from planetrecon.detector import nearest_debayer_preview
    frames = np.arange(3*12*16, dtype='u2').reshape(3, 12, 16)
    if color == 'BGR':
        frames = np.stack((frames, frames+100, frames+200), axis=-1)
    reads = []
    class Source:
        def n_frames(self): return 3
        def color_mode(self): return color
        def read_raw(self, index):
            reads.append(index)
            return frames[index]
    payload = input_frame_preview(Source(), 2)
    expected = (frames[2, ..., ::-1] if color == 'BGR' else
                nearest_debayer_preview(frames[2], color) if color == 'RGGB' else frames[2])
    np.testing.assert_array_equal(payload['input_image'], expected)
    assert reads == [2] and payload['input_frame_index'] == 2
    with pytest.raises(ValueError, match='outside'):
        input_frame_preview(Source(), 3)
    assert reads == [2]
