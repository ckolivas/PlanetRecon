"""The plot must describe the actual next-run mask, including cache lifecycle."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np

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
    assert 'unavailable' in plot.frame_description(5)
    plot.order_control.setCurrentIndex(1)
    assert plot.order.tolist() == [3, 4, 2, 1, 0, 5]
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
    fields['frame_selection_mode'].setCurrentIndex(1)
    assert plot.selected.tolist() == [True, True, False, False]
    assert plot.cutoff is None  # A horizontal line cannot represent tied count selection.
    win._set_preprocessing(report((7,), (True,)))
    app.processEvents()
    assert plot.selected.tolist() == [True]
    assert not plot.canvas.grab().isNull()
    win._set_preprocessing(report((np.nan,), (False,)))
    assert not plot.selected.any() and plot.cutoff is None
    assert 'unavailable' in plot.frame_description(0)
    assert not plot.canvas.grab().isNull()


def test_worker_preprocess_and_inspect_deliver_plot_without_export_bloat(gui, tmp_path):
    from test_preprocess_cache import capture
    app, win = gui
    win.path = capture(tmp_path)
    win.controls.fields['frame_preselection'].setChecked(True)
    win._preprocess()
    pump(app, lambda: win.job is None, timeout=20)
    assert not win.error.text()
    plot = win.quality_plot
    assert len(plot.selected) == 34
    assert np.flatnonzero(~plot.selection.accepted).tolist() == [32, 33]
    original = plot.selection.measurements.copy()
    win.preprocessing_info = {}
    win._refresh_preprocessing()
    win._inspect()
    pump(app, lambda: win.job is None, timeout=20)
    np.testing.assert_array_equal(plot.selection.measurements, original)
    win._run()
    pump(app, lambda: win.job is None, timeout=20)
    assert not win.error.text()
    assert len(plot.selected) == 34
    assert 'frame_quality' not in win.last_result.provenance['preprocessing_cache']
