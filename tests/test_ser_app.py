"""SER tool: cached or fresh frame quality, frame viewing and filtered export."""
import json
import os
import time

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
import pytest

from planetrecon.io.ser import SERSource, write_ser
from planetrecon.pipeline.preprocess_cache import (
    cache_report, default_cache_path, load_cache, preprocess_source, quality_plot_data,
)
from planetrecon.reconstruction import ReconstructionConfig
from test_ser_export import capture

pytest.importorskip('PySide6')


def pump(app, condition, timeout=30):
    deadline = time.monotonic() + timeout
    while not condition() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(.005)
    assert condition(), 'Qt condition timed out'


@pytest.fixture
def tool(tmp_path):
    from planetrecon.gui.ser_app import SERWindow, create_app
    app = create_app(['ser-tool-test'])
    windows = []

    def make(path=None):
        win = SERWindow(path, settings_path=tmp_path/'ser-settings.json')
        win.controls.fields['device'].setCurrentIndex(win.controls.fields['device'].findData('cpu'))
        win.controls.fields['threads'].setValue(2)
        windows.append(win)
        return win

    yield app, make
    for win in windows:
        win._shutdown()
        if win.export_worker is not None:
            win.export_worker.wait(5000)
            app.processEvents()
        win.window.close()
    app.processEvents()


def planet(tmp_path, n=16):
    y, x = np.indices((48, 48))
    disc = 1000*np.exp(-((x-24)**2+(y-24)**2)/90)
    # Fading texture gives every frame a distinct quality score.
    frames = [(disc*(1+.1*(1-i/n)*np.cos(x))).astype('u2') for i in range(n)]
    return write_ser(tmp_path/'input.ser', np.stack(frames))


def select(win, mode):
    chooser = win.controls.fields['selection_mode']
    chooser.setCurrentIndex(chooser.findData(mode))


def test_open_measures_without_geometry_then_reuses_the_cache(tool, tmp_path):
    app, make = tool
    path = planet(tmp_path)
    cache = path.with_name(path.name + '.planetrecon-preprocess.npz')
    win = make()
    assert not win.preprocess_btn.isEnabled() and not win.export_btn.isEnabled()
    assert win.open_capture(path)
    assert win.stage == 'preprocess' and not win.open_btn.isEnabled()
    pump(app, lambda: win.job is None)
    assert not win.error.text()
    assert win.preprocessing_info['status'] == 'ready'
    assert '16 frames' in win.source_label.text()
    with np.load(cache, allow_pickle=False) as data:
        summary = json.loads(str(data['metadata']))['summary']
    assert summary['geometry_estimate']['status'] == 'not_measured'
    assert 'geometry_analysis_config' not in summary
    plot = win.quality_plot
    assert len(plot.selected) == 16 and win.export_btn.isEnabled()
    assert plot.preview_index == 0 and not plot.frame_image.image.isNull()
    assert 'Before registration' not in plot.legend.text()
    plot.frameRequested.emit(9)
    assert plot.preview_index == 9
    assert plot.frame_image.image.width() == 48
    assert 'Frame 10' in plot.frame_label.text()

    stamp = cache.stat().st_mtime_ns
    digest = win.preprocessing_info['digest']
    assert win.open_capture(path)
    assert win.stage == 'inspect'
    pump(app, lambda: win.job is None)
    assert win.preprocessing_info['digest'] == digest
    assert cache.stat().st_mtime_ns == stamp
    assert 'validated' in win.status.text()


def test_selection_modes_set_the_graph_and_the_exported_frames(tool, tmp_path):
    app, make = tool
    path, _, selected = capture(tmp_path)
    win = make()
    win.path = path
    with SERSource(path) as source:
        report = cache_report(selected, default_cache_path(source))
    win._set_preprocessing({**report, 'frame_quality': quality_plot_data(selected)})
    plot = win.quality_plot
    fields = win.controls.fields
    np.testing.assert_array_equal(np.flatnonzero(plot.selected), [0, 3, 5, 6])
    assert win.export_count.text() == 'Export: 4/8 frames'
    assert not fields['minimum_quality'].isEnabled()
    select(win, 'frame_count')
    np.testing.assert_array_equal(np.flatnonzero(plot.selected), [3, 5, 6])
    select(win, 'absolute')
    assert fields['minimum_quality'].isEnabled() and not fields['stack_percent'].isEnabled()
    # Strictly above the score; frame 1 is sharper but fails the size screen.
    fields['minimum_quality'].setText('70')
    np.testing.assert_array_equal(np.flatnonzero(plot.selected), [3, 5])
    assert plot.cutoff == 70 and plot.cutoff_label == 'Quality > 70'
    for text in ('bad', '-1', 'nan', '1000'):
        fields['minimum_quality'].setText(text)
        assert not plot.selected.any() and not win.export_btn.isEnabled()
    assert not win.export_ser(tmp_path/'none.ser')
    fields['minimum_quality'].setText('7e1')
    out = tmp_path/'absolute.ser'
    assert win.export_ser(out)
    assert not win.open_btn.isEnabled() and win.cancel_export_btn.isEnabled()
    pump(app, lambda: win.export_worker is None)
    assert '2/8 frames in capture order' in win.export_status.text()
    with SERSource(path) as source, SERSource(out) as filtered:
        assert [filtered.read_frame_bytes(i) for i in range(2)] == [source.read_frame_bytes(i) for i in [3, 5]]
        np.testing.assert_array_equal(filtered.timestamps(), source.timestamps()[[3, 5]])
    assert win.open_btn.isEnabled() and win.export_btn.isEnabled()


def test_unmatched_cache_is_kept_until_preprocess_is_requested(tool, tmp_path):
    app, make = tool
    path, _, selected = capture(tmp_path)
    cache = path.with_name(path.name + '.planetrecon-preprocess.npz')
    original = cache.read_bytes()
    win = make()
    override = win.controls.fields['bayer_override']
    override.setCurrentIndex(override.findData('RGGB'))
    win.open_capture(path)
    pump(app, lambda: win.job is None)
    assert win.preprocessing_info['status'] == 'stale'
    assert 'replace it' in win.preprocessing_label.text()
    assert cache.read_bytes() == original
    assert not win.export_btn.isEnabled() and win.quality_plot.selection is None
    # Restoring the cached interpretation validates the cache; nothing is measured.
    override.setCurrentIndex(override.findData(None))
    assert win.preprocessing_info['status'] == 'unverified'
    win.preprocess_btn.click()
    assert win.stage == 'inspect'
    pump(app, lambda: win.job is None)
    assert win.preprocessing_info['status'] == 'ready'
    assert win.preprocessing_info['digest'] == selected.digest
    assert win.export_btn.isEnabled()
    override.setCurrentIndex(override.findData('RGGB'))
    assert win.preprocessing_info['status'] == 'unverified' and not win.export_btn.isEnabled()


def test_cancel_and_non_ser_input_leave_the_tool_usable(tool, tmp_path):
    app, make = tool
    win = make()
    win.open_capture(tmp_path/'capture.avi')
    assert win.job is None and 'SER captures only' in win.error.text()
    path = planet(tmp_path)
    win.open_capture(path)
    assert win.cancel_btn.isEnabled()
    win._cancel()
    pump(app, lambda: win.job is None)
    assert win.open_btn.isEnabled() and win.preprocess_btn.isEnabled()
    assert not path.with_name(path.name + '.planetrecon-preprocess.npz').exists()
    win.preprocess_btn.click()
    pump(app, lambda: win.job is None)
    assert win.preprocessing_info['status'] == 'ready'


def test_preferences_are_separate_and_restored(tool, tmp_path, monkeypatch):
    from planetrecon.gui import settings
    app, make = tool
    monkeypatch.setenv('PLANETRECON_SETTINGS_PATH', str(tmp_path/'main.json'))
    monkeypatch.setenv('PLANETRECON_SER_SETTINGS_PATH', str(tmp_path/'ser.json'))
    assert settings.default_path() == tmp_path/'main.json'
    assert settings.default_path('ser-settings.json', 'PLANETRECON_SER_SETTINGS_PATH') == tmp_path/'ser.json'
    win = make()
    select(win, 'absolute')
    win.controls.fields['minimum_quality'].setText('1.5e-3')
    win.controls.fields['stack_percent'].setValue(35)
    win.controls.fields['reject_saturated'].setChecked(False)
    win.quality_plot.log_control.setChecked(True)
    win._save_settings()
    restored = make()
    assert restored.controls.selection_mode() == 'absolute'
    assert restored.controls.minimum_quality() == '1.5e-3'
    cfg = restored.controls.configuration()
    assert (cfg.stack_percent, cfg.reject_saturated, cfg.frame_selection_mode) == (35, False, 'quality_range')
    assert cfg.frame_preselection and cfg.geometry_mode == 'none'
    assert restored.quality_plot.log_control.isChecked()
    restored.controls.fields['gain_e_per_adu'].setText('fast')
    with pytest.raises(ValueError, match='Gain'):
        restored.controls.configuration()


def test_geometry_free_cache_is_valid_but_marks_geometry_for_refresh(tmp_path):
    from planetrecon.jobs import start_stack_job
    path = planet(tmp_path)
    cfg = ReconstructionConfig(device='cpu', threads=2)
    with SERSource(path) as source:
        selected = preprocess_source(source, cfg, measure_geometry=False)
        loaded, report = load_cache(source, cfg)
    assert report['status'] == 'ready' and loaded.digest == selected.digest
    assert report['geometry_estimate']['applicable'] is False
    assert report['geometry_estimate']['suggestions'] == {}
    with pytest.raises(ValueError, match='geometry can only be skipped'):
        start_stack_job(path, cfg, inspect_only=True, measure_geometry=False)


def test_cli_starts_the_ser_tool(monkeypatch, tmp_path):
    from planetrecon import cli
    from planetrecon.gui import ser_app
    calls = []
    monkeypatch.setattr(ser_app, 'main', lambda argv: calls.append(argv) or 0)
    assert cli.main(['ser']) == 0
    assert cli.main(['ser', str(tmp_path/'a.ser')]) == 0
    assert cli.main(['ser', '--path', str(tmp_path/'b.ser')]) == 0
    assert calls == [[], [str(tmp_path/'a.ser')], [str(tmp_path/'b.ser')]]
