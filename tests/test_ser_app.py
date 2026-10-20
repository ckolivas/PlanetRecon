"""SER tool: cached or fresh frame quality, frame viewing and filtered export."""
import json
import os
import time

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
import pytest

from planetrecon.io.ser import SERSource, write_ser
from planetrecon.pipeline.preprocess import best_frame_mask
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


def planet(tmp_path, n=16, name='input.ser'):
    y, x = np.indices((48, 48))
    disc = 1000*np.exp(-((x-24)**2+(y-24)**2)/90)
    # Fading texture gives every frame a distinct quality score.
    frames = [(disc*(1+.1*(1-i/n)*np.cos(x))).astype('u2') for i in range(n)]
    return write_ser(tmp_path/name, np.stack(frames))


def cache_of(path):
    return path.with_name(path.name + '.planetrecon-preprocess.npz')


def results(win):
    rows = [win.batch_list.topLevelItem(i) for i in range(win.batch_list.topLevelItemCount())]
    return [row.text(1) for row in rows]


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


def test_open_displays_a_cache_and_export_validates_it(tool, tmp_path):
    app, make = tool
    path = planet(tmp_path)
    with SERSource(path) as source:
        preprocess_source(source, make().controls.configuration(), measure_geometry=False)
    stamp = path.stat()
    os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns + 1_000_000))  # Breaks the persisted proof.
    win = make()
    stages = []
    progress = win._progress
    win._progress = lambda payload: (stages.append(payload.get('stage', '')), progress(payload))[1]
    assert win.open_capture(path)
    pump(app, lambda: win.job is None)
    assert not win.error.text()
    assert not any('Validating' in stage for stage in stages)
    assert win.preprocessing_info['status'] == 'ready' and win.preprocessing_info['verified'] is False
    assert 'validated when exported' in win.status.text()
    assert win.export_btn.isEnabled() and win.quality_plot.selected.any()
    out = tmp_path/'checked.ser'
    assert win.export_ser(out)
    pump(app, lambda: win.export_worker is None)
    assert out.is_file(), win.export_status.text()


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


def test_batch_preprocess_reuses_matching_caches_and_measures_the_rest(tool, tmp_path):
    app, make = tool
    win = make()
    cached, fresh = planet(tmp_path, name='cached.ser'), planet(tmp_path, name='fresh.ser')
    with SERSource(cached) as source:
        preprocess_source(source, win.controls.configuration(), measure_geometry=False)
    stamp = cache_of(cached).stat().st_mtime_ns
    assert not win.batch_box.isVisibleTo(win.window)
    assert win.start_batch([cached, fresh, tmp_path/'notes.avi', fresh])
    assert not win.start_batch([fresh])
    assert win.batch_box.isVisibleTo(win.window) and win.batch_list.topLevelItemCount() == 3
    assert win.cancel_btn.isEnabled() and not win.controls.isEnabled()
    assert not any(button.isEnabled() for button in (
        win.open_btn, win.preprocess_btn, win.batch_preprocess_btn, win.batch_export_btn, win.export_btn))
    pump(app, lambda: win.batch is None)
    assert results(win)[:2] == ['Cache reused: 14/16 frames retained', 'Measured: 14/16 frames retained']
    assert 'SER captures only' in results(win)[2] and not win.error.text()
    assert win.batch_box.title() == 'Batch preprocess finished: 1 measured, 1 cached, 1 failed'
    assert win.batch_box.title() in win.status.text()
    assert cache_of(cached).stat().st_mtime_ns == stamp and cache_of(fresh).is_file()
    assert win.open_btn.isEnabled() and win.batch_export_btn.isEnabled() and win.controls.isEnabled()
    assert not win.cancel_btn.isEnabled()
    win.batch_list.itemActivated.emit(win.batch_list.topLevelItem(0), 0)
    assert win.path == cached and win.stage == 'inspect'
    pump(app, lambda: win.job is None)
    assert win.preprocessing_info['status'] == 'ready' and win.export_btn.isEnabled()


def test_batch_export_preprocesses_when_needed_and_follows_the_selection(tool, tmp_path):
    app, make = tool
    (tmp_path/'scored').mkdir()
    scored, _, _ = capture(tmp_path/'scored')
    fresh = planet(tmp_path, name='fresh.ser')
    out = tmp_path/'out'
    win = make()
    select(win, 'frame_count')
    assert win.start_batch([scored, fresh], export_directory=out)
    pump(app, lambda: win.batch is None)
    assert win.batch_box.title() == 'Batch export finished: 2 exported'
    first, second = out/'input_filtered.ser', out/'fresh_filtered.ser'
    assert results(win)[0] == 'Saved input_filtered.ser · 3/8 frames'
    assert str(first) in win.batch_list.topLevelItem(0).toolTip(1)
    with SERSource(fresh) as source:
        measured, _ = load_cache(source, win.controls.configuration())
    best = np.flatnonzero(best_frame_mask(measured, 50, 'frame_count'))
    assert len(best) == 7
    assert results(win)[1] == 'Saved fresh_filtered.ser · 7/16 frames · frame quality measured first'
    with SERSource(scored) as source, SERSource(first) as filtered:
        assert [filtered.read_frame_bytes(i) for i in range(3)] == [source.read_frame_bytes(i) for i in [3, 5, 6]]
        np.testing.assert_array_equal(filtered.timestamps(), source.timestamps()[[3, 5, 6]])
    with SERSource(fresh) as source, SERSource(second) as filtered:
        assert [filtered.read_frame_bytes(i) for i in range(7)] == [source.read_frame_bytes(i) for i in best]
    with np.load(cache_of(fresh), allow_pickle=False) as data:
        assert json.loads(str(data['metadata']))['summary']['geometry_estimate']['status'] == 'not_measured'

    # Existing outputs are kept unless replacement is requested.
    stamps = [path.stat().st_mtime_ns for path in (first, second)]
    select(win, 'absolute')
    win.controls.fields['minimum_quality'].setText('1790')
    assert win.start_batch([scored, fresh], export_directory=out)
    pump(app, lambda: win.batch is None)
    assert win.batch_box.title() == 'Batch export finished: 2 skipped'
    assert [path.stat().st_mtime_ns for path in (first, second)] == stamps
    assert win.start_batch([scored, fresh], export_directory=out, existing='replace')
    pump(app, lambda: win.batch is None)
    # A capture with nothing selected does not stop the batch or lose its old export.
    assert win.batch_box.title() == 'Batch export finished: 1 exported, 1 failed'
    assert 'No frames' in results(win)[0] and first.stat().st_mtime_ns == stamps[0]
    with SERSource(fresh) as source, SERSource(second) as filtered:
        assert filtered.n_frames() == 1 and filtered.read_frame_bytes(0) == source.read_frame_bytes(1)


def test_batch_replaces_an_unmatched_cache_and_cancel_stops_the_queue(tool, tmp_path, monkeypatch):
    app, make = tool
    (tmp_path/'scored').mkdir()
    scored, _, selected = capture(tmp_path/'scored')
    original = cache_of(scored).read_bytes()
    win = make()
    override = win.controls.fields['bayer_override']
    override.setCurrentIndex(override.findData('RGGB'))
    assert win.start_batch([scored])
    pump(app, lambda: win.batch is None)
    assert results(win)[0].startswith('Measured: ')
    assert cache_of(scored).read_bytes() != original
    assert win.preprocessing_info['status'] == 'ready' and win.preprocessing_info['digest'] != selected.digest

    first, second = planet(tmp_path, name='a.ser'), planet(tmp_path, name='b.ser')
    assert win.start_batch([first, second], export_directory=tmp_path/'out')
    win.cancel_btn.click()
    assert not win.cancel_btn.isEnabled()
    pump(app, lambda: win.batch is None)
    assert win.batch_box.title() == 'Batch export cancelled: 2 cancelled'
    assert results(win) == ['Cancelled', 'Cancelled']
    assert not cache_of(first).exists() and not (tmp_path/'out').exists()
    assert win.open_btn.isEnabled() and win.batch_preprocess_btn.isEnabled()

    # Either cancel button stops the whole batch while a capture is being written.
    from planetrecon.export import ExportCancelled
    from planetrecon.gui import app as main_app

    def blocked(*args, should_cancel, **kwargs):
        while not should_cancel():
            time.sleep(.005)
        raise ExportCancelled('SER export cancelled')

    monkeypatch.setattr(main_app, 'export_filtered_ser', blocked)
    for button in (win.cancel_btn, win.cancel_export_btn):
        assert win.start_batch([first, second], export_directory=tmp_path/'out')
        pump(app, lambda: win.export_worker is not None)
        assert button.isEnabled()
        button.click()
        pump(app, lambda: win.batch is None)
        assert results(win) == ['SER export cancelled.', 'Cancelled']
        assert win.batch_box.title() == 'Batch export cancelled: 2 cancelled'
    assert cache_of(first).is_file() and not cache_of(second).exists()
    # Closing the window abandons the queue with the running step.
    assert win.start_batch([first, second])
    win.window.close()
    pump(app, lambda: win.batch is None)
    assert results(win) == ['Cancelled', 'Cancelled'] and win.job is None


def test_batch_export_is_refused_when_it_cannot_be_unambiguous(tool, tmp_path):
    app, make = tool
    win = make()
    (tmp_path/'other').mkdir()
    first, twin = planet(tmp_path, name='a.ser'), planet(tmp_path/'other', name='a.ser')
    output = planet(tmp_path, name='a_filtered.ser')
    assert not win.start_batch([])
    assert not win.start_batch([first, twin], export_directory=tmp_path/'out')
    assert 'same file' in win.error.text()
    assert not win.start_batch([first, output], export_directory=tmp_path)
    assert 'replace one of the selected captures' in win.error.text()
    select(win, 'absolute')
    win.controls.fields['minimum_quality'].setText('sharp')
    assert not win.start_batch([first], export_directory=tmp_path/'out')
    assert 'Batch export not started' in win.error.text()
    assert win.batch is None and win.open_btn.isEnabled() and not cache_of(first).exists()
    # Preprocessing does not depend on the export selection.
    assert win.start_batch([first])
    pump(app, lambda: win.batch is None)
    assert results(win) == ['Measured: 14/16 frames retained']
    assert win.start_batch([tmp_path/'absent.ser'])
    assert win.batch is None and results(win) == [f"Capture not found: {tmp_path/'absent.ser'}"]


def test_batch_dialogs_choose_captures_folder_and_existing_outputs(tool, tmp_path, monkeypatch):
    from planetrecon.gui import ser_app
    app, make = tool
    win = make()
    first, second = planet(tmp_path, name='a.ser'), planet(tmp_path, name='b.ser')
    out = tmp_path/'out'
    out.mkdir()
    (out/'a_filtered.ser').write_bytes(b'kept')
    monkeypatch.setattr(ser_app.QFileDialog, 'getOpenFileNames',
                        lambda *args, **kwargs: ([str(first), str(second)], ''))
    monkeypatch.setattr(ser_app.QFileDialog, 'getExistingDirectory', lambda *args, **kwargs: str(out))
    answers = []

    def answer(box):
        wanted = answers.pop(0)
        button = next(b for b in box.buttons() if b.text() == wanted)
        monkeypatch.setattr(box, 'clickedButton', lambda: button)

    monkeypatch.setattr(ser_app.QMessageBox, 'exec', answer)
    answers.append('Cancel')
    win.batch_export_btn.click()
    assert win.batch is None and not win.batch_box.isVisibleTo(win.window)
    answers.append('Skip existing')
    win.batch_export_btn.click()
    pump(app, lambda: win.batch is None)
    assert win.batch_box.title() == 'Batch export finished: 1 exported, 1 skipped'
    assert (out/'a_filtered.ser').read_bytes() == b'kept' and (out/'b_filtered.ser').is_file()
    answers.append('Replace')
    win.batch_export_btn.click()
    pump(app, lambda: win.batch is None)
    assert win.batch_box.title() == 'Batch export finished: 2 exported'
    assert (out/'a_filtered.ser').read_bytes() != b'kept'
    monkeypatch.setattr(ser_app.QFileDialog, 'getExistingDirectory', lambda *args, **kwargs: '')
    win.batch_preprocess_btn.click()
    pump(app, lambda: win.batch is None)
    assert win.batch_box.title() == 'Batch preprocess finished: 2 cached'


def test_cli_starts_the_ser_tool(monkeypatch, tmp_path):
    from planetrecon import cli
    from planetrecon.gui import ser_app
    calls = []
    monkeypatch.setattr(ser_app, 'main', lambda argv: calls.append(argv) or 0)
    assert cli.main(['ser']) == 0
    assert cli.main(['ser', str(tmp_path/'a.ser')]) == 0
    assert cli.main(['ser', '--path', str(tmp_path/'b.ser')]) == 0
    assert calls == [[], [str(tmp_path/'a.ser')], [str(tmp_path/'b.ser')]]
