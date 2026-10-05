"""Filtered SER exports preserve raw observations, timing and capture order."""
from dataclasses import replace
import os
from pathlib import Path
import struct
import threading
import time

import numpy as np
import pytest

from planetrecon.export import ExportCancelled
from planetrecon.io.ser import SERSource, write_ser, SER_HEADER_SIZE
from planetrecon.pipeline.preprocess import FrameSelection
from planetrecon.pipeline.preprocess_cache import (
    CacheValidation, cache_report, default_cache_path, identity, load_cache,
    quality_plot_data, save_cache, selection_digest,
)
from planetrecon.reconstruction import ReconstructionConfig
from planetrecon.ser_export import export_filtered_ser, export_frame_mask


def capture(tmp_path, *, color=0, depth=8, endian=0, timestamps=True, convention='ecosystem'):
    cfg = ReconstructionConfig(device='cpu', threads=2, frame_preselection=True,
                               stack_percent=50, frame_selection_mode='frame_count',
                               endian_convention=convention)
    shape = (8, 6, 10, 3) if color in (100, 101) else (8, 6, 10)
    frames = (np.arange(np.prod(shape)).reshape(shape) % (2**depth)).astype('u1' if depth == 8 else 'u2')
    times = 638631501000000000 + np.array([0, 10, 22, 36, 49, 62, 74, 88], dtype='i8') * 10000
    path = write_ser(tmp_path/'input.ser', frames, color_id=color, pixel_depth=depth,
                     little_endian_flag=endian, endian_convention=convention,
                     timestamps=times if timestamps else None, observer='Observer',
                     instrument='Camera', telescope='Scope', datetime=111, datetime_utc=222)
    # Preserve header bytes even when they cannot be decoded as UTF-8.
    with path.open('r+b') as stream:
        stream.seek(42)
        stream.write(b'\xff\x80')
    accepted = np.ones(8, dtype=bool)
    accepted[[1, 4]] = False
    measures = np.column_stack(([60, 100, 10, 80, 20, 90, 70, 50],
                                np.full(8, 5), np.full(8, 4), np.zeros(8))).astype('f8')
    reasons = {key: [] for key in ('invalid', 'saturated', 'no_target', 'clipped_target',
                                   'low_quality', 'width_outlier', 'height_outlier')}
    reasons.update(width_outlier=[1], height_outlier=[4])
    selected = FrameSelection(accepted, measures,
        {'complete': True, 'n_measured': 8, 'rejected_indices_by_reason': reasons})
    with SERSource(path, endian_convention=convention) as source:
        selected.identity = identity(source, cfg, None)
        selected.digest = selection_digest(selected)
        save_cache(default_cache_path(source), selected, source, cfg)
    return path, cfg, selected


@pytest.mark.parametrize('color', [0, 8, 9, 10, 11, 100, 101])
@pytest.mark.parametrize('depth,endian,timestamps', [(8, 0, True), (12, 1, True), (16, 0, False)])
def test_lossless_format_order_metadata_and_timestamps(tmp_path, color, depth, endian, timestamps):
    path, cfg, selection = capture(tmp_path, color=color, depth=depth, endian=endian, timestamps=timestamps)
    out = tmp_path/'filtered.ser'
    report = export_filtered_ser(path, out, cfg, expected_digest=selection.digest)
    assert (report.n_input, report.n_written, report.has_timestamps) == (8, 3, timestamps)
    kept = [3, 5, 6]  # Quality rank is 5,3,6; frame 1 fails size despite highest quality.
    with SERSource(path) as source, SERSource(out) as filtered:
        assert filtered.n_frames() == 3
        assert filtered.frame_shape() == source.frame_shape()
        assert filtered.color_mode() == source.color_mode()
        for j, i in enumerate(kept):
            assert filtered.read_frame_bytes(j) == source.read_frame_bytes(i)
            np.testing.assert_array_equal(filtered.read_raw(j), source.read_raw(i))
        if timestamps:
            np.testing.assert_array_equal(filtered.timestamps(), source.timestamps()[kept])
        else:
            assert filtered.timestamps() is None
        assert out.stat().st_size == SER_HEADER_SIZE + 3*len(source.read_frame_bytes(0)) + (24 if timestamps else 0)
    before = bytearray(path.read_bytes()[:SER_HEADER_SIZE])
    struct.pack_into('<i', before, 38, 3)
    assert out.read_bytes()[:SER_HEADER_SIZE] == before


@pytest.mark.parametrize('mode,percent,kept', [('quality_range', 50, [0, 3, 5, 6]),
                                            ('frame_count', 100, [0, 2, 3, 5, 6, 7])])
def test_graph_selection_rules(tmp_path, mode, percent, kept):
    path, cfg, _ = capture(tmp_path)
    out = tmp_path/'filtered.ser'
    export_filtered_ser(path, out, replace(cfg, frame_selection_mode=mode, stack_percent=percent))
    with SERSource(path) as source, SERSource(out) as filtered:
        assert filtered.n_frames() == len(kept)
        assert [filtered.read_frame_bytes(j) for j in range(len(kept))] == [source.read_frame_bytes(i) for i in kept]


def test_absolute_export_quality_overrides_stack_percentage_and_preserves_size_mask(tmp_path):
    path, cfg, _ = capture(tmp_path)
    out = tmp_path/'absolute.ser'
    # The top 1% quality range has no accepted frames (best score fails size).
    # An absolute cutoff of 70 keeps scores 80 and 90, not the equal score 70.
    report = export_filtered_ser(path, out,
        replace(cfg, frame_selection_mode='quality_range', stack_percent=1), minimum_quality=70)
    assert report.n_written == 2
    with SERSource(path) as source, SERSource(out) as filtered:
        assert [filtered.read_frame_bytes(i) for i in range(2)] == [source.read_frame_bytes(i) for i in [3, 5]]
        np.testing.assert_array_equal(filtered.timestamps(), source.timestamps()[[3, 5]])


@pytest.mark.parametrize('threshold', [-1, float('nan'), float('inf'), '', 'bad', True])
def test_invalid_absolute_export_quality(tmp_path, threshold):
    path, cfg, _ = capture(tmp_path)
    out = tmp_path/'invalid.ser'
    with pytest.raises(ValueError, match='finite, non-negative'):
        export_filtered_ser(path, out, cfg, minimum_quality=threshold)
    assert not out.exists()


def test_absolute_quality_is_not_normalized_and_excludes_nonfinite_scores(tmp_path):
    _, _, selected = capture(tmp_path)
    selected.measurements[:, 0] = 2e-8
    selected.measurements[0, 0] = np.nan
    mask = export_frame_mask(selected, 1, 'frame_count', '1e-8')
    np.testing.assert_array_equal(np.flatnonzero(mask), [2, 3, 5, 6, 7])
    assert not export_frame_mask(selected, 100, 'quality_range', 2e-8).any()


def test_spec_endian_is_encoded_for_other_ser_readers(tmp_path):
    path, cfg, _ = capture(tmp_path, depth=16, endian=1, convention='spec')
    out = tmp_path/'filtered.ser'
    export_filtered_ser(path, out, cfg)
    assert struct.unpack_from('<i', out.read_bytes(), 22)[0] == 0
    with SERSource(path, endian_convention='spec') as source, SERSource(out) as filtered:
        np.testing.assert_array_equal(filtered.read_raw(0), source.read_raw(3))


def test_bayer_override_recorded_without_debayering(tmp_path):
    path, cfg, selected = capture(tmp_path)
    cfg = replace(cfg, bayer_override='RGGB')
    with SERSource(path, bayer_override='RGGB') as source:
        selected.identity = identity(source, cfg, None)
        selected.digest = selection_digest(selected)
        save_cache(default_cache_path(source), selected, source, cfg)
    out = tmp_path/'filtered.ser'
    export_filtered_ser(path, out, cfg)
    with SERSource(path) as source, SERSource(out) as filtered:
        assert filtered.color_mode() == 'RGGB'
        assert filtered.read_frame_bytes(0) == source.read_frame_bytes(3)


def test_reuses_validation_receipt_without_second_capture_hash(tmp_path, monkeypatch):
    path, cfg, _ = capture(tmp_path)
    validation = CacheValidation()
    with SERSource(path) as source:
        assert load_cache(source, cfg, validation=validation)[0] is not None
    from planetrecon.pipeline import preprocess_cache
    monkeypatch.setattr(preprocess_cache, 'identity', lambda *a, **kw: pytest.fail('rehashed unchanged capture'))
    export_filtered_ser(path, tmp_path/'filtered.ser', cfg, validation=validation)


@pytest.mark.parametrize('alias', ['same', 'symlink', 'hardlink'])
def test_never_overwrites_input(tmp_path, alias):
    path, cfg, _ = capture(tmp_path)
    out = path if alias == 'same' else tmp_path/'alias.ser'
    if alias == 'symlink':
        out.symlink_to(path)
    elif alias == 'hardlink':
        os.link(path, out)
    before = path.read_bytes()
    with pytest.raises(ValueError, match='input capture'):
        export_filtered_ser(path, out, cfg, overwrite=True)
    assert path.read_bytes() == before


def test_atomic_overwrite_and_cancel_cleanup(tmp_path):
    path, cfg, _ = capture(tmp_path)
    out = tmp_path/'filtered.ser'
    out.write_bytes(b'existing file')
    with pytest.raises(FileExistsError):
        export_filtered_ser(path, out, cfg)
    cancel = threading.Event()
    def progress(stage, done, total):
        if stage == 'Exporting SER' and done == 1:
            assert out.read_bytes() == b'existing file'
            cancel.set()
    with pytest.raises(ExportCancelled):
        export_filtered_ser(path, out, cfg, overwrite=True, should_cancel=cancel.is_set, on_progress=progress)
    assert out.read_bytes() == b'existing file'
    assert not list(tmp_path.glob('.*.tmp'))
    export_filtered_ser(path, out, cfg, overwrite=True)
    with SERSource(out) as source:
        assert source.n_frames() == 3


def test_refuses_stale_cache_or_graph_and_empty_selection(tmp_path):
    path, cfg, selected = capture(tmp_path)
    out = tmp_path/'filtered.ser'
    with pytest.raises(ValueError, match='Preprocessing changed'):
        export_filtered_ser(path, out, cfg, expected_digest='wrong')
    with SERSource(path) as source:
        selected.accepted[:] = False
        selected.digest = selection_digest(selected)
        save_cache(default_cache_path(source), selected, source, cfg)
    with pytest.raises(ValueError, match='No frames remain'):
        export_filtered_ser(path, out, cfg)
    with path.open('r+b') as stream:
        stream.seek(SER_HEADER_SIZE)
        stream.write(b'\xff')
    with pytest.raises(ValueError, match='Preprocess again'):
        export_filtered_ser(path, out, cfg)
    assert not out.exists()


def test_source_change_during_copy_prevents_publication(tmp_path):
    path, cfg, _ = capture(tmp_path)
    out = tmp_path/'filtered.ser'
    def progress(stage, done, total):
        if stage == 'Exporting SER' and done == 1:
            with path.open('r+b') as stream:
                stream.seek(SER_HEADER_SIZE)
                stream.write(b'\xff')
    with pytest.raises(OSError, match='changed'):
        export_filtered_ser(path, out, cfg, on_progress=progress)
    assert not out.exists()
    assert not list(tmp_path.glob('.*.tmp'))


def test_concurrent_destination_is_not_replaced(tmp_path):
    path, cfg, _ = capture(tmp_path)
    out = tmp_path/'filtered.ser'
    def progress(stage, done, total):
        if stage == 'Exporting SER' and done == total:
            out.write_bytes(b'other writer')
    with pytest.raises(FileExistsError):
        export_filtered_ser(path, out, cfg, on_progress=progress)
    assert out.read_bytes() == b'other writer'
    assert not list(tmp_path.glob('.*.tmp'))


def test_missing_or_disabled_preprocessing_cannot_export(tmp_path):
    path, cfg, _ = capture(tmp_path)
    out = tmp_path/'filtered.ser'
    with pytest.raises(ValueError, match='Enable cached preprocessing'):
        export_filtered_ser(path, out, replace(cfg, frame_preselection=False))
    with SERSource(path) as source:
        default_cache_path(source).unlink()
    with pytest.raises(ValueError, match='run Preprocess'):
        export_filtered_ser(path, out, cfg)
    assert not out.exists()


def pump(app, condition):
    deadline = time.monotonic() + 10
    while not condition() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(.005)
    assert condition(), 'Qt condition timed out'


def test_gui_exports_graph_selection_without_stacking(tmp_path):
    pytest.importorskip('PySide6')
    from planetrecon.gui.app import MainWindow, create_app
    path, cfg, selected = capture(tmp_path)
    app = create_app(['ser-export-test'])
    win = MainWindow(config=cfg, settings_path=tmp_path/'settings.json')
    try:
        assert not win.export_ser_btn.isEnabled()
        assert not win.cancel_ser_btn.isEnabled()
        win.path = path
        with SERSource(path) as source:
            report = cache_report(selected, default_cache_path(source))
        win._set_preprocessing({**report, 'frame_quality': quality_plot_data(selected)})
        assert win.export_ser_btn.isEnabled()
        assert win.last_result is None
        win.controls.fields['frame_preselection'].setChecked(False)
        assert not win.export_ser_btn.isEnabled()
        win.controls.fields['frame_preselection'].setChecked(True)
        win.controls.fields['frame_selection_mode'].setCurrentIndex(
            win.controls.fields['frame_selection_mode'].findData('quality_range'))
        np.testing.assert_array_equal(np.flatnonzero(win.quality_plot.selected), [0, 3, 5, 6])
        out = tmp_path/'gui.ser'
        assert win.save_filtered_ser(out)
        assert not win.open_btn.isEnabled()
        assert win.cancel_ser_btn.isEnabled()
        assert not win.export_ser_btn.isEnabled()
        pump(app, lambda: win.export_worker is None)
        assert '4/8 frames in capture order' in win.ser_export_status.text()
        with SERSource(path) as source, SERSource(out) as filtered:
            assert [filtered.read_frame_bytes(j) for j in range(4)] == [source.read_frame_bytes(i) for i in [0, 3, 5, 6]]
        assert win.open_btn.isEnabled() and win.export_ser_btn.isEnabled()
        assert not win.cancel_ser_btn.isEnabled()
        win._preprocessing_settings_changed()
        assert not win.export_ser_btn.isEnabled()
    finally:
        win._shutdown()
        if win.export_worker is not None:
            win.export_worker.wait(5000)
            app.processEvents()
        win.window.close()
        app.processEvents()


def test_gui_export_cancel_and_dialog(tmp_path, monkeypatch):
    pytest.importorskip('PySide6')
    from planetrecon.gui import app as module
    path, cfg, selected = capture(tmp_path)
    app = module.create_app(['ser-export-test'])
    win = module.MainWindow(config=cfg, settings_path=tmp_path/'settings.json')
    entered = threading.Event()
    observed = {}
    def slow_export(source, destination, config, **kwargs):
        observed.update(source=source, destination=destination, config=config)
        entered.set()
        deadline = time.monotonic() + 5
        while not kwargs['should_cancel']() and time.monotonic() < deadline:
            time.sleep(.005)
        raise ExportCancelled()
    def dialog(*args, **kwargs):
        assert args[1] == 'Export filtered SER'
        assert Path(args[2]).name == 'input_filtered.ser'
        return str(tmp_path/'exported'), 'SER capture (*.ser)'
    monkeypatch.setattr(module, 'export_filtered_ser', slow_export)
    monkeypatch.setattr(module.QFileDialog, 'getSaveFileName', dialog)
    try:
        win.path = path
        win._set_preprocessing({**cache_report(selected), 'frame_quality': quality_plot_data(selected)})
        win.export_ser_btn.click()
        pump(app, entered.is_set)
        assert observed['source'] == path
        assert observed['destination'] == tmp_path/'exported.ser'
        assert observed['config'].stack_percent == cfg.stack_percent
        assert Path(win._dialog_directory()) == tmp_path
        win.cancel_ser_btn.click()
        pump(app, lambda: win.export_worker is None)
        assert win.ser_export_status.text() == 'SER export cancelled.'
        assert win.export_ser_btn.isEnabled()
        assert not (tmp_path/'exported.ser').exists()
    finally:
        win._shutdown()
        if win.export_worker is not None:
            win.export_worker.wait(5000)
            app.processEvents()
        win.window.close()
        app.processEvents()


def test_gui_absolute_export_quality_count_and_saved_preference(tmp_path):
    pytest.importorskip('PySide6')
    from planetrecon.gui.app import MainWindow, create_app
    path, cfg, selected = capture(tmp_path)
    app = create_app(['ser-quality-test'])
    settings_path = tmp_path/'settings.json'
    win = MainWindow(config=cfg, settings_path=settings_path)
    restored = None
    try:
        win.path = path
        with SERSource(path) as source:
            report = cache_report(selected, default_cache_path(source))
        win._set_preprocessing({**report, 'frame_quality': quality_plot_data(selected)})
        assert not win.ser_quality_check.isChecked()
        assert not win.ser_min_quality.isEnabled()
        win.controls.fields['frame_selection_mode'].setCurrentIndex(
            win.controls.fields['frame_selection_mode'].findData('quality_range'))
        win.controls.fields['stack_percent'].setValue(1)
        assert not win.quality_plot.selected.any() and not win.export_ser_btn.isEnabled()
        win.ser_quality_check.setChecked(True)
        for value in ('bad', 'nan', '-1', '1000'):
            win.ser_min_quality.setText(value)
            assert not win.export_ser_btn.isEnabled()
        win.ser_min_quality.setText('7e1')
        assert win.ser_export_count.text() == 'Export: 2/8 frames'
        assert win.export_ser_btn.isEnabled()
        assert not win.quality_plot.selected.any()  # Stacking selection is unchanged.
        out = tmp_path/'gui-quality.ser'
        assert win.save_filtered_ser(out)
        assert not win.ser_quality_check.isEnabled() and not win.ser_min_quality.isEnabled()
        pump(app, lambda: win.export_worker is None)
        assert '2/8 frames in capture order' in win.ser_export_status.text()
        with SERSource(path) as source, SERSource(out) as filtered:
            assert [filtered.read_frame_bytes(i) for i in range(2)] == [source.read_frame_bytes(i) for i in [3, 5]]
        win._save_settings()
        restored = MainWindow(config=cfg, settings_path=settings_path)
        assert restored.ser_quality_check.isChecked() and restored.ser_min_quality.text() == '7e1'
        assert restored.path is None and restored.job is None
    finally:
        for window in (win, restored):
            if window is not None:
                window._shutdown()
                if window.export_worker is not None:
                    window.export_worker.wait(5000)
                    app.processEvents()
                window.window.close()
        app.processEvents()
