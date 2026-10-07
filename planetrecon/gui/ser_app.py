"""SER-only desktop tool: frame quality, frame inspection and lossless filtered export.

Shares the capture reader, preprocessing cache, quality graph and SER export
with the reconstruction application, without any stacking or geometry.
"""
from __future__ import annotations

from pathlib import Path
import sys
import time

import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QMainWindow, QMessageBox, QProgressBar, QPushButton, QScrollArea, QSpinBox, QSplitter,
    QVBoxLayout, QWidget,
)

from planetrecon.export import ExportCancelled
from planetrecon.gui.app import SERExportWorker, _to_qimage, create_app
from planetrecon.gui.help import CONTROL_HELP
from planetrecon.gui.quality import QualityPlot
from planetrecon.jobs import JobHandle, input_frame_preview, start_stack_job
from planetrecon.reconstruction import ReconstructionConfig
from planetrecon.runtime import apply_thread_limits, detected_thread_count

# Settings that change the measured samples; the rest only select or schedule.
INPUT_KEYS = ('bayer_override', 'endian_override', 'endian_convention', 'recover_complete_frames',
              'reject_saturated', 'bias_path', 'dark_path', 'flat_path', 'gain_e_per_adu',
              'read_noise_e', 'saturate_adu')
SELECTION_KEYS = ('selection_mode', 'stack_percent', 'minimum_quality')
FRAME_PREVIEW_LIMIT = 1024

HELP = {
    **CONTROL_HELP,
    'selection_mode': 'Frames shown green and written by Export SER. Quality range keeps scores above a cutoff between the capture worst and best. Frame count keeps a ranked percentage of screened frames. Absolute quality keeps scores strictly above a fixed estimator score. Shape and validity screening applies in every mode.',
    'stack_percent': 'Quality range: 50 keeps scores strictly above (best + worst) / 2. Frame count: keeps this percentage of screened frames ranked by quality. 100 keeps every screened frame. Changing it reuses the existing measurements.',
    'minimum_quality': 'Absolute Kraaikamp-style estimator score, as shown by Absolute quality on the graph. Scientific notation is accepted. Only scores strictly above this value are selected.',
    'bayer_override': 'Use the capture header, force monochrome, or select the sensor Bayer pattern. Changes the measured frames and the output header; exported samples are unchanged.',
    'reject_saturated': 'Screen out frames with more than 5% of samples at the detector ceiling, or any sample at an explicitly set saturation threshold.',
    'device': 'Backend for frame measurements. Auto uses a supported CUDA GPU for filtering when available; silhouette measurements use CPU.',
    'threads': 'CPU worker limit for frame measurements and cache validation. Defaults to available logical CPUs (up to 32).',
    'batch_frames': 'Number of frames read per batch. Larger batches use more memory; progress updates after each batch.',
    'gain_e_per_adu': 'Electrons per ADU applied before measuring quality. Leave blank to measure in the capture units. Exported samples are never calibrated.',
    'read_noise_e': 'Optional read noise in electrons, recorded with the measurements. Must match to reuse a cache made with it.',
}


class SERControls(QScrollArea):
    """Selection, capture interpretation, processing and calibration settings."""
    selectionChanged = Signal()
    inputChanged = Signal()
    changed = Signal()

    def __init__(self, *, dialog_directory=None, remember_directory=None):
        super().__init__()
        self.dialog_directory = dialog_directory
        self.remember_directory = remember_directory
        self.fields = {}
        self.labels = {}
        self.setWidgetResizable(True)
        body = QWidget()
        layout = QVBoxLayout(body)
        defaults = ReconstructionConfig()

        form = self._group(layout, 'Frame selection')
        self._choice(form, 'selection_mode', 'Select by', [
            ('Quality range', 'quality_range'), ('Frame count', 'frame_count'),
            ('Absolute quality', 'absolute')])
        self._integer(form, 'stack_percent', 'Upper quality range (%)', 1, 100, 50)
        self.percent_label = form.labelForField(self.fields['stack_percent'])
        self._text(form, 'minimum_quality', 'Quality above', '0')

        form = self._group(layout, 'SER interpretation')
        self._choice(form, 'bayer_override', 'Raw colour override', [
            ('Header / automatic', None), *((name, name) for name in ('mono', 'RGGB', 'GRBG', 'GBRG', 'BGGR'))])
        self._choice(form, 'endian_override', 'Byte order override', [
            ('Header / automatic', None), ('little', 'little'), ('big', 'big')])
        self._choice(form, 'endian_convention', 'Byte-order convention', [
            ('ecosystem', 'ecosystem'), ('spec', 'spec')])
        self._check(form, 'recover_complete_frames', 'Recover complete frames', defaults.recover_complete_frames)
        self._check(form, 'reject_saturated', 'Reject saturated frames', defaults.reject_saturated)

        form = self._group(layout, 'Processing')
        self._choice(form, 'device', 'Device', [(name, name) for name in ('auto', 'cpu', 'gpu')])
        self._integer(form, 'threads', 'CPU threads', 1, 32, detected_thread_count())
        self._integer(form, 'batch_frames', 'Frames per batch', 1, 4096, defaults.batch_frames)

        form = self._group(layout, 'Calibration (optional)')
        note = QLabel('NPY tables must match the raw detector shape. Used for measurements only; '
                      'exported frames stay uncalibrated.')
        note.setWordWrap(True)
        form.addRow(note)
        for name in ('bias', 'dark', 'flat'):
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            edit = QLineEdit()
            edit.setPlaceholderText('No table')
            button = QPushButton('…')
            button.setMaximumWidth(30)
            button.setToolTip(f'Choose the {name} calibration NPY table.')
            button.clicked.connect(lambda checked=False, e=edit: self._choose_table(e))
            row_layout.addWidget(edit)
            row_layout.addWidget(button)
            self.fields[name + '_path'] = edit
            self.labels[name + '_path'] = name.title()
            form.addRow(name.title(), row)
        self._text(form, 'gain_e_per_adu', 'Gain (e−/ADU)')
        self._text(form, 'read_noise_e', 'Read noise (e−; metadata)')
        self._text(form, 'saturate_adu', 'Saturation threshold (ADU)')
        layout.addStretch(1)
        self.setWidget(body)

        for key, edit in self.fields.items():
            edit.setToolTip(HELP[key])
            signal = (edit.currentIndexChanged if isinstance(edit, QComboBox) else
                      edit.toggled if isinstance(edit, QCheckBox) else
                      edit.valueChanged if isinstance(edit, QSpinBox) else edit.textChanged)
            signal.connect(lambda *_, key=key: self._edited(key))
        self._selection_mode_changed()

    def _group(self, layout, title):
        box = QGroupBox(title)
        form = QFormLayout(box)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        layout.addWidget(box)
        return form

    def _choice(self, form, key, label, choices):
        edit = QComboBox()
        for text, value in choices:
            edit.addItem(text, value)
        self.fields[key], self.labels[key] = edit, label
        form.addRow(label, edit)

    def _integer(self, form, key, label, low, high, value):
        edit = QSpinBox()
        edit.setRange(low, high)
        edit.setValue(value)
        self.fields[key], self.labels[key] = edit, label
        form.addRow(label, edit)

    def _check(self, form, key, label, value):
        edit = QCheckBox(label)
        edit.setChecked(value)
        self.fields[key], self.labels[key] = edit, label
        form.addRow(edit)

    def _text(self, form, key, label, value=''):
        edit = QLineEdit(value)
        edit.setPlaceholderText('Optional')
        self.fields[key], self.labels[key] = edit, label
        form.addRow(label, edit)

    def _choose_table(self, edit):
        directory = self.dialog_directory() if self.dialog_directory else ''
        path, _ = QFileDialog.getOpenFileName(edit, 'Calibration table', directory, 'NumPy table (*.npy)')
        if path:
            edit.setText(path)
            if self.remember_directory:
                self.remember_directory(Path(path).parent)

    def _edited(self, key):
        if key == 'selection_mode':
            self._selection_mode_changed()
        if key in SELECTION_KEYS:
            self.selectionChanged.emit()
        elif key in INPUT_KEYS:
            self.inputChanged.emit()
        self.changed.emit()

    def _selection_mode_changed(self):
        mode = self.selection_mode()
        self.percent_label.setText('Best screened frames (%)' if mode == 'frame_count'
                                   else 'Upper quality range (%)')
        self.fields['stack_percent'].setEnabled(mode != 'absolute')
        self.fields['minimum_quality'].setEnabled(mode == 'absolute')

    def selection_mode(self):
        return self.fields['selection_mode'].currentData()

    def minimum_quality(self):
        """Typed absolute cutoff, or None while a percentage selects frames."""
        return self.fields['minimum_quality'].text() if self.selection_mode() == 'absolute' else None

    def _value(self, key):
        edit = self.fields[key]
        if isinstance(edit, QComboBox):
            return edit.currentData()
        if isinstance(edit, QCheckBox):
            return edit.isChecked()
        if isinstance(edit, QSpinBox):
            edit.interpretText()  # Commit typed text before focus leaves the editor.
            return edit.value()
        text = edit.text().strip()
        if key.endswith('_path'):
            return text or None
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            raise ValueError(f'{self.labels[key]}: enter a number') from None

    def configuration(self):
        """Engine settings for measurement, cache validation and export."""
        values = {key: self._value(key) for key in self.fields
                  if key not in ('selection_mode', 'minimum_quality')}
        mode = self.selection_mode()
        return ReconstructionConfig(frame_preselection=True,
            frame_selection_mode='quality_range' if mode == 'absolute' else mode, **values)

    def settings_state(self):
        state = {}
        for key, edit in self.fields.items():
            state[key] = edit.text() if isinstance(edit, QLineEdit) else self._value(key)
        return state

    def restore_settings(self, state):
        if not isinstance(state, dict):
            raise ValueError('invalid saved controls')
        for key, value in state.items():
            edit = self.fields.get(key)
            if edit is None:
                continue
            edit.blockSignals(True)
            try:
                if isinstance(edit, QComboBox):
                    index = edit.findData(value)
                    if index >= 0:
                        edit.setCurrentIndex(index)
                elif isinstance(edit, QCheckBox) and isinstance(value, bool):
                    edit.setChecked(value)
                elif isinstance(edit, QSpinBox) and type(value) is int:
                    edit.setValue(value)
                elif isinstance(edit, QLineEdit) and isinstance(value, str):
                    edit.setText(value)
            finally:
                edit.blockSignals(False)
        self._selection_mode_changed()


class SERWindow:
    def __init__(self, path: Path | None = None, *, settings_path: Path | None = None):
        from planetrecon.gui import settings
        self.settings_path = (Path(settings_path) if settings_path is not None else
                              settings.default_path('ser-settings.json', 'PLANETRECON_SER_SETTINGS_PATH'))
        saved, settings_error = settings.load(self.settings_path)
        self.last_directory = saved.get('last_directory', '')
        if not isinstance(self.last_directory, str):
            self.last_directory = ''
        self.path = Path(path) if path else None
        self.config = None  # Capture interpretation of the active or last job.
        self.job: JobHandle | None = None
        self.stage = None
        self.measure_after = False
        self.cancel_started = None
        self.started = None
        self.preprocessing_info = {}
        self.cache_validation = None
        self.frame_source = None
        self.levels = None
        self.export_worker = None
        self.closing = False
        owner = self

        class OwnedWindow(QMainWindow):
            def closeEvent(self, event):
                owner._shutdown()
                if owner.export_worker is not None:
                    owner.closing = True
                    owner.status.setText('Cancelling export; the window will close after the writer returns.')
                    event.ignore()
                else:
                    event.accept()

        self.window = OwnedWindow()
        self.window.setWindowTitle('PlanetRecon SER — frame quality and filtered export')
        self.window.resize(1160, 820)
        root = QWidget()
        layout = QVBoxLayout(root)
        buttons = QHBoxLayout()
        self.open_btn = QPushButton('Open SER…')
        self.open_btn.setToolTip('Open a SER capture. A preprocessing cache beside the capture is validated and reused. Without a cache, frame quality is measured and cached; a cache that does not match is kept until Preprocess replaces it.')
        self.preprocess_btn = QPushButton('Preprocess')
        self.preprocess_btn.setToolTip('Measure frame quality and shape, and save the reusable cache beside the capture, replacing an existing cache. After a settings change, a cache that still matches is reused instead.')
        self.cancel_btn = QPushButton('Cancel')
        self.cancel_btn.setToolTip('Cancel validation or preprocessing. An existing cache is not replaced.')
        for button, slot in ((self.open_btn, self._choose), (self.preprocess_btn, self._preprocess),
                             (self.cancel_btn, self._cancel)):
            button.clicked.connect(slot)
            buttons.addWidget(button)
        buttons.addStretch()
        layout.addLayout(buttons)
        self.source_label = QLabel(str(self.path) if self.path else 'Open a SER capture')
        self.source_label.setWordWrap(True)
        layout.addWidget(self.source_label)
        self.preprocessing_label = QLabel('No preprocessing measurements loaded.')
        self.preprocessing_label.setWordWrap(True)
        layout.addWidget(self.preprocessing_label)

        split = QSplitter()
        self.controls = SERControls(dialog_directory=self._dialog_directory,
                                    remember_directory=self._remember_directory)
        self.controls.setMinimumWidth(300)
        split.addWidget(self.controls)
        self.quality_plot = QualityPlot(registration_note=False)
        self.quality_plot.frameRequested.connect(self._frame_requested)
        export_row = QHBoxLayout()
        self.export_btn = QPushButton('Export SER…')
        self.export_btn.setToolTip('Save the green selected frames as a new SER. Original full frames, capture order and timestamps are retained; samples are copied unchanged.')
        self.export_btn.clicked.connect(self._choose_export)
        self.cancel_export_btn = QPushButton('Cancel export')
        self.cancel_export_btn.clicked.connect(self._cancel_export)
        self.export_count = QLabel('')
        self.export_status = QLabel('')
        self.export_status.setWordWrap(True)
        for widget in (self.export_btn, self.cancel_export_btn, self.export_count, self.export_status):
            export_row.addWidget(widget)
        export_row.setStretch(3, 1)
        self.quality_plot.layout().addLayout(export_row)
        split.addWidget(self.quality_plot)
        split.setSizes([320, 840])
        layout.addWidget(split, 1)

        self.progress = QProgressBar()
        layout.addWidget(self.progress)
        self.status = QLabel('idle')
        self.status.setWordWrap(True)
        self.error = QLabel('')
        self.error.setWordWrap(True)
        self.error.setTextFormat(Qt.PlainText)
        self.error.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard)
        self.error.setStyleSheet('color: #bb3333; font-weight: bold')
        layout.addWidget(self.status)
        layout.addWidget(self.error)
        self.window.setCentralWidget(root)

        if saved:
            try:
                self.controls.restore_settings(saved.get('controls', {}))
                plot = saved.get('quality_plot', {})
                if isinstance(plot, dict):
                    for key in ('absolute', 'log'):
                        if isinstance(plot.get(key), bool):
                            getattr(self.quality_plot, f'{key}_control').setChecked(plot[key])
            except (TypeError, ValueError) as exc:
                settings_error = f'Could not fully restore settings: {exc}'
        if settings_error:
            self.error.setText(settings_error)
        self.timer = QTimer(self.window)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self._poll)
        self.settings_timer = QTimer(self.window)
        self.settings_timer.setSingleShot(True)
        self.settings_timer.setInterval(500)
        self.settings_timer.timeout.connect(self._save_settings)
        self.controls.changed.connect(self.settings_timer.start)
        self.controls.selectionChanged.connect(self._refresh)
        self.controls.inputChanged.connect(self._input_changed)
        for control in (self.quality_plot.absolute_control, self.quality_plot.log_control):
            control.toggled.connect(lambda *_: self.settings_timer.start())
        self._refresh()
        if self.path is not None:
            self._load()

    def show(self):
        self.window.show()

    # Preferences

    def _save_settings(self):
        from planetrecon.gui import settings
        try:
            settings.save(self.settings_path, dict(
                controls=self.controls.settings_state(),
                quality_plot=dict(absolute=self.quality_plot.absolute_control.isChecked(),
                                  log=self.quality_plot.log_control.isChecked()),
                capture=str(self.path) if self.path else None,
                last_directory=self.last_directory))
        except (OSError, ValueError) as exc:
            self.error.setText(f'Could not save settings: {exc}')

    def _dialog_directory(self):
        if self.last_directory and Path(self.last_directory).is_dir():
            return self.last_directory
        if self.path is not None and self.path.absolute().parent.is_dir():
            return str(self.path.absolute().parent)
        return str(Path.cwd())

    def _remember_directory(self, directory):
        self.last_directory = str(Path(directory).absolute())
        self._save_settings()

    # Capture and preprocessing

    def _buttons(self):
        idle = self.job is None and self.export_worker is None and not self.closing
        info = self.preprocessing_info
        ready = (self.path is not None and info.get('status') == 'ready' and bool(info.get('digest'))
                 and self.quality_plot.selection is not None)
        count = int(self.quality_plot.selected.sum()) if ready else 0
        self.export_count.setText(f'Export: {count:,}/{len(self.quality_plot.selected):,} frames' if ready else '')
        self.open_btn.setEnabled(idle)
        self.preprocess_btn.setEnabled(idle and self.path is not None)
        self.cancel_btn.setEnabled(self.job is not None and self.cancel_started is None)
        self.controls.setEnabled(idle)
        self.quality_plot.set_browsing_enabled(self.job is None and not self.closing)
        self.export_btn.setEnabled(bool(ready and idle and count))
        self.cancel_export_btn.setEnabled(self.export_worker is not None)

    def _choose(self):
        name, _ = QFileDialog.getOpenFileName(self.window, 'Open SER capture', self._dialog_directory(),
                                              'SER captures (*.ser)')
        if name:
            self.open_capture(name)

    def open_capture(self, path):
        if self.job is not None or self.export_worker is not None or self.closing:
            return False
        self.path = Path(path)
        self.last_directory = str(self.path.absolute().parent)
        self.cache_validation = None
        self.export_status.clear()
        self._save_settings()
        self._load()
        return True

    def _cache_exists(self):
        return self.path.with_name(self.path.name + '.planetrecon-preprocess.npz').is_file()

    def _load(self):
        """Validate a cache beside the capture; measure only when there is none."""
        self._clear_frames()
        self.source_label.setText(str(self.path))
        cached = self._cache_exists()
        self.preprocessing_info = {'status': 'unverified', 'reason': (
            'Loading cached preprocessing; validating capture data.' if cached
            else 'No preprocessing cache; measuring frame quality.')}
        self._refresh()
        self._start('inspect' if cached else 'preprocess')

    def _preprocess(self):
        if self.path is None:
            return
        # Settings changed since the last check: an existing cache may still
        # match them. Otherwise the request is for new measurements.
        if self.preprocessing_info.get('status') == 'unverified' and self._cache_exists():
            self._start('inspect', measure_after=True)
        else:
            self._start('preprocess')

    def _start(self, stage, *, measure_after=False):
        if self.job is not None or self.export_worker is not None or self.path is None or self.closing:
            return
        self._close_frame_source()
        self.error.clear()
        try:
            if self.path.suffix.lower() != '.ser':
                raise ValueError('This tool reads SER captures only.')
            cfg = self.controls.configuration()
            handle = start_stack_job(self.path, cfg, emit_previews=False,
                inspect_only=stage == 'inspect', preprocess_only=stage == 'preprocess',
                measure_geometry=stage != 'preprocess', cache_validation=self.cache_validation)
        except (ValueError, TypeError, OSError) as exc:
            self.error.setText(str(exc))
            if self.preprocessing_info.get('status') == 'unverified':
                self._set_preprocessing({'status': 'unverified', 'reason':
                    'Capture not loaded. Correct the reported problem, then click Preprocess.'})
            self._buttons()
            return
        self.config = cfg
        self._save_settings()
        self.job, self.stage, self.measure_after = handle, stage, measure_after
        self.cancel_started = None
        self.started = time.monotonic()
        self.progress.setRange(0, 0)
        self.status.setText(f'Opening {self.path.name}')
        self.timer.start()
        self._buttons()

    def _cancel(self):
        if self.job is not None and self.cancel_started is None:
            self.cancel_started = time.monotonic()
            self.job.cancel_event.set()
            self.status.setText('Cancelling…')
        self._buttons()

    def _poll(self):
        try:
            self._poll_events()
        except Exception as exc:
            # A consumed terminal event must never strand its finished worker.
            self.error.setText(f'Could not display processing update: {exc}')
            self.status.setText('Processing stopped. Reopen the capture or retry Preprocess.')
            self._finish_job()

    def _poll_events(self):
        handle = self.job
        if handle is None:
            return
        if self.cancel_started is not None:
            elapsed = time.monotonic() - self.cancel_started
            if elapsed > 3 and handle.process.is_alive():
                handle.process.kill()
            elif elapsed > 2 and handle.process.is_alive():
                handle.process.terminate()
        for event in handle.poll():
            if event.kind == 'source':
                self._set_input(event.payload)
            elif event.kind == 'progress' and self.cancel_started is None:
                self._progress(event.payload)
            elif event.kind in ('completed', 'cancelled', 'error'):
                self._complete(event)
                break

    def _progress(self, payload):
        fraction = payload.get('fraction')
        self.progress.setRange(0, 0 if fraction is None else 100)
        if fraction is not None:
            self.progress.setValue(min(99, max(0, round(100 * fraction))))
        elapsed = time.monotonic() - self.started
        eta = f' · ~{elapsed * (1 - fraction) / fraction:.0f}s remaining' if fraction and 0 < fraction < 1 else ''
        backend = payload.get('backend')
        self.status.setText(f"{payload.get('stage', '')}" + (f' · {backend.upper()}' if backend else '')
                            + f' · {elapsed:.1f}s elapsed{eta}')

    def _complete(self, event):
        stage, kind = self.stage, event.kind
        measure = False
        try:
            if kind == 'completed':
                payload = event.payload
                self.cache_validation = payload.get('cache_validation')
                self._set_input(payload)
                info = payload['preprocessing_cache']
                ready = info.get('status') == 'ready'
                # Opening a capture never replaces a cache made with other settings.
                measure = not ready and (self.measure_after or info.get('status') == 'missing')
                if not ready and not measure:
                    info = {**info, 'reason': (
                        'The preprocessing cache beside this capture was measured with a different capture '
                        'interpretation or calibration. Restore those settings and click Preprocess, or '
                        'click Preprocess with the current settings to measure again and replace it.'
                        if info.get('status') == 'stale' else
                        info.get('reason', 'The preprocessing cache cannot be used').rstrip('.')
                        + '. Click Preprocess to measure again and replace it.')}
                self._set_preprocessing(info)
                self.status.setText(
                    ('Frame quality measured and cached.' if stage == 'preprocess'
                     else 'Cached frame quality validated.') + ' Click the graph to view a frame.' if ready
                    else 'No matching preprocessing cache; measuring frame quality.' if measure
                    else 'The existing preprocessing cache was not used.')
                self.progress.setRange(0, 100)
                self.progress.setValue(100)
            else:
                if kind == 'error':
                    self.error.setText('Processing failed: ' + event.payload.get('message', 'unknown error'))
                    self.status.setText('No new measurements were saved.')
                else:
                    self.status.setText('Cancelled. An existing cache was not replaced.')
                if self.preprocessing_info.get('status') == 'unverified':
                    self._set_preprocessing({'status': 'unverified', 'reason':
                        'Frame quality is not loaded. Click Preprocess to validate or measure this capture.'})
        except Exception as exc:
            measure = False
            self.error.setText(f'Could not display processing result: {exc}')
            self.status.setText('Processing stopped. Reopen the capture or retry Preprocess.')
        finally:
            self._finish_job()
        if measure:
            self._start('preprocess')

    def _finish_job(self):
        self.timer.stop()
        if self.job is not None:
            self.job.close()
            self.job = None
        self.stage = None
        if self.progress.maximum() == 0:
            self.progress.setRange(0, 100)
            self.progress.setValue(0)
        self.cancel_started = None
        self._buttons()

    def _set_input(self, payload):
        meta = payload['source_metadata']
        image = np.asarray(payload['input_image'])
        samples = image[np.isfinite(image)]
        black = float(np.percentile(samples, 1)) if samples.size else 0.
        maximum = payload.get('input_max', float(samples.max()) if samples.size else 1.)
        bit_depth = meta.get('bit_depth', 32)
        white = min(1.43 * maximum, (1 << bit_depth) - 1 if bit_depth <= 16 else float('inf'))
        # One linear mapping for the whole capture keeps frames comparable.
        self.levels = (black, max(black + 1., white))
        try:
            frame = self._read_frame(payload.get('input_frame_index', 0))
        except (OSError, ValueError, IndexError):
            frame = payload
        self._display_frame(frame)
        timing = payload.get('capture_timing', {})
        duration = (f"{timing['duration_s']:.6g} s ({timing.get('origin', 'measured')})"
                    if timing.get('status') == 'available' else 'unavailable')
        exposure = meta.get('extras', {}).get('exposure_s', {}).get('value')
        self.source_label.setText(
            f"{meta['path']} · {meta['width']}×{meta['height']} · {meta['n_frames']:,} frames · "
            f"{meta['color_mode']} · {meta['bit_depth']} bit · duration: {duration} · "
            f"recorded exposure: {str(exposure) + ' s' if exposure is not None else 'unavailable'}")

    def _set_preprocessing(self, info):
        self.preprocessing_info = info
        self._refresh()

    def _refresh(self, *_):
        info = self.preprocessing_info
        mode = self.controls.selection_mode()
        self.quality_plot.set_report(info, self.controls.fields['stack_percent'].value(),
            'quality_range' if mode == 'absolute' else mode, True,
            minimum_quality=self.controls.minimum_quality())
        if info.get('status') == 'ready':
            execution = info.get('execution') or {}
            best = info.get('best_reference_index')
            self.preprocessing_label.setText(
                f"Screening: {info['quality']} quality / {info['shape']} shape exclusions "
                f"({info['quality_shape_overlap']} overlap), {info['other']} other; "
                f"{info['excluded']:,} excluded total, {info['accepted']:,}/{info['n_total']:,} retained. "
                f"Sharpest retained frame: {'unavailable' if best is None else format(best + 1, ',')}."
                + (f" Measured on {execution['backend'].upper()}." if execution.get('backend') else ''))
        else:
            self.preprocessing_label.setText(info.get('reason', 'No preprocessing measurements loaded.'))
        self._buttons()

    def _input_changed(self):
        if self.path is None or self.preprocessing_info.get('status') in (None, 'unverified'):
            return
        self._clear_frames()
        self.preprocessing_info = {'status': 'unverified', 'reason':
            'Input or calibration settings changed. Click Preprocess to reuse a matching cache or measure again.'}
        self._refresh()

    # Frame display: one short read in this process keeps browsing immediate.

    def _read_frame(self, index):
        if self.frame_source is None:
            from planetrecon.io.ser import SERSource
            cfg = self.config
            self.frame_source = SERSource(self.path, bayer_override=cfg.bayer_override,
                endian_override=cfg.endian_override, endian_convention=cfg.endian_convention,
                recover_complete_frames=cfg.recover_complete_frames)
        return input_frame_preview(self.frame_source, int(index), FRAME_PREVIEW_LIMIT)

    def _display_frame(self, payload):
        black, white = self.levels or (None, None)
        self.quality_plot.set_frame_preview(payload.get('input_frame_index', 0),
            _to_qimage(payload['input_image'], black, white))

    def _frame_requested(self, index):
        if (self.path is None or self.config is None or self.job is not None or self.closing
                or not 0 <= index < len(self.quality_plot.selected)):
            return
        try:
            self._display_frame(self._read_frame(index))
        except (OSError, ValueError, IndexError) as exc:
            self._close_frame_source()
            self.quality_plot.clear_preview()
            self.quality_plot.frame_label.setText(f'Could not load frame {index + 1:,}: {exc}')

    def _close_frame_source(self):
        if self.frame_source is not None:
            self.frame_source.close()
            self.frame_source = None

    def _clear_frames(self):
        self._close_frame_source()
        self.levels = None
        self.quality_plot.clear_preview()

    # Filtered export

    def _choose_export(self):
        if not self.export_btn.isEnabled():
            return
        path, _ = QFileDialog.getSaveFileName(self.window, 'Export filtered SER',
            str(Path(self._dialog_directory()) / (self.path.stem + '_filtered.ser')),
            'SER capture (*.ser)', options=QFileDialog.Option.DontConfirmOverwrite)
        if not path:
            return
        destination = Path(path)
        if not destination.suffix:
            destination = destination.with_suffix('.ser')
        self._remember_directory(destination.parent)
        overwrite = False
        if destination.exists() or destination.is_symlink():
            overwrite = QMessageBox.question(self.window, 'Replace SER?', f'Replace {destination}?',
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes
            if not overwrite:
                return
        self.export_ser(destination, overwrite=overwrite)

    def export_ser(self, path, *, overwrite=False):
        self._buttons()
        if not self.export_btn.isEnabled():
            return False
        try:
            config = self.controls.configuration()
            minimum = self.controls.minimum_quality()
            minimum = None if minimum is None else float(minimum)
        except ValueError as exc:
            self.export_status.setText(f'Export settings: {exc}')
            return False
        info = self.preprocessing_info
        self.export_worker = SERExportWorker(self.path, Path(path), config, info.get('path'), info['digest'],
            dict(self.cache_validation) if self.cache_validation else None, overwrite, minimum)
        self.export_worker.progress.connect(self._export_progress)
        self.export_worker.outcome.connect(self._export_outcome)
        self.export_worker.finished.connect(self._export_finished)
        self.export_status.setText('Preparing filtered SER export…')
        self.export_worker.start()
        self._buttons()
        return True

    def _export_progress(self, stage, done, total):
        self.export_status.setText(f'{stage}: {done:,}/{total:,} frames')

    def _cancel_export(self):
        if self.export_worker is not None:
            self.export_worker.cancel_event.set()
            self.export_status.setText('Cancelling SER export…')

    def _export_outcome(self, report, error):
        if error is not None:
            self.export_status.setText('SER export cancelled.' if isinstance(error, ExportCancelled)
                                       else f'SER export failed: {error}')
        else:
            self.export_status.setText(
                f'Saved {report.path} · {report.n_written:,}/{report.n_input:,} frames in capture order')

    def _export_finished(self):
        worker = self.export_worker
        if worker is not None:
            worker.wait()
            worker.deleteLater()
            self.export_worker = None
        self._buttons()
        if self.closing:
            self.window.close()

    def _shutdown(self, *_):
        self.settings_timer.stop()
        self._save_settings()
        self._finish_job()
        self._close_frame_source()
        self._cancel_export()


def main(argv=None):
    apply_thread_limits()
    args = list(sys.argv[1:] if argv is None else argv)
    app = create_app(args)
    win = SERWindow(Path(args[0]) if args else None)
    app.aboutToQuit.connect(win._shutdown)
    win.show()
    code = app.exec()
    # An external application quit also owns any remaining writer thread.
    win._shutdown()
    if win.export_worker is not None:
        win.export_worker.wait()
    return code


if __name__ == '__main__':
    from multiprocessing import freeze_support

    freeze_support()
    raise SystemExit(main())
