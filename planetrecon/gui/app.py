"""Qt6 capture workflow with owned workers and full-resolution scientific saves."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import threading
import time

import numpy as np
from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QHBoxLayout,
    QLabel, QLineEdit, QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar,
    QPushButton, QScrollArea, QSplitter, QVBoxLayout, QWidget,
)

from planetrecon.export import ExportCancelled, ExportConfig, export_result
from planetrecon.gui.controls import ConfigControls
from planetrecon.gui.preview import display_result_preview
from planetrecon.jobs import JobHandle, result_from_payload, start_stack_job
from planetrecon.reconstruction import ReconstructionConfig
from planetrecon.runtime import apply_thread_limits


def _to_qimage(image, black=None, white=None, validity=None):
    arr = np.asarray(image, dtype=np.float64)
    valid = np.isfinite(arr)
    if validity is not None:
        mask = np.asarray(validity)
        if mask.ndim == 2 and arr.ndim == 3:
            mask = mask[..., None]
        valid &= mask
    finite = arr[valid]
    if black is None:
        black = float(np.percentile(finite, 1)) if finite.size else 0.
    automatic_white = white is None
    if automatic_white:
        dtype = np.asarray(image).dtype
        full_scale = float(np.iinfo(dtype).max) if np.issubdtype(dtype, np.integer) else float('inf')
        white = min(1.43 * float(finite.max()), full_scale) if finite.size else 1.
    if white <= black:
        if automatic_white:black = white - 1.
        else:white = black + 1.
    scaled = np.clip((np.where(valid, arr, black) - black) / (white - black), 0, 1)
    rgb = np.repeat(scaled[..., None], 3, axis=2) if arr.ndim == 2 else scaled
    rgb8 = (rgb * 255).astype(np.uint8)
    supported = valid if arr.ndim == 2 else valid.all(axis=2)
    # Magenta marks incomplete colour support; it never enters the science array.
    rgb8[~supported] = [180, 40, 160]
    rgb8 = np.ascontiguousarray(rgb8)
    h, w = rgb8.shape[:2]
    return QImage(rgb8.data, w, h, 3*w, QImage.Format.Format_RGB888).copy()


def _pixmap(image):
    return QPixmap.fromImage(_to_qimage(image))


def create_app(argv=None):
    return QApplication.instance() or QApplication(sys.argv if argv is None else argv)


class ExportWorker(QThread):
    outcome = Signal(object, object)

    def __init__(self, result, path, config, overwrite):
        super().__init__()
        self.result, self.path, self.config, self.overwrite = result, path, config, overwrite
        self.cancel_event = threading.Event()

    def run(self):
        try:
            report = export_result(self.result, self.path, self.config, overwrite=self.overwrite,
                                   should_cancel=self.cancel_event.is_set)
            self.outcome.emit(report, None)
        except Exception as exc:
            self.outcome.emit(None, exc)
        finally:
            self.result = None


class MainWindow:
    def __init__(self, path: Path | None = None, config: ReconstructionConfig | None = None):
        # New interactive jobs use the preferred stacking preset; supplied jobs
        # and saved engine configurations retain their explicit settings.
        self.config = config or ReconstructionConfig(
            device='auto', local_alignment=True, stack_percent=50,
            frame_selection_mode='quality_range')
        self.path = Path(path) if path else None
        self.job: JobHandle | None = None
        self.last_result = None
        self.preview = None
        self.input_image = None
        self.input_metadata = {}
        self.input_max = None
        self.export_worker = None
        self.closing = False
        self.cancel_started = None
        self.last_seq = 0
        self.auto_levels = True
        self.inspecting = False
        self.started = None
        owner = self

        class OwnedWindow(QMainWindow):
            def closeEvent(self, event):
                owner._shutdown()
                if owner.export_worker is not None:
                    owner.closing = True
                    owner.status.setText('Cancelling save; the window will close after the encoder returns.')
                    event.ignore()
                else:
                    event.accept()

            def resizeEvent(self, event):
                super().resizeEvent(event)
                if hasattr(owner, 'canvas'):
                    owner._draw()

        self.window = OwnedWindow()
        self.redraw_timer = QTimer(self.window)
        self.redraw_timer.setSingleShot(True)
        self.redraw_timer.setInterval(0)
        self.redraw_timer.timeout.connect(self._draw)
        self.window.setWindowTitle('PlanetRecon — capture reconstruction')
        self.window.resize(1160, 820)
        root = QWidget()
        layout = QVBoxLayout(root)
        buttons = QHBoxLayout()
        self.open_btn = QPushButton('Open capture…')
        self.inspect_btn = QPushButton('Inspect input')
        self.preprocess_btn = QPushButton('Preprocess')
        self.preprocess_btn.setToolTip('Measure quality, shape and geometry independently, then save a reusable cache beside the capture. Replaces only this capture’s preprocessing cache; does not reconstruct an image.')
        self.run_btn = QPushButton('Run')
        self.cancel_btn = QPushButton('Cancel processing')
        for b, slot in ((self.open_btn, self._choose), (self.inspect_btn, self._inspect),
                        (self.preprocess_btn, self._preprocess),
                        (self.run_btn, self._run), (self.cancel_btn, self._cancel)):
            b.clicked.connect(slot)
            buttons.addWidget(b)
        buttons.addStretch()
        layout.addLayout(buttons)
        checkpoint_row = QHBoxLayout()
        self.checkpoint_path = QLineEdit()
        self.checkpoint_path.setPlaceholderText('Optional accumulator checkpoint file')
        self.checkpoint_btn = QPushButton('Checkpoint file…')
        self.checkpoint_btn.clicked.connect(self._choose_checkpoint)
        self.resume_check = QCheckBox('Resume this checkpoint')
        checkpoint_row.addWidget(self.checkpoint_path, 1)
        checkpoint_row.addWidget(self.checkpoint_btn)
        checkpoint_row.addWidget(self.resume_check)
        layout.addLayout(checkpoint_row)
        self.source_label = QLabel(str(self.path) if self.path else 'Open a SER, AVI or observed HDF5 capture')
        self.source_label.setWordWrap(True)
        layout.addWidget(self.source_label)
        self.preprocessing_info = {}
        self.preprocessing_label = QLabel('No preprocessing measurements loaded.')
        self.preprocessing_label.setWordWrap(True)
        layout.addWidget(self.preprocessing_label)
        split = QSplitter()
        self.controls = ConfigControls(self.config)
        self.controls.fields['frame_preselection'].toggled.connect(self._refresh_preprocessing)
        self.controls.fields['stack_percent'].valueChanged.connect(self._refresh_preprocessing)
        self.controls.fields['frame_selection_mode'].currentIndexChanged.connect(self._refresh_preprocessing)
        for key in ('bayer_override', 'endian_override', 'endian_convention', 'crop',
                    'recover_complete_frames', 'reject_saturated', 'bias_path', 'dark_path',
                    'flat_path', 'gain_e_per_adu', 'read_noise_e', 'saturate_adu'):
            edit = self.controls.fields[key]
            signal = edit.currentIndexChanged if isinstance(edit, QComboBox) else (edit.toggled if isinstance(edit, QCheckBox) else edit.textChanged)
            signal.connect(self._preprocessing_settings_changed)
        self.controls.setMinimumWidth(360)
        self.device = self.controls.fields['device']
        split.addWidget(self.controls)
        right = QWidget()
        body = QVBoxLayout(right)
        toolbar = QHBoxLayout()
        self.view = QComboBox()
        self.view.addItems(['Result', 'Input', 'Coverage', 'Validity', 'Globe coverage', 'Ring coverage'])
        self.channel = QComboBox()
        self.channel.addItems(['RGB / mono', 'R', 'G', 'B'])
        self.zoom = QComboBox()
        self.zoom.addItems(['Fit', '25%', '50%', '100%', '200%', '400%'])
        for widget in (self.view, self.channel, self.zoom):
            toolbar.addWidget(widget)
            widget.currentIndexChanged.connect(self._draw)
        body.addLayout(toolbar)
        class ResultCanvas(QScrollArea):
            def resizeEvent(self, event):
                super().resizeEvent(event)
                # Result metadata can resize the canvas without resizing the
                # main window. Fit against the settled layout dimensions.
                owner.redraw_timer.start()
        self.canvas = ResultCanvas()
        self.canvas.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label = QLabel('Run a capture to build an image')
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.canvas.setWidget(self.image_label)
        body.addWidget(self.canvas, 1)
        levels = QHBoxLayout()
        self.black = QDoubleSpinBox()
        self.white = QDoubleSpinBox()
        for edit in (self.black, self.white):
            edit.setRange(-1e12, 1e12)
            edit.setDecimals(5)
            edit.valueChanged.connect(self._draw)
        self.white.setValue(1.)
        levels.addWidget(QLabel('Display black'))
        levels.addWidget(self.black)
        levels.addWidget(QLabel('white'))
        levels.addWidget(self.white)
        fit = QPushButton('Fit levels')
        fit.setToolTip('Set display black to the first percentile and white to 1.43 times the brightest valid pixel, capped at the capture range. Changes viewing levels; TIFF32 can use these levels for reversible linear scaling.')
        fit.clicked.connect(self._fit_levels)
        levels.addWidget(fit)
        body.addLayout(levels)
        self.histogram = QLabel('Display is linear; magenta marks incomplete sample support.')
        self.histogram.setWordWrap(True)
        body.addWidget(self.histogram)
        self.result_label = QLabel('No scientific result yet')
        self.result_label.setWordWrap(True)
        body.addWidget(self.result_label)
        self.warnings = QLabel('')
        self.warnings.setWordWrap(True)
        body.addWidget(self.warnings)
        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        self.details.setMaximumHeight(105)
        self.details.setPlaceholderText('Source metadata, result provenance and warnings')
        body.addWidget(self.details)
        save_row = QHBoxLayout()
        self.encoding = QComboBox()
        self.encoding.addItems(['tiff32', 'tiff16', 'png16', 'tiff32_raw'])
        self.save_black = QLineEdit()
        self.save_black.setPlaceholderText('Black (display black if empty)')
        self.save_white = QLineEdit()
        self.save_white.setPlaceholderText('White (display white if empty)')
        self.save_gamma = QLineEdit()
        self.save_gamma.setPlaceholderText('Display gamma (optional)')
        self.save_btn = QPushButton('Save result…')
        self.save_btn.clicked.connect(self._choose_save)
        self.cancel_save_btn = QPushButton('Cancel save')
        self.cancel_save_btn.clicked.connect(self._cancel_save)
        for widget in (self.encoding, self.save_black, self.save_white, self.save_gamma, self.save_btn, self.cancel_save_btn):
            save_row.addWidget(widget)
        body.addLayout(save_row)
        self.save_status = QLabel('TIFF32 uses reversible linear levels for image editors; tiff32_raw keeps original units. Blank black/white boxes use display levels.')
        self.save_status.setWordWrap(True)
        body.addWidget(self.save_status)
        self.encoding.currentIndexChanged.connect(self._export_options)
        split.addWidget(right)
        split.setSizes([380, 780])
        layout.addWidget(split, 1)
        self.progress = QProgressBar()
        layout.addWidget(self.progress)
        self.run_device = QLabel('No processing run started')
        self.run_device.setWordWrap(True)
        layout.addWidget(self.run_device)
        self.status = QLabel('idle')
        self.error = QLabel('')
        self.error.setWordWrap(True)
        self.error.setTextFormat(Qt.PlainText)
        self.error.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard)
        self.error.setToolTip('Select error text with the mouse or keyboard and copy it with Ctrl+C.')
        self.error.setStyleSheet('color: #bb3333; font-weight: bold')
        layout.addWidget(self.status)
        layout.addWidget(self.error)
        self.window.setCentralWidget(root)
        tips = {
            self.open_btn: 'Choose a SER, AVI or observed HDF5 capture, then inspect its metadata and input preview.',
            self.inspect_btn: 'Read capture metadata and preview the input using the current settings, without starting reconstruction.',
            self.run_btn: 'Start reconstruction with the current settings, or continue the selected checkpoint when Resume is enabled.',
            self.cancel_btn: 'Request cancellation of processing. The last received result remains available for viewing and saving.',
            self.checkpoint_btn: 'Choose the NPZ file used to save accumulator progress after each batch. To resume, choose an existing file and enable Resume.',
            self.checkpoint_path: 'Optional accumulator checkpoint path. Progress is saved here during processing; leave blank to disable checkpoints.',
            self.resume_check: 'Continue from the selected accumulator checkpoint. The capture and processing settings must match the saved run.',
            self.view: 'Choose the reconstructed result, input preview, accumulation weights, validity mask or Saturn layer coverage.',
            self.channel: 'Display all colour channels together, or inspect only red, green or blue. This does not change the exported channels.',
            self.zoom: 'Fit the preview to the window or choose a display zoom. This does not resize the saved full-resolution result.',
            self.black: 'Intensity mapped to black in the display. Also used for integer export when its black box is empty.',
            self.white: 'Intensity mapped to white in the display. Also used for integer export when its white box is empty.',
            self.encoding: 'TIFF32 scales black/white to 0/1 without clipping, for GIMP and other editors. TIFF32_raw preserves original camera units and may display overexposed. Integer TIFF/PNG maps levels into 16-bit values.',
            self.save_black: 'Intensity mapped to zero in TIFF32 or integer PNG/TIFF. Leave blank to use display black. Ignored for TIFF32_raw.',
            self.save_white: 'Intensity mapped to 1 in TIFF32 or 65535 in integer PNG/TIFF. Leave blank to use display white. Ignored for TIFF32_raw.',
            self.save_gamma: 'Optional display gamma for integer export. Leave blank for linear output. Ignored for float TIFF.',
            self.save_btn: 'Save the latest full-resolution result in the selected format, with provenance and coverage companions. Partial results are labelled incomplete.',
            self.cancel_save_btn: 'Request cancellation of the active export. Cancellation can wait for the current encoding operation to finish.',
        }
        for widget, tip in tips.items():
            widget.setToolTip(tip)
        self.timer = QTimer(self.window)
        self.timer.setInterval(200)
        self.timer.timeout.connect(self._poll)
        self._export_options()
        self._buttons()
        if self.path is not None:
            self.controls.suggest_capture_planet(self.path)

    def show(self):
        self.window.show()

    def _buttons(self):
        busy = self.job is not None
        self.open_btn.setEnabled(not busy and not self.closing)
        self.inspect_btn.setEnabled(not busy and self.path is not None and not self.closing)
        self.preprocess_btn.setEnabled(not busy and self.path is not None and not self.closing)
        self.run_btn.setEnabled(not busy and self.path is not None and not self.closing)
        self.controls.setEnabled(not busy and not self.closing)
        for control in (self.checkpoint_path, self.checkpoint_btn, self.resume_check):
            control.setEnabled(not busy and not self.closing)
        self.cancel_btn.setEnabled(busy and self.cancel_started is None)
        self.save_btn.setEnabled(self.last_result is not None and self.export_worker is None and not self.closing)
        self.cancel_save_btn.setEnabled(self.export_worker is not None)

    def _choose(self):
        name, _ = QFileDialog.getOpenFileName(self.window, 'Open capture', '', 'Captures (*.ser *.avi *.h5 *.hdf5)')
        if name:
            self.controls.clear_geometry_estimate()
            self.preprocessing_info = {}
            self._refresh_preprocessing()
            self.path = Path(name)
            self.checkpoint_path.clear()
            self.resume_check.setChecked(False)
            self.source_label.setText(str(self.path))
            self._inspect()

    def _inspect(self):
        self._start(inspect_only=True)

    def _preprocess(self):
        self._start(inspect_only=False, preprocess_only=True)

    def _choose_checkpoint(self):
        name, _ = QFileDialog.getSaveFileName(self.window, 'Accumulator checkpoint',
                                             self.checkpoint_path.text(), 'NumPy state (*.npz)')
        if name:
            self.checkpoint_path.setText(name)

    def _run(self):
        self._start(inspect_only=False)

    def _start(self, inspect_only, preprocess_only=False):
        if self.job is not None or self.path is None or self.closing:
            return
        try:
            if not self.resume_check.isChecked():
                self.controls.suggest_capture_planet(self.path)
            cfg = self.controls.configuration()
            checkpoint_options = {}
            if not inspect_only and not preprocess_only:
                missing = cfg.missing_motion_parameters()
                missing.pop('sub_obs_lat_rad', None)  # Worker resolves automatic SER viewing geometry.
                if missing:
                    edit = self.controls.fields[next(iter(missing))]
                    for tab in range(self.controls.count()):
                        if self.controls.widget(tab).isAncestorOf(edit):
                            self.controls.setCurrentIndex(tab)
                            break
                    edit.setFocus()
                    edit.selectAll()
                    cfg.require_motion_parameters(self.preprocessing_info, allow_auto_latitude=True)
                path = self.checkpoint_path.text().strip()
                if self.resume_check.isChecked() and not path:
                    raise ValueError('Select an accumulator checkpoint to resume')
                if path:
                    checkpoint_options['state_checkpoint'] = Path(path)
                    if self.resume_check.isChecked():
                        checkpoint_options['resume_from'] = Path(path)
            if preprocess_only:
                handle = start_stack_job(self.path, cfg, preprocess_only=True)
            else:
                handle = (start_stack_job(self.path, cfg, inspect_only=True) if inspect_only
                          else start_stack_job(self.path, cfg, **checkpoint_options))
        except (ValueError, TypeError, OSError) as exc:
            self.error.setText(str(exc))
            return
        self.config = cfg
        self.job = handle
        handle.snapshot_request.set()
        self.inspecting = inspect_only
        self.cancel_started = None
        self.started = time.monotonic()
        self.last_seq = 0
        self.auto_levels = not inspect_only
        self.error.clear()
        self.warnings.clear()
        self.run_device.setText('Input inspection; no reconstruction backend selected' if inspect_only
                                else f'Run requested {cfg.device.upper()} · preparing input; backend pending')
        self.details.setPlainText(json.dumps({'current_run_config': cfg.to_dict(),
                                             'inspect_only': inspect_only}, indent=2))
        if self.last_result is not None and not self.result_label.text().startswith('Previous result'):
            self.result_label.setText('Previous result (retained until this run produces an image):\n'
                                      + self.result_label.text())
        self.progress.setRange(0, 0)
        self.status.setText(f'Opening {self.path.name} on {cfg.device}')
        self.timer.start()
        self._buttons()

    def _cancel(self):
        if self.job is not None and self.cancel_started is None:
            self.cancel_started = time.monotonic()
            self.job.cancel_event.set()
            self.status.setText('Cancelling processing; last received result remains available.')
            self._buttons()

    def _set_input(self, payload):
        self.input_image = payload['input_image']
        meta = payload['source_metadata']
        self.input_metadata = meta
        exposure = meta.get('extras', {}).get('exposure_s', {}).get('value')
        self.controls.fields['exposure_s'].setPlaceholderText(
            f'Automatic: {exposure:.6g} s' if exposure is not None else 'No recorded exposure')

        if 'preprocessing_cache' in payload:
            self._set_preprocessing(payload['preprocessing_cache'])
        self.input_max = payload.get('input_max')
        timing = payload.get('capture_timing', meta.get('extras', {}).get('capture_timing', {}).get('value', {}))
        duration = (f"{timing['duration_s']:.6g} s ({timing.get('origin', 'measured')})"
                    if timing.get('status') == 'available' else 'unavailable')
        self.source_label.setText(f"{meta['path']} · {meta['width']}×{meta['height']} · "
                                  f"{meta['n_frames']} frames · {meta['color_mode']} · "
                                  f"{meta['bit_depth']} bit · duration: {duration} · "
                                  f"recorded exposure: {str(exposure) + ' s' if exposure is not None else 'unavailable'} · "
                                  f"input view: {payload['input_view']}")
        if self.inspecting:
            self.view.setCurrentText('Input')
            self.details.setPlainText(json.dumps(meta, indent=2))
            self._fit_levels()
        self._draw()

    def _refresh_preprocessing(self, checked=None):
        info = self.preprocessing_info
        if info.get('status') == 'ready':
            usage = 'Will use cache' if self.controls.fields['frame_preselection'].isChecked() else 'Cache disabled for runs'
            percent = self.controls.fields['stack_percent'].value()
            mode = self.controls.fields['frame_selection_mode'].currentData()
            if percent == 100:
                selection_text = f" Next run: all {info['accepted']} screened frames."
            elif mode == 'frame_count':
                selected = (info['accepted'] * percent + 99) // 100
                selection_text = f' Next run: best {percent}% by count = {selected} screened frames.'
            elif 'quality_range' in info:
                quality = info['quality_range']
                selected = quality['retained_counts'][percent - 1]
                low, high = quality['minimum'], quality['maximum']
                cutoff = None if low is None else low + (high - low) * (1 - percent / 100.)
                threshold = (f'quality > {cutoff:.6g}' if low is not None and low != high
                             else 'flat or unavailable quality range; no additional exclusions')
                fraction = 100 * selected / max(1, info['n_total'])
                selection_text = (f' Next run: upper {percent}% of quality range ({threshold}): '
                                  f'{selected} frames ({fraction:.1f}% of capture).')
            else:
                selection_text = ' Inspect input or Preprocess to refresh quality-range counts.'
            if self.controls.fields['frame_preselection'].isChecked():
                selection_text += ' Before registration rejection.'
            else:
                selection_text = ''
            self.preprocessing_label.setText(
                f"{usage}: {info['quality']} quality / {info['shape']} shape exclusions "
                f"({info['quality_shape_overlap']} overlap), {info['other']} other; "
                f"{info['excluded']} excluded total, {info['accepted']}/{info['n_total']} retained. "
                f"Best reference frame (from 0): {info.get('best_reference_index', 'unavailable')}.{selection_text}")
        else:
            reason = info.get('reason', 'No preprocessing cache. Run Preprocess; runs without a cache use no quality/shape filtering.')
            if (self.controls.fields['frame_preselection'].isChecked()
                    and self.controls.fields['stack_percent'].value() < 100):
                reason += ' Best-frame percentage selection needs a matching cache; run Preprocess first.'
            self.preprocessing_label.setText(reason)

    def _preprocessing_settings_changed(self, value=None):
        if self.preprocessing_info.get('status') == 'ready':
            self.preprocessing_info = {'status': 'unverified',
                'reason': 'Input/calibration settings changed. Inspect input to validate the cache, or Preprocess again.'}
            self.controls.clear_geometry_estimate()
            self._refresh_preprocessing()

    def _set_preprocessing(self, info):
        if info.get('status') != 'disabled' or self.preprocessing_info.get('status') != 'ready':
            self.preprocessing_info = info
        self._refresh_preprocessing()
        if info.get('status') == 'ready':
            self.controls.prefill_geometry(info.get('geometry_estimate', {}),
                                          allow_prefill=not self.checkpoint_path.text().strip())

    def _update_run_device(self, payload):
        backend = payload.get('backend')
        if backend:
            report = payload.get('device_report') or {}
            reason = report.get('reason', '')
            text = f'Run requested {self.config.device.upper()} · using {backend.upper()}'
            if reason and reason not in ('ok', 'explicit_cpu'):
                text += ' · ' + reason
            self.run_device.setText(text)
        if 'warnings' in payload:
            self.warnings.setText('; '.join(payload['warnings']))

    def _accept_result(self, payload):
        result = result_from_payload(payload)
        if 'preprocessing_cache' in result.provenance:
            self._set_preprocessing(result.provenance['preprocessing_cache'])
        estimate = (None if 'preprocessing_cache' in result.provenance else
                    result.provenance.get('preprocessing', {}).get('geometry_estimate'))
        if estimate is not None:
            self.controls.prefill_geometry(estimate, allow_prefill=not self.checkpoint_path.text().strip())
        self._update_run_device({'backend': result.backend, 'warnings': result.warnings,
                                 'device_report': result.provenance.get('device_report')})
        if result.n_used < 1 or result.spatial_stride != 1 or not np.any(result.validity):
            return
        # Received arrays are independent of the process accumulator. Export retains
        # this particular object even as a later snapshot replaces the displayed one.
        for arr in (result.image, result.coverage, result.validity, *result.layer_coverage.values()):
            arr.flags.writeable = False
        self.last_result = result
        self.preview = display_result_preview(result)
        if self.auto_levels:
            self.auto_levels = False
            self.view.setCurrentText('Result')
            self._fit_levels()
        source = result.provenance.get('source', {}).get('path', 'unknown source')
        epoch = result.reference_epoch if result.reference_epoch is not None else 'not specified'
        self.result_label.setText(f"{'Intermediate' if result.incomplete else 'Final'} {result.stage} · "
            f"{result.image.shape[1]}×{result.image.shape[0]} · {result.n_used} used / {result.n_rejected} rejected · "
            f"{result.units} · epoch {epoch} · {result.backend}/{result.precision}\nResult source: {source}")
        self.details.setPlainText(json.dumps(result.metadata(), indent=2))
        self.warnings.setText('; '.join(result.warnings[:2]) +
                              (' (more in metadata)' if len(result.warnings) > 2 else ''))
        self._draw()
        self._buttons()

    def _poll(self):
        if self.job is None:
            return
        handle = self.job
        if self.cancel_started is not None:
            elapsed = time.monotonic() - self.cancel_started
            if elapsed > 3 and handle.process.is_alive():
                handle.process.kill()
            elif elapsed > 2 and handle.process.is_alive():
                handle.process.terminate()
        for event in handle.poll():
            if event.job_id != handle.job_id or event.seq <= self.last_seq:
                continue
            self.last_seq = event.seq
            if event.kind == 'source':
                self._set_input(event.payload)
            elif event.kind == 'preprocessing_cache':
                self._set_preprocessing(event.payload)
            elif event.kind == 'geometry_estimate' and self.cancel_started is None:
                self.controls.prefill_geometry(event.payload, allow_prefill=not self.checkpoint_path.text().strip())
            elif event.kind == 'snapshot':
                self._accept_result(event.payload)
                handle.snapshot_request.set()
            elif event.kind == 'progress' and self.cancel_started is None:
                self._update_run_device(event.payload)
                frac = event.payload.get('fraction')
                self.progress.setRange(0, 0 if frac is None else 100)
                if frac is not None:
                    self.progress.setValue(min(99, max(0, round(100*frac))))
                elapsed = time.monotonic() - self.started
                eta = f' · ~{elapsed*(1-frac)/frac:.0f}s remaining' if frac and 0 < frac < 1 else ''
                self.status.setText(f"{event.payload.get('stage', '')} · {event.payload.get('backend', '')} "
                    f"· {event.payload.get('n_used', 0)} used · {elapsed:.1f}s elapsed{eta}")
            elif event.kind in ('completed', 'cancelled', 'error'):
                if event.kind == 'completed':
                    if 'source_metadata' in event.payload:
                        self._set_input(event.payload)
                        self.status.setText('Input inspected. Configure settings and run.')
                    elif 'preprocessing_cache' in event.payload:
                        self._set_preprocessing(event.payload['preprocessing_cache'])
                        self.status.setText('Preprocessing cached. Choose whether to use it, then run with any processing settings.')
                    else:
                        self._accept_result(event.payload)
                        self.status.setText('Processing complete; scientific result is ready to save.')
                    self.progress.setRange(0, 100)
                    self.progress.setValue(100)
                elif event.kind == 'error':
                    self.error.setText('Processing failed: ' + event.payload.get('message', 'unknown error'))
                    self.status.setText('Last received result remains inspectable and saveable.')
                else:
                    self.status.setText('Processing cancelled. Last received result remains available.')
                self._finish_job()
                break

    def _finish_job(self):
        self.timer.stop()
        if self.job is not None:
            self.job.close()
            self.job = None
        if self.progress.maximum() == 0:
            self.progress.setRange(0, 100)
            self.progress.setValue(0)
        self.cancel_started = None
        self._buttons()

    def _view_data(self):
        mode = self.view.currentText()
        if mode == 'Input':
            return self.input_image, None
        if self.preview is None:
            return None, None
        if mode == 'Result':
            return self.preview.image, self.preview.validity
        if mode == 'Validity':
            return self.preview.validity.astype(float), None
        if mode == 'Coverage':
            return self.preview.coverage, None
        return self.preview.layer_coverage.get('globe' if mode == 'Globe coverage' else 'ring'), None

    def _fit_levels(self):
        arr, mask = self._view_data()
        if arr is None:
            return
        valid = np.isfinite(arr)
        if mask is not None:
            valid &= mask
        values = np.asarray(arr)[valid]
        low = float(np.percentile(values, 1)) if values.size else 0.
        # The preview may omit the brightest pixel. Fit white against the full
        # scientific view, retaining common levels across RGB channels.
        mode = self.view.currentText()
        if mode in ('Coverage', 'Globe coverage', 'Ring coverage', 'Validity'):
            # Zero means no support. Uniform positive coverage stays visible.
            low = 0.
        source = self.input_metadata if mode == 'Input' else {}
        gain = 1.
        maximum = float(values.max()) if values.size else 0.
        if mode == 'Input':
            if self.input_max is not None:maximum = max(maximum, self.input_max)
        elif self.last_result is not None:
            result = self.last_result
            full_mask = None
            if mode == 'Result':
                full, full_mask = result.image, result.validity
                source = result.provenance.get('source', {})
                if result.units == 'e-':
                    gain = result.provenance.get('config', {}).get('gain_e_per_adu') or 1.
            elif mode == 'Coverage':full = result.coverage
            elif mode == 'Validity':full = result.validity
            else:full = result.layer_coverage.get('globe' if mode == 'Globe coverage' else 'ring')
            if full is not None:
                supported = np.isfinite(full)
                if full_mask is not None:supported &= full_mask
                maximum = max(maximum, float(np.max(full, where=supported, initial=0)))
        # Floating radiance/coverage has no nominal detector ceiling. Validity
        # is bounded by one; raw ADU and gain-scaled ADU use the capture range.
        full_scale = 1. if mode == 'Validity' else float('inf')
        bits = source.get('bit_depth')
        if source.get('units') == 'adu' and isinstance(bits, int) and 0 < bits <= 32:
            full_scale = ((1 << bits)-1) * gain
        high = min(1.43 * maximum, full_scale) if maximum > 0 else min(1., full_scale)
        if high <= low:
            low = high - 1.
        self.black.blockSignals(True)
        self.white.blockSignals(True)
        self.black.setValue(float(low))
        self.white.setValue(float(high))
        self.black.blockSignals(False)
        self.white.blockSignals(False)
        self._draw()

    def _draw(self, *_args):
        if not hasattr(self, 'histogram'):
            return
        arr, mask = self._view_data()
        if arr is None:
            self.image_label.setText('This view is not available for the current source/result.')
            return
        if arr.ndim == 3 and self.channel.currentIndex():
            c = self.channel.currentIndex()-1
            arr = arr[..., c]
            mask = mask[..., c] if mask is not None and mask.ndim == 3 else mask
        lo, hi = self.black.value(), self.white.value()
        q = _to_qimage(arr, lo, hi, mask)
        pixmap = QPixmap.fromImage(q)
        if self.zoom.currentText() == 'Fit':
            size = self.canvas.viewport().size()
            pixmap = pixmap.scaled(size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.FastTransformation)
        else:
            scale = float(self.zoom.currentText().rstrip('%')) / 100
            pixmap = pixmap.scaled(max(1, int(q.width()*scale)), max(1, int(q.height()*scale)))
        self.image_label.setPixmap(pixmap)
        self.image_label.resize(pixmap.size())
        valid = np.isfinite(arr)
        if mask is not None:
            valid &= mask
        samples = arr[valid]
        clipped = int(np.count_nonzero((samples < lo) | (samples > hi)))
        bins = np.histogram(samples, bins=16, range=(lo, hi if hi > lo else lo+1))[0] if samples.size else np.zeros(16)
        heights = np.rint(bins / max(1, bins.max()) * 7).astype(int)
        bars = ''.join('▁▂▃▄▅▆▇█'[n] for n in heights)
        sampling = ('Nearest-neighbour Bayer display; scientific masks are unchanged.\n'
                    if self.view.currentText() != 'Input' and self.preview is not None
                    and 'display_sampling' in self.preview.provenance else '')
        self.histogram.setText(sampling + f'Preview histogram {bars} · valid {valid.mean():.1%} · '
            f'{clipped} display-clipped samples · magenta = missing support.\n'
            'Coverage shows accumulation weights, not calibrated uncertainty. Zoom refers to preview pixels.')

    def _export_options(self):
        encoding = self.encoding.currentText()
        for edit in (self.save_black, self.save_white):
            edit.setEnabled(encoding != 'tiff32_raw')
        self.save_gamma.setEnabled(encoding in ('png16', 'tiff16'))

    def _parse_save_number(self, text, name, fallback=None):
        text = text.strip().replace('\u2212', '-')
        if not text:
            return fallback
        try:
            return float(text)
        except ValueError:
            try:
                return float(text.replace(',', '.'))
            except ValueError:
                raise ValueError(f'{name} {text!r} is not a number') from None

    def _export_config(self):
        encoding = self.encoding.currentText()
        if encoding == 'tiff32_raw':
            return ExportConfig('tiff32_raw')
        return ExportConfig(encoding,
            self._parse_save_number(self.save_black.text(), 'Export black', self.black.value()),
            self._parse_save_number(self.save_white.text(), 'Export white', self.white.value()),
            self._parse_save_number(self.save_gamma.text(), 'Display gamma') if encoding != 'tiff32' else None)

    def _choose_save(self):
        if self.last_result is None or self.export_worker is not None:
            return
        try:
            cfg = self._export_config()
        except ValueError as exc:
            self.save_status.setText(f'Save settings: {exc}')
            return
        extension = '.png' if cfg.encoding == 'png16' else '.tif'
        path, _ = QFileDialog.getSaveFileName(self.window, 'Save scientific result',
                    'intermediate'+extension if self.last_result.incomplete else 'result'+extension,
                    'PNG (*.png)' if extension == '.png' else 'TIFF (*.tif *.tiff)',
                    options=QFileDialog.Option.DontConfirmOverwrite)
        if not path:
            return
        dest = Path(path)
        if not dest.suffix:
            dest = dest.with_suffix(extension)
        overwrite = False
        if dest.exists() or dest.is_symlink():
            overwrite = QMessageBox.question(self.window, 'Replace image?', f'Replace {dest}?',
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes
            if not overwrite:
                return
        self.save_result(dest, cfg, overwrite=overwrite)

    def save_result(self, path, config=ExportConfig(), *, overwrite=False):
        if self.last_result is None or self.export_worker is not None or self.closing:
            return False
        self.export_worker = ExportWorker(self.last_result, Path(path), config, overwrite)
        self.export_worker.outcome.connect(self._save_outcome)
        self.export_worker.finished.connect(self._save_finished)
        self.save_status.setText(f"Saving {'intermediate' if self.last_result.incomplete else 'final'} "
                                 f"{config.encoding} from {self.last_result.n_used} frames…")
        self.export_worker.start()
        self._buttons()
        return True

    def _cancel_save(self):
        if self.export_worker is not None:
            self.export_worker.cancel_event.set()
            self.save_status.setText('Cancelling save before publication…')

    def _save_outcome(self, report, error):
        if error is not None:
            self.save_status.setText('Save cancelled.' if isinstance(error, ExportCancelled) else f'Save failed: {error}')
        else:
            counts = report.metadata['counts']
            self.save_status.setText(f"Saved {report.path} · {counts['clipped_pixels']} clipped pixels · "
                f"{counts['invalid_pixels']} invalid pixels · {'incomplete' if report.metadata['result']['incomplete'] else 'final'}\n"
                f"Metadata: {report.sidecar}")

    def _save_finished(self):
        worker = self.export_worker
        if worker is not None:
            worker.wait()
            worker.deleteLater()
            self.export_worker = None
        self._buttons()
        if self.closing:
            self.window.close()

    def _shutdown(self, *_args):
        self._finish_job()
        self._cancel_save()


def main(argv=None):
    apply_thread_limits()
    args = list(sys.argv[1:] if argv is None else argv)
    app = create_app(args)
    win = MainWindow(Path(args[0]) if args else None)
    app.aboutToQuit.connect(win._shutdown)
    win.show()
    code = app.exec()
    # An external application quit also owns any remaining encoder thread.
    win._shutdown()
    if win.export_worker is not None:
        win.export_worker.wait()
    return code
