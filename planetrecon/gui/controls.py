"""Human-facing configuration controls; values remain in engine units on commit."""
from dataclasses import replace
import math

from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QFormLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QScrollArea, QSpinBox, QTabWidget, QToolButton, QWidget,
)


from planetrecon.gui.help import CONTROL_HELP


class ConfigControls(QTabWidget):
    def __init__(self, config):
        super().__init__()
        self.base = config
        self.fields = {}
        self.angular = set()
        self.geometry_manual = set()
        self.geometry_auto = {}
        capture = self._tab('Capture')
        self._choice(capture, 'device', 'Device', ['auto', 'cpu', 'gpu'])
        self._integer(capture, 'threads', 'CPU threads', 1, 32)
        self._integer(capture, 'batch_frames', 'Frames per batch', 1, 4096)
        self._number(capture, 'max_ram_bytes', 'Linux CPU process cap (MiB; blank = default)')
        if config.max_ram_bytes is not None:
            self.fields['max_ram_bytes'].setText(str(config.max_ram_bytes / 1024**2))
        self._number(capture, 'max_vram_bytes', 'CUDA tensor budget (MiB; blank = default)')
        if config.max_vram_bytes is not None:
            self.fields['max_vram_bytes'].setText(str(config.max_vram_bytes / 1024**2))
        capture.addRow(QLabel('CPU cap includes mapped libraries; excludes this UI.\nCUDA cap excludes driver/library memory.\nGeometry uses CPU float64.'))
        self._choice(capture, 'crop', 'HDF5 crop', ['feature', 'bland'])
        self._choice(capture, 'bayer_override', 'Raw colour override', [None, 'mono', 'RGGB', 'GRBG', 'GBRG', 'BGGR'])
        self._choice(capture, 'endian_override', 'Byte order override', [None, 'little', 'big'])
        self._choice(capture, 'endian_convention', 'SER byte-order convention', ['ecosystem', 'spec'])
        self._check(capture, 'recover_complete_frames', 'Recover complete frames')
        self._check(capture, 'reject_saturated', 'Reject saturated frames')
        self._check(capture, 'frame_preselection', 'Use cached preprocessing (quality and shape)')
        self._check(capture, 'local_alignment', 'Local patch alignment (experimental)')
        self._check(capture, 'squared_quality_weights', 'Stronger quality weighting (experimental)')
        self.fields['local_alignment'].toggled.connect(self._mode_changed)
        self.fields['frame_preselection'].toggled.connect(self._mode_changed)
        self._choice(capture, 'frame_selection_mode', 'Optional frame selection',
                     ['quality_range', 'frame_count'])
        mode = self.fields['frame_selection_mode']
        mode.setItemText(0, 'Quality range')
        mode.setItemText(1, 'Frame count')
        self._integer(capture, 'stack_percent', 'Upper quality range (%)', 1, 100)
        self.stack_percent_label = capture.labelForField(self.fields['stack_percent'])
        mode.currentIndexChanged.connect(self._selection_mode_changed)
        self._selection_mode_changed()
        for key in ('stack_percent', 'frame_selection_mode'):
            self.fields[key].setEnabled(config.frame_preselection)
            self.fields['frame_preselection'].toggled.connect(self.fields[key].setEnabled)
        self._integer(capture, 'reference_index', 'Reference frame (0 = automatic)', 0, 2**31-1)
        self._number(capture, 'max_shift_px', 'Maximum shift (px)')
        self._number(capture, 'cadence_s', 'Cadence (s; blank = timestamps)')
        self._number(capture, 'exposure_s', 'Exposure (s; blank = recorded)')
        cal = self._tab('Calibration')
        label = QLabel('NPY tables must match the raw detector shape.\nDark is already exposure-scaled; flat is positive.\nCFA tables stay on the raw pixel lattice.')
        label.setWordWrap(True)
        cal.addRow(label)
        for name in ('bias', 'dark', 'flat'):
            row = QWidget()
            layout = QHBoxLayout(row)
            layout.setContentsMargins(0, 0, 0, 0)
            edit = QLineEdit(getattr(config, name+'_path') or '')
            edit.setPlaceholderText('No table')
            self.fields[name+'_path'] = edit
            button = QPushButton('…')
            button.setToolTip(f'Choose the {name} calibration NPY table for the next run. The array must match the raw detector frame shape.')
            button.setMaximumWidth(30)
            button.clicked.connect(lambda checked=False, e=edit: self._choose_table(e))
            layout.addWidget(edit)
            layout.addWidget(button)
            cal.addRow(name.title(), row)
        self._number(cal, 'gain_e_per_adu', 'Gain (e−/ADU)')
        self._number(cal, 'read_noise_e', 'Read noise (e−; metadata)')
        self._number(cal, 'saturate_adu', 'Saturation threshold (ADU)')
        cal.addRow(QLabel('Blank gain preserves ADU / approximate noise.'))
        geo = self._tab('Geometry')
        guidance = QLabel('Motion model None stacks any planet, including Saturn and its rings. '
                          'Saturn mode separates globe/ring motion and requires manual viewing geometry.')
        guidance.setWordWrap(True)
        geo.addRow(guidance)
        self.geometry_estimate_label = QLabel('Preprocessing can prefill geometry for the next run. User edits are preserved.')
        self.geometry_estimate_label.setWordWrap(True)
        geo.addRow(self.geometry_estimate_label)
        self._choice(geo, 'geometry_mode', 'Motion model', ['none', 'field', 'surface', 'combined', 'saturn'])
        for key, label in [
            ('reference_epoch_s', 'Output epoch (s from start)'),
            ('field_angle0_rad', 'Reference field angle (°)'),
            ('field_rate_rad_s', 'Field rate (°/s; blank = fit)'),
            ('surface_rate_rad_s', 'Surface rate (°/s; required for spin)'),
            ('field_center_x', 'Centre x (px)'), ('field_center_y', 'Centre y (px)'),
            ('equatorial_radius_px', 'Globe equatorial radius (px)'), ('flattening', 'Globe flattening'),
            ('pole_pa_rad', 'Pole position angle (°)'), ('sub_obs_lat_rad', 'Signed observer latitude (°)'),
            ('sub_obs_lon0_rad', 'Reference longitude (°)'),
        ]:
            self._number(geo, key, label, angular=key.endswith(('_rad', '_rad_s')))
        geo.addRow(QLabel('Centre anchors tracking; rates are rigid.\nSaturn moon tracks keep a fixed centre. Exposure uses its midpoint.'))
        sat = self._tab('Saturn')
        sat.addRow(QLabel('Saturn requires globe/ring radii and signed\nobserver latitude in the Geometry tab.'))
        for key, label in [
            ('ring_inner_radius_px', 'Inner ring radius (px)'), ('ring_outer_radius_px', 'Outer ring radius (px)'),
            ('ring_transmission', 'Ring transmission (0–1)'), ('sun_lon_rad', 'Sun longitude (°)'),
            ('sun_lat_rad', 'Sun latitude (°)'), ('moon_x', 'Moon x (px)'), ('moon_y', 'Moon y (px)'),
            ('moon_radius_px', 'Moon mask radius (px)'), ('moon_vx_px_s', 'Moon velocity x (px/s)'),
            ('moon_vy_px_s', 'Moon velocity y (px/s)'),
        ]:
            self._number(sat, key, label, angular=key.endswith(('_rad', '_rad_s')))
        self.saturn_page = self.widget(3)
        self.fields['geometry_mode'].currentIndexChanged.connect(self._mode_changed)
        self._mode_changed()
        for key, edit in self.fields.items():
            edit.setToolTip(CONTROL_HELP[key])
            if isinstance(edit, QLineEdit):
                edit.textEdited.connect(lambda text, key=key: self.geometry_manual.add(key))
        for index, text in enumerate((
                'Input interpretation, processing device, memory and frame handling.',
                'Optional detector calibration tables and intensity units.',
                'Field rotation and globe surface geometry.',
                'Ring, illumination and moon-mask settings for the Saturn motion model.')):
            self.setTabToolTip(index, text)
        for button in self.tabBar().findChildren(QToolButton):
            if button.objectName() == 'ScrollLeftButton':
                button.setToolTip('Scroll the settings tabs to the left.')
            elif button.objectName() == 'ScrollRightButton':
                button.setToolTip('Scroll the settings tabs to the right.')

    def _selection_mode_changed(self):
        self.stack_percent_label.setText(
            'Upper quality range (%)' if self.fields['frame_selection_mode'].currentData() == 'quality_range'
            else 'Best retained frames (%)')

    def _tab(self, name):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        body = QWidget()
        layout = QFormLayout(body)
        layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        scroll.setWidget(body)
        self.addTab(scroll, name)
        return layout

    def _choice(self, form, key, label, choices):
        edit = QComboBox()
        for value in choices:
            edit.addItem('Header / automatic' if value is None else value, value)
        edit.setCurrentIndex(choices.index(getattr(self.base, key)))
        self.fields[key] = edit
        form.addRow(label, edit)

    def _integer(self, form, key, label, low, high):
        edit = QSpinBox()
        edit.setRange(low, high)
        edit.setValue(getattr(self.base, key))
        self.fields[key] = edit
        form.addRow(label, edit)

    def _check(self, form, key, label):
        edit = QCheckBox(label)
        edit.setChecked(getattr(self.base, key))
        self.fields[key] = edit
        form.addRow(edit)

    def _number(self, form, key, label, angular=False):
        value = getattr(self.base, key)
        if angular:
            self.angular.add(key)
            if value is not None:
                value = math.degrees(value)
        edit = QLineEdit('' if value is None else repr(value))
        edit.setPlaceholderText('Optional')
        self.fields[key] = edit
        form.addRow(label, edit)

    def _mode_changed(self):
        if (self.fields['geometry_mode'].currentData() == 'saturn'
                and 'sub_obs_lat_rad' in self.geometry_auto):
            from planetrecon.geometry.discovery import SATURN_VIEW_DEPENDENT_KEYS
            from planetrecon.reconstruction import ReconstructionConfig
            defaults = ReconstructionConfig()
            for key in SATURN_VIEW_DEPENDENT_KEYS:
                old = self.geometry_auto.get(key)
                if old is not None and key not in self.geometry_manual and self.fields[key].text() == old:
                    value = getattr(defaults, key)
                    self.fields[key].setText('' if value is None else str(math.degrees(value)))
                    del self.geometry_auto[key]
            # A user-supplied replacement latitude is retained, but it must not
            # leave dependent auto rates from the old assumption attached.
            self.geometry_auto.pop('sub_obs_lat_rad', None)
            self.geometry_estimate_label.setText('Saturn needs a supplied signed viewing latitude; the assumed equator-on view and dependent motion prefills were cleared. User edits are preserved.')
        self.fields['local_alignment'].setEnabled(
            self.fields['frame_preselection'].isChecked()
            and self.fields['geometry_mode'].currentData() == 'none')
        self.fields['squared_quality_weights'].setEnabled(
            self.fields['local_alignment'].isChecked()
            and self.fields['frame_preselection'].isChecked()
            and self.fields['geometry_mode'].currentData() == 'none')
        self.saturn_page.setEnabled(self.fields['geometry_mode'].currentData() == 'saturn')

    @staticmethod
    def _choose_table(edit):
        path, _ = QFileDialog.getOpenFileName(edit, 'Calibration table', '', 'NumPy table (*.npy)')
        if path:
            edit.setText(path)

    def configuration(self):
        values = {}
        saturn = self.fields['geometry_mode'].currentData() == 'saturn'
        saturn_defaults = dict(ring_inner_radius_px=None, ring_outer_radius_px=None,
            sun_lon_rad=None, sun_lat_rad=None, moon_x=None, moon_y=None, moon_radius_px=None,
            ring_transmission=.35, moon_vx_px_s=0., moon_vy_px_s=0.)
        for key, edit in self.fields.items():
            if key == 'squared_quality_weights' and (
                    not self.fields['local_alignment'].isChecked()
                    or not self.fields['frame_preselection'].isChecked()
                    or self.fields['geometry_mode'].currentData() != 'none'):
                values[key] = False
                continue
            if key == 'local_alignment' and (
                    not self.fields['frame_preselection'].isChecked()
                    or self.fields['geometry_mode'].currentData() != 'none'):
                values[key] = False
                continue
            if not saturn and key in saturn_defaults:
                values[key] = saturn_defaults[key]
                continue
            if isinstance(edit, QComboBox):
                value = edit.currentData()
            elif isinstance(edit, QSpinBox):
                # Commit typed text even when Run is invoked before focus leaves
                # the editor (for example through a shortcut or automation).
                edit.interpretText()
                value = edit.value()
            elif isinstance(edit, QCheckBox):
                value = edit.isChecked()
            elif key.endswith('_path'):
                value = edit.text().strip() or None
            else:
                try:
                    value = float(edit.text()) if edit.text().strip() else None
                except ValueError:
                    raise ValueError(f'{key}: enter a number') from None
                if value is not None and key in self.angular:
                    value = math.radians(value)
                if key in ('max_ram_bytes', 'max_vram_bytes') and value is not None:
                    if not math.isfinite(value) or value <= 0:
                        raise ValueError('Memory budget must be positive')
                    value = int(value * 1024**2)
            values[key] = value
        if values['geometry_mode'] != 'saturn':
            for key in ('ring_inner_radius_px', 'ring_outer_radius_px', 'sun_lon_rad', 'sun_lat_rad',
                        'moon_x', 'moon_y', 'moon_radius_px'):
                values[key] = None
            values.update(ring_transmission=.35, moon_vx_px_s=0., moon_vy_px_s=0.)
        return replace(self.base, **values)

    def clear_geometry_estimate(self):
        from planetrecon.reconstruction import ReconstructionConfig
        defaults = ReconstructionConfig()
        for key, text in self.geometry_auto.items():
            if key not in self.geometry_manual and self.fields[key].text() == text:
                value = getattr(defaults, key)
                self.fields[key].setText('' if value is None else str(math.degrees(value) if key in self.angular else value))
        self.geometry_auto.clear()
        self.geometry_estimate_label.setText('Preprocessing can prefill geometry for the next run. User edits are preserved.')

    def prefill_geometry(self, estimate, *, allow_prefill=True):
        from planetrecon.reconstruction import ReconstructionConfig
        if self.fields['geometry_mode'].currentData() == 'saturn':
            from planetrecon.geometry.discovery import without_assumed_saturn_view
            estimate = without_assumed_saturn_view(estimate)
        defaults = ReconstructionConfig()
        applicable = estimate.get('applicable', True)
        allow_prefill = allow_prefill and applicable
        allowed = {'field_center_x', 'field_center_y', 'equatorial_radius_px',
                   'pole_pa_rad', 'sub_obs_lat_rad', 'surface_rate_rad_s', 'field_rate_rad_s', 'flattening'}
        if allow_prefill:
            for key in ('pole_pa_rad', 'surface_rate_rad_s', 'field_rate_rad_s', 'flattening'):
                old = self.geometry_auto.get(key)
                if (old is not None and key not in estimate.get('suggestions', {})
                        and key not in self.geometry_manual and self.fields[key].text() == old):
                    value = getattr(defaults, key)
                    self.fields[key].setText('' if value is None else str(math.degrees(value) if key in self.angular else value))
                    del self.geometry_auto[key]
        applied = []
        for key, value in estimate.get('suggestions', {}).items():
            if not allow_prefill or key not in allowed or key in self.geometry_manual or not math.isfinite(value):
                continue
            edit = self.fields[key]
            text = edit.text()
            try:
                current = float(text) if text.strip() else None
            except ValueError:
                continue
            if current is not None and key in self.angular:
                current = math.radians(current)
            if text != self.geometry_auto.get(key) and current != getattr(defaults, key):
                continue
            new_text = format(math.degrees(value) if key in self.angular else value, '.10g')
            edit.setText(new_text)
            self.geometry_auto[key] = new_text
            applied.append(key)
        direction = estimate.get('surface_direction', 'unresolved')
        roll = estimate.get('roll_direction', 'unresolved')
        usage = ('Geometry estimates need refreshing.' if not applicable else
                 'Prefills apply to the next run.' if allow_prefill else
                 'Checkpoint settings retained; estimates are available in result metadata.')
        self.geometry_estimate_label.setText(
            f'Surface drift: {direction}; image roll: {roll}. {usage}\n'
            + 'Orientation: ' + estimate.get('orientation_origin', 'unresolved') + '. '
            + ' '.join(estimate.get('notes', [])))
        return applied
