"""Human-facing configuration controls; values remain in engine units on commit."""
from dataclasses import replace
import math

from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QFormLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QScrollArea, QSpinBox, QTabWidget, QWidget,
)


class ConfigControls(QTabWidget):
    def __init__(self, config):
        super().__init__()
        self.base = config
        self.fields = {}
        self.angular = set()
        capture = self._tab('Capture')
        self._choice(capture, 'device', 'Device', ['auto', 'cpu', 'gpu'])
        self._integer(capture, 'threads', 'CPU threads', 1, 32)
        self._integer(capture, 'batch_frames', 'Frames per batch', 1, 4096)
        capture.addRow(QLabel('RAM/VRAM budgets are not enforced.\nGeometry uses CPU float64.'))
        self._choice(capture, 'crop', 'HDF5 crop', ['feature', 'bland'])
        self._choice(capture, 'bayer_override', 'Raw colour override', [None, 'mono', 'RGGB', 'GRBG', 'GBRG', 'BGGR'])
        self._choice(capture, 'endian_override', 'Byte order override', [None, 'little', 'big'])
        self._choice(capture, 'endian_convention', 'SER byte-order convention', ['ecosystem', 'spec'])
        self._check(capture, 'recover_complete_frames', 'Recover complete frames')
        self._check(capture, 'reject_saturated', 'Reject saturated frames')
        self._integer(capture, 'reference_index', 'Reference frame (from 0)', 0, 2**31-1)
        self._number(capture, 'max_shift_px', 'Maximum shift (px)')
        self._number(capture, 'cadence_s', 'Cadence (s; blank = timestamps)')
        self._number(capture, 'exposure_s', 'Exposure (s)')
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
        self._choice(geo, 'geometry_mode', 'Motion model', ['none', 'field', 'surface', 'combined', 'saturn'])
        for key, label in [
            ('reference_epoch_s', 'Output epoch (s from start)'),
            ('field_angle0_rad', 'Reference field angle (°)'),
            ('field_rate_rad_s', 'Field rate (°/s; blank = fit)'),
            ('surface_rate_rad_s', 'Surface rate (°/s)'),
            ('field_center_x', 'Centre x (px)'), ('field_center_y', 'Centre y (px)'),
            ('equatorial_radius_px', 'Globe equatorial radius (px)'), ('flattening', 'Globe flattening'),
            ('pole_pa_rad', 'Pole position angle (°)'), ('sub_obs_lat_rad', 'Signed observer latitude (°)'),
            ('sub_obs_lon0_rad', 'Reference longitude (°)'),
        ]:
            self._number(geo, key, label, angular=key.endswith(('_rad', '_rad_s')))
        geo.addRow(QLabel('Fixed centre and rigid rates.\nExposure uses the midpoint approximation.'))
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
            if not saturn and key in saturn_defaults:
                values[key] = saturn_defaults[key]
                continue
            if isinstance(edit, QComboBox):
                value = edit.currentData()
            elif isinstance(edit, QSpinBox):
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
            values[key] = value
        if values['geometry_mode'] != 'saturn':
            for key in ('ring_inner_radius_px', 'ring_outer_radius_px', 'sun_lon_rad', 'sun_lat_rad',
                        'moon_x', 'moon_y', 'moon_radius_px'):
                values[key] = None
            values.update(ring_transmission=.35, moon_vx_px_s=0., moon_vy_px_s=0.)
        return replace(self.base, **values)
