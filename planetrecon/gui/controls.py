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
        self.capture_planet_path = None
        self.planet_choice_manual = config.rotation_planet is not None
        self.setting_capture_planet = False
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
        capture.addRow(QLabel('CPU cap includes mapped libraries; excludes this UI.\nCUDA cap excludes driver/library memory.\nSurface, combined and Saturn motion support CUDA float64.'))
        self._choice(capture, 'crop', 'HDF5 crop', ['feature', 'bland'])
        self._choice(capture, 'bayer_override', 'Raw colour override', [None, 'mono', 'RGGB', 'GRBG', 'GBRG', 'BGGR'])
        self._choice(capture, 'endian_override', 'Byte order override', [None, 'little', 'big'])
        self._choice(capture, 'endian_convention', 'SER byte-order convention', ['ecosystem', 'spec'])
        self._check(capture, 'recover_complete_frames', 'Recover complete frames')
        self._check(capture, 'reject_saturated', 'Reject saturated frames')
        self._check(capture, 'frame_preselection', 'Use cached preprocessing (quality and shape)')
        self._check(capture, 'local_alignment', 'Local patch alignment (experimental)')
        self._integer(capture, 'local_patch_size', 'Alignment patch size (odd pixels)', 15, 255)
        self.fields['local_patch_size'].setSingleStep(2)
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
        self.geometry_estimate_text = 'Preprocessing can prefill geometry for the next run. User edits are preserved.'
        self._choice(geo, 'geometry_mode', 'Motion model', ['none', 'field', 'surface', 'combined', 'saturn'])
        self.motion_model_label = geo.labelForField(self.fields['geometry_mode'])
        from planetrecon.geometry.rotation import PERIOD_DAYS
        self._choice(geo, 'rotation_planet', 'Planet rotation preset', [None, *PERIOD_DAYS])
        self.fields['rotation_planet'].setItemText(0, 'Measured / manual rate')
        for index, planet in enumerate(PERIOD_DAYS, 1):
            self.fields['rotation_planet'].setItemText(index, planet.title())
        self.planet_identification_label = QLabel('A clear planet name in the capture filename selects its rotation preset. Manual choices are preserved.')
        self.planet_identification_label.setWordWrap(True)
        geo.addRow(self.planet_identification_label)
        self._check(geo, 'reverse_rotation', 'Reverse preset rotation direction')
        self.rotation_label = QLabel()
        self.rotation_label.setWordWrap(True)
        geo.addRow(self.rotation_label)
        for key, label in [
            ('reference_epoch_s', 'Output epoch (s from start)'),
            ('field_angle0_rad', 'Reference field angle (°)'),
            ('field_rate_rad_s', 'Field rate (°/s; blank = fit)'),
            ('surface_rate_rad_s', 'Surface rate override (°/s; blank = preset/estimate)'),
            ('field_center_x', 'Centre x (px)'), ('field_center_y', 'Centre y (px)'),
            ('equatorial_radius_px', 'Globe equatorial radius (px)'), ('flattening', 'Globe flattening'),
            ('pole_pa_rad', 'Pole position angle (°)'), ('sub_obs_lat_rad', 'Planet-facing latitude override (°; blank = auto)'),
            ('sub_obs_lon0_rad', 'Reference longitude (°)'),
        ]:
            self._number(geo, key, label, angular=key.endswith(('_rad', '_rad_s')))
            if key == 'pole_pa_rad':
                self.flip_pole_button = QPushButton('Flip pole 180° (north / south)')
                self.flip_pole_button.setToolTip(
                    'Add 180° to the pole position angle, wrapped to 0–360°, when '
                    'preprocessing identifies the opposite pole. Uses the existing '
                    'surface rate, including manual overrides. Applies to the next run '
                    'and is preserved as a manual geometry edit during preprocessing. '
                    'Click again to restore the original orientation.')
                self.flip_pole_button.clicked.connect(self._flip_pole)
                geo.addRow('', self.flip_pole_button)
        geo.addRow(QLabel('Centre anchors tracking; rates are rigid.\nSaturn moon tracks keep a fixed centre. Exposure uses its midpoint.'))
        sat = self._tab('Saturn')
        sat.addRow(QLabel('Preprocess measures globe/ring radii when resolved.\nViewing latitude comes from SER UTC or a Geometry override.'))
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
        self.fields['rotation_planet'].currentIndexChanged.connect(self._rotation_changed)
        self.fields['rotation_planet'].currentIndexChanged.connect(self._planet_choice_changed)
        self.fields['rotation_planet'].activated.connect(self._planet_choice_changed)
        self.fields['reverse_rotation'].toggled.connect(self._rotation_changed)
        for key in ('surface_rate_rad_s', 'equatorial_radius_px'):
            self.fields[key].textChanged.connect(self._rotation_changed)
        self._rotation_changed()
        self._set_geometry_help(self.geometry_estimate_text)
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

    def _planet_choice_changed(self, value=None):
        if not self.setting_capture_planet:
            self.planet_choice_manual = True
            self.planet_identification_label.setText('Manual planet selection; capture names will not replace it.')

    def suggest_capture_planet(self, path):
        """Apply a filename hint once per capture, preserving explicit choices."""
        from planetrecon.geometry.rotation import planet_from_capture_name
        path = str(path)
        if self.capture_planet_path == path or self.planet_choice_manual:
            return
        self.capture_planet_path = path
        planet = planet_from_capture_name(path)
        self.setting_capture_planet = True
        try:
            chooser = self.fields['rotation_planet']
            chooser.setCurrentIndex(chooser.findData(planet))
        finally:
            self.setting_capture_planet = False
        self.planet_identification_label.setText(
            f'{planet.title()} preset selected from the capture filename; review the planet and apparent rotation direction.'
            if planet is not None else
            'Capture filename does not identify one planet. Select a planet preset or supply a measured/manual rate for surface motion.')

    def _rotation_changed(self, value=None):
        from planetrecon.geometry.rotation import rotation_preset
        planet = self.fields['rotation_planet'].currentData()
        if planet is not None and 'sub_obs_lat_rad' in self.geometry_auto:
            from planetrecon.reconstruction import ReconstructionConfig
            defaults = ReconstructionConfig()
            # Previous image-only discovery assumed an equator-on view. Its
            # latitude and dependent prefills must not mask automatic geometry.
            for key in ('sub_obs_lat_rad', 'field_rate_rad_s', 'pole_pa_rad', 'flattening'):
                old = self.geometry_auto.get(key)
                if old is not None and key not in self.geometry_manual and self.fields[key].text() == old:
                    self.geometry_auto.pop(key)
                    default = getattr(defaults, key)
                    self.fields[key].setText('' if default is None else str(math.degrees(default) if key in self.angular else default))
            self.geometry_auto.pop('sub_obs_lat_rad', None)
        edit = self.fields['surface_rate_rad_s']
        old = self.geometry_auto.get('surface_rate_rad_s')
        if (planet is not None and old is not None and edit.text() == old
                and 'surface_rate_rad_s' not in self.geometry_manual):
            self.geometry_auto.pop('surface_rate_rad_s')
            edit.clear()
        self.fields['reverse_rotation'].setEnabled(planet is not None)
        if planet is None:
            self.rotation_label.setText('Use a measured or manually entered rate. A preset can supply rotation when bands do not reveal motion.')
            return
        try:
            radius = float(self.fields['equatorial_radius_px'].text()) if self.fields['equatorial_radius_px'].text().strip() else None
            detail = rotation_preset(planet, reverse=self.fields['reverse_rotation'].isChecked(), radius_px=radius)
            rate = math.degrees(detail['surface_rate_rad_s'])
            scale = (f' Equator-on motion scale: {detail["equator_on_speed_px_s"]:.5g} px/s.'
                     if radius is not None else ' Set or measure the globe radius for the pixel-motion scale.')
            override = ' The explicit rate below overrides this preset.' if edit.text().strip() else ''
            self.rotation_label.setText(f'{planet.title()}: {abs(detail["sidereal_period_days"])*24:.7g} h sidereal period; {rate:.7g} °/s.'
                + scale + override + ' Bulk rotation, not a texture measurement. Check pole orientation and apparent direction; cloud winds may differ.')
        except ValueError:
            self.rotation_label.setText('Enter a numeric globe radius to show the preset pixel-motion scale.')

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
            self._set_geometry_help('Saturn uses automatic viewing latitude from SER UTC, or a manual override; the assumed equator-on view and dependent motion prefills were cleared. User edits are preserved.')
        self.fields['local_alignment'].setEnabled(
            self.fields['frame_preselection'].isChecked()
            and self.fields['geometry_mode'].currentData() == 'none')
        self.fields['local_patch_size'].setEnabled(
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

    def wants_midpoint_epoch(self):
        key = 'reference_epoch_s'
        if key in self.geometry_manual:
            return False
        text = self.fields[key].text()
        try:
            value = float(text)
        except ValueError:
            return False
        return value == 0. or text == self.geometry_auto.get(key)

    def prefill_output_epoch(self, timing, *, allow_prefill=True):
        if not allow_prefill or not self.wants_midpoint_epoch():
            return
        duration = timing.get('duration_s')
        if (timing.get('status') != 'available' or not isinstance(duration, (int, float))
                or not math.isfinite(duration) or duration < 0):
            return
        text = format(duration / 2., '.12g')
        self.fields['reference_epoch_s'].setText(text)
        self.geometry_auto['reference_epoch_s'] = text

    def _flip_pole(self):
        edit = self.fields['pole_pa_rad']
        try:
            angle = float(edit.text())
            if not math.isfinite(angle):
                raise ValueError
        except ValueError:
            self._set_geometry_help('Enter a finite pole position angle before flipping it.')
            return
        self.geometry_manual.add('pole_pa_rad')
        self.geometry_auto.pop('pole_pa_rad', None)
        edit.setText(format((angle + 180.) % 360., '.12g'))
        self._set_geometry_help(
            'Pole direction flipped by 180°. This manual orientation applies to the next run '
            'and is preserved when preprocessing is repeated.')

    def clear_geometry_estimate(self):
        from planetrecon.reconstruction import ReconstructionConfig
        defaults = ReconstructionConfig()
        for key, text in self.geometry_auto.items():
            if key not in self.geometry_manual and self.fields[key].text() == text:
                value = getattr(defaults, key)
                self.fields[key].setText('' if value is None else str(math.degrees(value) if key in self.angular else value))
        self.geometry_auto.clear()
        self._set_geometry_help('Preprocessing can prefill geometry for the next run. User edits are preserved.')

    def _set_geometry_help(self, text):
        self.geometry_estimate_text = text
        tooltip = CONTROL_HELP['geometry_mode'] + '\n\n' + text
        self.fields['geometry_mode'].setToolTip(tooltip)
        self.motion_model_label.setToolTip(tooltip)

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
        if (self.fields['geometry_mode'].currentData() == 'saturn'
                or estimate.get('saturn_geometry', {}).get('status') == 'estimated'):
            allowed.update(('ring_inner_radius_px', 'ring_outer_radius_px'))
        if allow_prefill:
            for key in ('pole_pa_rad', 'surface_rate_rad_s', 'field_rate_rad_s', 'flattening',
                        'ring_inner_radius_px', 'ring_outer_radius_px'):
                old = self.geometry_auto.get(key)
                if (old is not None and key not in estimate.get('suggestions', {})
                        and key not in self.geometry_manual and self.fields[key].text() == old):
                    value = getattr(defaults, key)
                    self.fields[key].setText('' if value is None else str(math.degrees(value) if key in self.angular else value))
                    del self.geometry_auto[key]
        applied = []
        for key, value in estimate.get('suggestions', {}).items():
            if key == 'surface_rate_rad_s' and self.fields['rotation_planet'].currentData() is not None:
                continue
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
        if allow_prefill and self.fields['rotation_planet'].currentData() is not None:
            usage = ('The selected planet preset supplies rotation even if texture motion is unresolved; '
                     'an explicit surface rate overrides it. Geometry prefills apply to the next run.')
        self._set_geometry_help(
            f'Surface drift: {direction}; image roll: {roll}. {usage}\n'
            + 'Orientation: ' + estimate.get('orientation_origin', 'unresolved') + '. '
            + ' '.join(estimate.get('notes', [])))
        return applied
