"""Fresh preprocessing replaces overrides; cached calculations can be restored."""
import numpy as np
import pytest

from planetrecon.io.ser import write_ser
from test_w14 import gui, pump
from test_midpoint_epoch import ready


def report():
    info = ready(360.)
    info['geometry_estimate'] = {'suggestions': {
        'pole_pa_rad': np.deg2rad(179.734), 'field_center_x': 349.96,
        'field_center_y': 201.96, 'equatorial_radius_px': 96.94,
        'field_rate_rad_s': -.00001, 'surface_rate_rad_s': .00016,
        'flattening': .1043, 'ring_inner_radius_px': 147.3,
        'ring_outer_radius_px': 221.3}, 'saturn_geometry': {'status': 'estimated'}}
    return info


def edit(controls, key, value):
    controls.fields[key].setText(value)
    controls.fields[key].textEdited.emit(value)


def test_one_button_restores_all_calculated_values_and_disables_itself(gui):
    _, win = gui
    c = win.controls
    c.fields['geometry_mode'].setCurrentText('saturn')
    win._set_preprocessing(report(), replace_manual=True)
    calculated = dict(c.calculated_geometry)
    assert not hasattr(c, 'measured_pole_button')
    assert not c.calculated_values_button.isEnabled()
    for key in calculated:
        edit(c, key, '123')
        assert c.calculated_values_button.isEnabled(), key
    # Capture controls are outside this action.
    edit(c, 'sampling_multiplier', '7')
    edit(c, 'max_shift_px', '2')
    c.calculated_values_button.click()
    assert all(c.fields[key].text() == text for key, text in calculated.items())
    assert not c.geometry_manual.intersection(calculated)
    assert not c.calculated_values_button.isEnabled()
    assert c.fields['sampling_multiplier'].text() == '7'
    assert c.fields['max_shift_px'].text() == '2'
    assert c.configuration().reference_epoch_s == 180.


def test_fresh_preprocessing_replaces_saved_manual_values_and_pole_flip(gui):
    _, win = gui
    c = win.controls
    state = c.settings_state()
    state['fields'].update(pole_pa_rad='173.571333103', field_center_x='700', reference_epoch_s='0')
    state['geometry_manual'] = ['pole_pa_rad', 'field_center_x', 'reference_epoch_s']
    c.restore_settings(state)
    win._set_preprocessing(report())  # Loading a cache still preserves edits.
    assert c.fields['pole_pa_rad'].text() == '173.571333103'
    assert c.calculated_values_button.isEnabled()
    c.flip_pole_button.click()
    win._set_preprocessing(report(), replace_manual=True)
    assert c.configuration().pole_pa_rad == pytest.approx(np.deg2rad(179.734))
    assert c.configuration().reference_epoch_s == 180.
    assert c.configuration().field_center_x == 349.96
    assert not c.pole_flipped and not c.calculated_values_button.isEnabled()


def test_non_geometry_edits_do_not_enable_calculated_values(gui):
    _, win = gui
    win._set_preprocessing(report(), replace_manual=True)
    for key in ('gain_e_per_adu', 'sampling_multiplier', 'max_shift_px'):
        edit(win.controls, key, '5')
    assert not win.controls.calculated_values_button.isEnabled()


def test_planet_preset_clears_manual_surface_rate_on_fresh_preprocess(gui):
    _, win = gui
    c = win.controls
    c.fields['rotation_planet'].setCurrentText('Saturn')
    edit(c, 'surface_rate_rad_s', '0.5')
    win._set_preprocessing(report(), replace_manual=True)
    assert c.configuration().surface_rate_rad_s is None
    assert c.configuration().rotation_planet == 'saturn'
    assert 'surface_rate_rad_s' not in c.geometry_manual
    assert not c.calculated_values_button.isEnabled()


def test_successful_preprocess_worker_applies_calculations_in_desktop(gui, tmp_path):
    app, win = gui
    y, x = np.indices((48, 48))
    frame = (1000*np.exp(-((x-24)**2+(y-24)**2)/90)*(1+.1*np.cos(x))).astype('u2')
    win.path = write_ser(tmp_path/'input.ser', np.repeat(frame[None], 16, axis=0))
    win.controls.fields['cadence_s'].setText('0.1')
    edit(win.controls, 'field_center_x', '999')
    edit(win.controls, 'reference_epoch_s', '0')
    win._preprocess()
    pump(app, lambda: win.job is None, timeout=20)
    assert not win.error.text()
    assert win.controls.configuration().field_center_x == pytest.approx(24, abs=1)
    assert win.controls.configuration().reference_epoch_s == .75
    assert not win.controls.calculated_values_button.isEnabled()
    assert not win.controls.geometry_manual.intersection({'field_center_x', 'reference_epoch_s'})


def test_calculated_viewing_latitude_can_replace_later_manual_override(gui):
    _, win = gui
    info = report()
    info['geometry_estimate']['viewing_geometry'] = {
        'origin': 'jpl_horizons_geocentric', 'sub_obs_lat_rad': .077}
    win._set_preprocessing(info, replace_manual=True)
    edit(win.controls, 'sub_obs_lat_rad', '10')
    assert win.controls.calculated_values_button.isEnabled()
    win.controls.calculated_values_button.click()
    assert win.controls.configuration().sub_obs_lat_rad is None
    assert not win.controls.calculated_values_button.isEnabled()
