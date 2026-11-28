"""Correcting north/south ambiguity updates the actual next-run geometry."""
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from test_w14 import gui


def test_flip_preserves_manual_direction_through_cache_refresh(gui):
    _, window = gui
    controls = window.controls
    estimate = {'suggestions': {'pole_pa_rad': np.deg2rad(12.5)}}
    controls.prefill_geometry(estimate)
    controls.flip_pole_button.click()
    assert controls.flip_pole_button.isChecked()
    assert 'Pole flipped' in controls.flip_pole_button.text()
    assert controls.configuration().pole_pa_rad == pytest.approx(np.deg2rad(192.5))
    assert 'pole_pa_rad' in controls.geometry_manual
    controls.prefill_geometry(estimate)
    controls.clear_geometry_estimate()
    assert controls.flip_pole_button.isChecked()
    assert controls.configuration().pole_pa_rad == pytest.approx(np.deg2rad(192.5))
    controls.flip_pole_button.click()
    assert not controls.flip_pole_button.isChecked()
    assert 'Pole flip undone' in controls.fields['geometry_mode'].toolTip()
    assert controls.configuration().pole_pa_rad == pytest.approx(np.deg2rad(12.5))
    assert controls.flip_pole_button.toolTip()


@pytest.mark.parametrize('text', ['', 'invalid', 'nan', 'inf'])
def test_invalid_pole_does_not_crash_or_become_manual_override(gui, text):
    _, window = gui
    controls = window.controls
    controls.fields['pole_pa_rad'].setText(text)
    controls.flip_pole_button.click()
    assert not controls.flip_pole_button.isChecked()
    assert controls.fields['pole_pa_rad'].text() == text
    assert 'pole_pa_rad' not in controls.geometry_manual
    assert 'finite pole position angle' in controls.fields['geometry_mode'].toolTip()


def test_flip_indicator_survives_settings_restore_and_clears_on_angle_edit(gui, tmp_path):
    from planetrecon.gui.app import MainWindow
    _, window = gui
    window.settings_path = tmp_path / 'settings.json'
    window.controls.flip_pole_button.click()
    window._save_settings()
    restored = MainWindow(settings_path=window.settings_path)
    try:
        assert restored.controls.flip_pole_button.isChecked()
        assert 'Pole flipped' in restored.controls.flip_pole_button.text()
        assert restored.controls.configuration().pole_pa_rad == pytest.approx(np.pi)
        restored.controls.fields['pole_pa_rad'].setText('25')
        assert not restored.controls.flip_pole_button.isChecked()
        restored.controls.flip_pole_button.click()
        assert restored.controls.flip_pole_button.isChecked()
        assert restored.controls.configuration().pole_pa_rad == pytest.approx(np.deg2rad(205))
    finally:
        restored.window.close()


def test_new_surface_runs_use_flipped_pole_without_changing_rate(gui, tmp_path, monkeypatch):
    _, window = gui
    configs = []
    def start(path, cfg, **kwargs):
        configs.append(cfg)
        return SimpleNamespace(snapshot_request=threading.Event(), close=lambda: None)
    monkeypatch.setattr('planetrecon.gui.app.start_stack_job', start)
    window.path = tmp_path/'input.ser'
    fields = window.controls.fields
    fields['geometry_mode'].setCurrentIndex(fields['geometry_mode'].findData('surface'))
    for key, value in {'pole_pa_rad': 10., 'sub_obs_lat_rad': 0., 'surface_rate_rad_s': 1.,
                       'field_center_x': 64., 'field_center_y': 64., 'equatorial_radius_px': 42.}.items():
        fields[key].setText(str(value))
    for angle in (10., 190., 10.):
        window._run()
        assert configs[-1].pole_pa_rad == pytest.approx(np.deg2rad(angle))
        assert configs[-1].effective_surface_rate() == pytest.approx(np.deg2rad(1.))
        window._finish_job()
        window.controls.flip_pole_button.click()
    # At equator-on viewing, reversing the pole reverses the projected motion
    # of a central surface point while keeping the same physical rotation rate.
    from planetrecon.geometry.pose import FramePose, globe_for_config
    from planetrecon.geometry.model import OblateGlobeModel
    deltas = []
    for cfg in configs[:2]:
        globe = globe_for_config(42., 0., cfg.pole_pa_rad, 0., 0., cfg.effective_surface_rate(), 0.)
        x, y, valid = OblateGlobeModel(globe).src_to_ref(
            np.array([64.]), np.array([64.]), FramePose(1., 0., 64., 64.), FramePose(0., 0., 64., 64.))
        assert valid.all()
        deltas.append([x[0]-64., y[0]-64.])
    np.testing.assert_allclose(deltas[0], -np.array(deltas[1]), atol=1e-12)


def test_saved_manual_angle_can_return_to_current_measurement(gui):
    _, window = gui
    controls = window.controls
    state = controls.settings_state()
    state['fields']['pole_pa_rad'] = '173.571333103'
    state['geometry_manual'] = ['pole_pa_rad']
    controls.restore_settings(state)
    estimate = {'suggestions': {'pole_pa_rad': np.deg2rad(179.73363931235963)}}
    controls.prefill_geometry(estimate)
    assert controls.fields['pole_pa_rad'].text() == '173.571333103'
    assert controls.calculated_values_button.text() == 'Use calculated values'
    assert controls.calculated_values_button.isEnabled()
    controls.calculated_values_button.click()
    assert controls.configuration().pole_pa_rad == pytest.approx(estimate['suggestions']['pole_pa_rad'])
    assert 'pole_pa_rad' not in controls.geometry_manual
    assert controls.geometry_auto['pole_pa_rad'] == controls.fields['pole_pa_rad'].text()
    # Saving and restoring now retains automatic intent, so later measurements apply.
    controls.restore_settings(controls.settings_state())
    controls.prefill_geometry({'suggestions': {'pole_pa_rad': np.deg2rad(179.8)}})
    assert controls.configuration().pole_pa_rad == pytest.approx(np.deg2rad(179.8))


@pytest.mark.parametrize('invalidate', ['new_capture', 'unresolved', 'inapplicable', 'checkpoint', 'nonfinite'])
def test_calculated_values_action_cannot_reuse_unavailable_estimate(gui, invalidate):
    _, window = gui
    controls = window.controls
    controls.prefill_geometry({'suggestions': {'pole_pa_rad': .1}})
    controls.flip_pole_button.click()
    before = controls.configuration().pole_pa_rad
    if invalidate == 'new_capture':
        controls.clear_geometry_estimate()
    else:
        controls.prefill_geometry({
            'suggestions': {} if invalidate == 'unresolved' else
                          {'pole_pa_rad': float('nan') if invalidate == 'nonfinite' else .2},
            'applicable': invalidate != 'inapplicable',
        }, allow_prefill=invalidate != 'checkpoint')
    assert not controls.calculated_values_button.isEnabled()
    controls.calculated_values_button.click()
    assert controls.configuration().pole_pa_rad == before
    assert controls.pole_flipped


def test_calculated_values_clear_flip_and_preserve_unmeasured_overrides(gui):
    _, window = gui
    controls = window.controls
    controls.prefill_geometry({'suggestions': {'pole_pa_rad': .1}})
    controls.fields['field_rate_rad_s'].setText('0.5')
    controls.fields['field_rate_rad_s'].textEdited.emit('0.5')
    controls.flip_pole_button.click()
    controls.calculated_values_button.click()
    assert controls.configuration().pole_pa_rad == pytest.approx(.1)
    assert not controls.pole_flipped
    assert controls.fields['field_rate_rad_s'].text() == '0.5'
    assert controls.geometry_manual == {'field_rate_rad_s'}


def _preprocessed(pole_deg):
    """The worker's completed-preprocessing payload (jobs._worker_run) with a pole estimate."""
    from planetrecon.io.source import ArraySource
    from planetrecon.jobs import input_frame_preview
    from test_midpoint_epoch import ready
    info = ready(60.)
    info['geometry_estimate'] = {'suggestions': {'pole_pa_rad': np.deg2rad(pole_deg)},
                                 'orientation_origin': 'silhouette (major axis assumed equatorial)'}
    source = ArraySource(np.ones((2, 8, 8)), bit_depth=16, timestamps=np.arange(2.))
    return {**input_frame_preview(source), 'source_metadata': source.metadata().as_dict(),
            'capture_timing': info['timing'], 'preprocessing_cache': info, 'preprocessing_completed': True}


def test_preprocessing_started_by_stack_keeps_the_pole_flip(gui, tmp_path, monkeypatch):
    """Stack preprocesses first when geometry needs it; that must not undo a manual flip."""
    from planetrecon.jobs import JobEvent
    _, window = gui
    configs = []
    def start(path, cfg, **kwargs):
        configs.append((cfg, kwargs))
        return SimpleNamespace(snapshot_request=threading.Event(), close=lambda: None, poll=lambda: [])
    monkeypatch.setattr('planetrecon.gui.app.start_stack_job', start)
    window.path = tmp_path/'input.ser'
    fields = window.controls.fields
    fields['geometry_mode'].setCurrentIndex(fields['geometry_mode'].findData('surface'))
    for key, value in {'sub_obs_lat_rad': 0., 'surface_rate_rad_s': 1., 'field_center_x': 4.,
                       'field_center_y': 4., 'equatorial_radius_px': 3.}.items():
        fields[key].setText(str(value))
    window.controls.prefill_geometry({'suggestions': {'pole_pa_rad': np.deg2rad(52.89)}})
    window.controls.flip_pole_button.click()
    # The Stack sequence's preprocessing stage completes with a fresh (unflipped) estimate.
    window.run_stage = 'preprocess'
    window.job = SimpleNamespace(close=lambda: None)
    window._complete_event(JobEvent('job', 1, 'completed', _preprocessed(52.85)))
    assert not window.error.text(), window.error.text()
    assert window.controls.pole_flipped
    assert window.controls.configuration().pole_pa_rad == pytest.approx(np.deg2rad(232.89))
    stack, options = configs[-1]
    assert not options.get('preprocess_only') and not options.get('inspect_only')
    assert stack.pole_pa_rad == pytest.approx(np.deg2rad(232.89))
    # Preprocess on request still restores the calculated geometry.
    window._finish_job()
    window.job = SimpleNamespace(close=lambda: None)
    window._complete_event(JobEvent('job', 2, 'completed', _preprocessed(52.85)))
    assert not window.controls.pole_flipped
    assert window.controls.configuration().pole_pa_rad == pytest.approx(np.deg2rad(52.85))


def oblate_capture(n=24):
    """A textured, tilted oblate disc: preprocessing measures a silhouette pole axis."""
    from planetrecon.io.source import ArraySource
    y, x = np.indices((96, 112), dtype=float)
    a, b, tilt = 34., 31., np.deg2rad(30.)
    u = (x+.5-56)*np.cos(tilt) - (48-y-.5)*np.sin(tilt)
    v = (x+.5-56)*np.sin(tilt) + (48-y-.5)*np.cos(tilt)
    disc = ((u/a)**2 + (v/b)**2 < 1) * (500 + 50*np.cos(u/3))
    rng = np.random.default_rng(3)
    frames = np.stack([disc + rng.normal(0, 2, disc.shape) for _ in range(n)])
    return ArraySource(frames, bit_depth=16, timestamps=np.arange(float(n)))


def test_a_flipped_pole_keeps_the_cached_geometry_estimate():
    """The pole only chooses north/south: no pole makes the cached estimate stale."""
    from dataclasses import replace
    from planetrecon.geometry.discovery import discover_geometry
    from planetrecon.pipeline.preprocess import screen_source
    from planetrecon.pipeline.preprocess_cache import GEOMETRY_KEYS, cache_report
    from planetrecon.reconstruction import ReconstructionConfig
    source = oblate_capture()
    base = ReconstructionConfig(device='cpu', threads=2, geometry_mode='surface', rotation_planet='jupiter')
    selection = screen_source(source, base)
    selection.summary['geometry_estimate'] = discover_geometry(source, base, selection)
    selection.summary['geometry_analysis_config'] = {key: getattr(base, key) for key in GEOMETRY_KEYS}
    pole = selection.summary['geometry_estimate']['suggestions']['pole_pa_rad']
    for angle in (pole, pole + np.pi, pole - np.pi, pole + .3, 0.):
        estimate = cache_report(selection, config=replace(base, pole_pa_rad=angle))['geometry_estimate']
        assert estimate.get('applicable', True), angle


def test_cleared_analysis_inputs_are_suggested_again_after_reload():
    """A capture reopened with cleared Geometry keeps the estimate its analysis assumed."""
    from dataclasses import replace
    from planetrecon.geometry.discovery import discover_geometry
    from planetrecon.pipeline.preprocess import screen_source
    from planetrecon.pipeline.preprocess_cache import GEOMETRY_KEYS, cache_report
    from planetrecon.reconstruction import ReconstructionConfig
    source = oblate_capture()
    # Analysed with a prefilled radius, field rate and flipped pole (as Stack's preprocessing is).
    analysed = ReconstructionConfig(device='cpu', threads=2, geometry_mode='combined', rotation_planet='jupiter',
                                    equatorial_radius_px=33., field_rate_rad_s=1e-4, pole_pa_rad=np.deg2rad(232.))
    selection = screen_source(source, analysed)
    selection.summary['geometry_estimate'] = discover_geometry(source, analysed, selection)
    selection.summary['geometry_analysis_config'] = {key: getattr(analysed, key) for key in GEOMETRY_KEYS}
    assert 'equatorial_radius_px' not in selection.summary['geometry_estimate']['suggestions']
    cleared = replace(analysed, equatorial_radius_px=None, field_rate_rad_s=None, pole_pa_rad=0.)
    estimate = cache_report(selection, config=cleared)['geometry_estimate']
    assert estimate.get('applicable', True)
    assert estimate['suggestions']['equatorial_radius_px'] == 33.
    assert estimate['suggestions']['field_rate_rad_s'] == pytest.approx(1e-4)
    # A different explicit value is a real change: the estimate assumed another radius.
    estimate = cache_report(selection, config=replace(cleared, equatorial_radius_px=40.))['geometry_estimate']
    assert estimate['applicable'] is False


@pytest.mark.parametrize('configured', [0., 180., 250., -100.])
def test_silhouette_pole_keeps_the_configured_branch(configured):
    """The silhouette gives an axis, not a direction: re-estimating keeps a chosen (flipped) pole."""
    from planetrecon.geometry.discovery import discover_geometry
    from planetrecon.pipeline.preprocess import screen_source
    from planetrecon.reconstruction import ReconstructionConfig
    source = oblate_capture()
    config = ReconstructionConfig(device='cpu', threads=2, pole_pa_rad=np.deg2rad(configured))
    selection = screen_source(source, config)
    report = discover_geometry(source, config, selection)
    assert report['orientation_origin'].startswith('silhouette')
    pole = report['suggestions']['pole_pa_rad']
    axis = (pole - report['silhouette_axis_rad']) % np.pi
    assert min(axis, np.pi - axis) < 1e-9
    assert np.cos(pole - config.pole_pa_rad) >= 0
