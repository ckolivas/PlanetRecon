"""Signed spin/roll recovery and safe GUI prefilling from observed frames."""
from dataclasses import replace
import json
import os

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import numpy as np
import pytest
from scipy.ndimage import shift

from planetrecon.geometry.discovery import discover_geometry
from planetrecon.geometry.globe import GlobeParams, render_globe_texture
from planetrecon.io.source import ArraySource
from planetrecon.pipeline.preprocess import screen_source
from planetrecon.reconstruction import ReconstructionConfig


def texture(lon, lat):
    return 100 + 20*np.sin(15*lon+3*lat) + 12*np.cos(17*lat-8*lon) + 15*np.sin(25*lon-13*lat)


def recording(pa=.4, spin=.002, roll=0., latitude=0., timestamps=True, translation=False):
    globe = GlobeParams(40, pole_pa_rad=pa, surface_rate_rad_s=spin, sub_obs_lat_rad=latitude)
    times = np.linspace(0, 60, 36)
    frames = np.stack([render_globe_texture(112, 128, 64, 56, roll*t, globe, t, texture) for t in times])
    if translation:
        frames = np.stack([shift(frame, (t*.015, -t*.02), order=1, mode='constant') for frame, t in zip(frames, times)])
    return ArraySource(frames, timestamps=times if timestamps else None)


def estimate(source, **values):
    cfg = ReconstructionConfig(device='cpu', threads=2, equatorial_radius_px=40, **values)
    return discover_geometry(source, cfg, screen_source(source, cfg))


@pytest.mark.parametrize('pa', [0., .4, 1.2])
@pytest.mark.parametrize('spin', [-.002, .002])
def test_projected_spin_axis_and_sign(pa, spin):
    result = estimate(recording(pa=pa, spin=spin))
    suggestions = result['suggestions']
    assert result['surface_resolved']
    assert suggestions['surface_rate_rad_s'] == pytest.approx(spin, rel=.2)
    assert suggestions['pole_pa_rad'] == pytest.approx(pa, abs=.16)
    assert suggestions['sub_obs_lat_rad'] == 0.
    assert any('equator-on' in note for note in result['notes'])
    assert sum(map(len, result['sample_indices'])) <= 96
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('roll', [-.0015, .0015])
def test_camera_roll_is_not_surface_spin(roll):
    result = estimate(recording(spin=0, roll=roll), sub_obs_lat_rad=0.)
    assert result['roll_resolved']
    assert not result['surface_resolved']
    assert result['suggestions']['field_rate_rad_s'] == pytest.approx(roll, rel=.2)
    assert 'surface_rate_rad_s' not in result['suggestions']
    assert result['roll_direction'] == ('counter-clockwise' if roll > 0 else 'clockwise')


def test_tracking_translation_does_not_invent_rotation():
    result = estimate(recording(spin=0, translation=True))
    assert not result.get('surface_resolved', False)
    assert not result.get('roll_resolved', False)
    assert 'surface_rate_rad_s' not in result['suggestions']


def test_missing_times_preserves_direction_but_does_not_invent_seconds():
    result = estimate(recording(timestamps=False))
    assert result['surface_resolved']
    assert 'pole_pa_rad' in result['suggestions']
    assert 'surface_rate_rad_s' not in result['suggestions']
    assert 'field_rate_rad_s' not in result['suggestions']
    assert result['time_origin'] == 'inferred'


@pytest.mark.parametrize('latitude', [-.3, .3])
def test_known_latitude_separates_spin_from_line_of_sight_roll(latitude):
    result = estimate(recording(spin=.002, roll=.0015, latitude=latitude), sub_obs_lat_rad=latitude)
    assert result['surface_resolved'] and result['roll_resolved']
    assert result['suggestions']['surface_rate_rad_s'] == pytest.approx(.002, rel=.25)
    assert result['suggestions']['field_rate_rad_s'] == pytest.approx(.0015, rel=.2)
    assert 'sub_obs_lat_rad' not in result['suggestions']


def test_cancel_and_low_texture_are_unresolved():
    source = recording()
    cfg = ReconstructionConfig(device='cpu', equatorial_radius_px=40)
    selection = screen_source(source, cfg)
    with pytest.raises(InterruptedError):
        discover_geometry(source, cfg, selection, should_cancel=lambda: True)
    y, x = np.indices((112, 128))
    disc = (((x-64)**2+(y-56)**2) < 40**2).astype(float)*100
    source = ArraySource(np.stack([disc]*36), timestamps=np.arange(36.))
    result = estimate(source)
    assert 'pole_pa_rad' not in result['suggestions']
    assert 'surface_rate_rad_s' not in result['suggestions']


def test_gui_prefills_degrees_preserves_manual_values_and_resets_for_new_capture():
    pytest.importorskip('PySide6')
    from planetrecon.gui.app import create_app
    from planetrecon.gui.controls import ConfigControls
    app = create_app(['geometry-prefill-test'])
    controls = ConfigControls(ReconstructionConfig(field_center_x=20))
    controls.fields['pole_pa_rad'].setText('0')
    controls.fields['pole_pa_rad'].textEdited.emit('0')  # Explicit zero is user-owned.
    report = {'suggestions': {'pole_pa_rad': .4, 'surface_rate_rad_s': -.002,
                              'field_center_x': 50, 'field_center_y': 40},
              'surface_direction': 'right / down', 'notes': ['equator-on approximation']}
    applied = controls.prefill_geometry(report)
    assert set(applied) == {'surface_rate_rad_s', 'field_center_y'}
    cfg = controls.configuration()
    assert cfg.surface_rate_rad_s == pytest.approx(-.002)
    assert cfg.pole_pa_rad == 0 and cfg.field_center_x == 20
    controls.fields['surface_rate_rad_s'].setText('1.2')
    controls.fields['surface_rate_rad_s'].textEdited.emit('1.2')
    controls.prefill_geometry(report)
    controls.clear_geometry_estimate()
    assert controls.fields['surface_rate_rad_s'].text() == '1.2'
    assert controls.fields['field_center_y'].text() == ''
    assert controls.configuration().field_center_x == 20
    controls.prefill_geometry({'suggestions': {'field_rate_rad_s': .01}})
    assert controls.configuration().field_rate_rad_s == pytest.approx(.01)
    controls.prefill_geometry({'suggestions': {}})
    assert controls.configuration().field_rate_rad_s is None
    controls.deleteLater()
    app.processEvents()


def test_checkpoint_geometry_settings_are_not_changed_by_discovery():
    pytest.importorskip('PySide6')
    from planetrecon.gui.app import MainWindow, create_app
    from planetrecon.jobs import result_payload
    from planetrecon.result import ReconstructionResult
    app = create_app(['geometry-checkpoint-test'])
    cfg = ReconstructionConfig(device='cpu', threads=2)
    win = MainWindow(config=cfg)
    win.checkpoint_path.setText('existing-state.npz')
    result = ReconstructionResult(np.ones((12, 12)), np.ones((12, 12)), np.ones((12, 12), bool),
        'adu', 'mono', 'cpu', 'float64', 'final', False, n_used=3,
        provenance={'preprocessing': {'geometry_estimate': {'suggestions': {'pole_pa_rad': .5}}}})
    win._accept_result(result_payload(result))
    assert win.controls.configuration() == cfg
    assert 'Checkpoint settings retained' in win.controls.geometry_estimate_label.text()
    win._shutdown()
    win.window.close()
    app.processEvents()


def test_estimates_reach_worker_without_mutating_active_config(tmp_path):
    import time
    from planetrecon.io.ser import write_ser
    from planetrecon.jobs import start_stack_job, result_from_payload
    source = recording()
    frames = np.stack([source.read_raw(i) for i in range(source.n_frames())]).astype('u2')
    path = write_ser(tmp_path/'spin.ser', frames)
    cfg = ReconstructionConfig(device='cpu', threads=2, cadence_s=60/35, equatorial_radius_px=40)
    from planetrecon.io.ser import SERSource
    from planetrecon.pipeline.preprocess_cache import preprocess_source
    with SERSource(path) as capture:
        preprocess_source(capture, cfg)
    handle = start_stack_job(path, cfg)
    events = []
    try:
        deadline = time.monotonic()+15
        while handle.state not in ('completed', 'failed') and time.monotonic() < deadline:
            events.extend(handle.poll(.05))
        assert handle.state == 'completed'
        estimates = [e.payload['geometry_estimate'] for e in events if e.kind == 'preprocessing_cache']
        final = result_from_payload(next(e.payload for e in events if e.kind == 'completed'))
        assert estimates and 'pole_pa_rad' in estimates[0]['suggestions']
        assert final.provenance['config'] == cfg.to_dict()
        assert final.provenance['preprocessing']['geometry_estimate'] == estimates[0]
    finally:
        handle.close()
