"""Optional latitude uses validated geocentric data, never an invented ring tilt."""
from dataclasses import replace
from datetime import datetime
import json
import math
from pathlib import Path

import numpy as np
import pytest

from planetrecon.geometry import viewing
from planetrecon.io.ser import SERSource, write_ser
from planetrecon.reconstruction import ReconstructionConfig

DATE = datetime(2023, 10, 10, 13, 20, 53, 223000)
TICKS = (DATE - datetime(1, 1, 1)).days * 864000000000 + (13*3600+20*60+53)*10000000 + 2230000
JD = 1721425.5 + TICKS/864000000000


def payload(latitude=12.606481, body=699):
    # Minimal factual fields matching Horizons' observer-table output format.
    return {'result': f'''Target body name: Saturn ({body}) {{source:sat441l}}
Center body name: Earth (399) {{source:DE441}}
Center-site name: GEOCENTRIC
Target radii    : 60268.0, 60268.0, 54364.0 km {{Equator_a, b, pole_c}}
Date__(UT)__HR:MN:SC.fff, , , ObsSub-LON, ObsSub-LAT,
$$SOE
 2023-Oct-10 13:20:53.223, , , 342.182526, {latitude},
$$EOE
'''}


def ser(tmp_path, frames=None, **kwargs):
    if frames is None:
        frames = np.ones((3, 24, 24), dtype='u2')
    return SERSource(write_ser(tmp_path/'capture.ser', frames,
                     timestamps=np.array([TICKS-10000000, TICKS, TICKS+10000000]), **kwargs))


def test_ser_midpoint_prefers_trailer_and_accepts_duplicate_times(tmp_path):
    with ser(tmp_path, datetime_utc=TICKS-864000000000) as source:
        assert viewing.capture_epoch_jd(source) == JD
    path = write_ser(tmp_path/'duplicates.ser', np.ones((4, 8, 8), dtype='u2'),
                     timestamps=np.array([TICKS, TICKS, TICKS+20000000, TICKS+20000000]))
    with SERSource(path) as source:
        assert viewing.capture_epoch_jd(source) == pytest.approx(JD+1/86400, abs=1e-9)
    path = write_ser(tmp_path/'header.ser', np.ones((3, 8, 8), dtype='u2'), datetime_utc=TICKS)
    with SERSource(path) as source:
        assert viewing.capture_epoch_jd(source, cadence_s=1.) == pytest.approx(JD+1/86400, abs=1e-9)
    path = write_ser(tmp_path/'undated.ser', np.ones((3, 8, 8), dtype='u2'), timestamps=np.arange(3)*10000000)
    with SERSource(path) as source, pytest.raises(ValueError, match='UTC capture date'):
        viewing.capture_epoch_jd(source)


def test_planetodetic_conversion_preserves_sign_and_uses_reference_ellipsoid():
    for lat in (-90., -12.606481, 0., 12.606481, 90.):
        result = viewing.parse_response(payload(lat), 'saturn', JD)
        expected = math.atan2((54364/60268)**2 * math.sin(math.radians(lat)), math.cos(math.radians(lat)))
        assert result['sub_obs_lat_rad'] == expected
        # Intersection of the viewing ray with an ellipsoid has a surface
        # normal whose latitude recovers Horizons' planetodetic coordinate.
        restored = math.atan2(math.sin(expected)/(54364/60268)**2, math.cos(expected))
        assert math.degrees(restored) == pytest.approx(lat)


@pytest.mark.parametrize('old,new', [
    ('Saturn (699)', 'Saturn (599)'), ('Earth (399)', 'Earth (499)'),
    ('GEOCENTRIC', 'BODYCENTRIC'), ('2023-Oct-10', '2023-Oct-11'),
    ('12.606481', 'n.a.'), ('12.606481', 'nan'), ('12.606481', '91'),
    ('60268.0, 60268.0, 54364.0', '60268.0, 60268.0, 70000.0'),
    ('$$EOE', ''), ('$$SOE', ''), ('Date__(UT)', 'Date__(TT)'),
])
def test_malformed_or_mismatched_ephemeris_is_refused(old, new):
    bad = {'result': payload()['result'].replace(old, new)}
    with pytest.raises(ValueError):
        viewing.parse_response(bad, 'saturn', JD)


def test_cache_works_offline_and_target_or_epoch_change_needs_new_lookup(tmp_path, monkeypatch):
    calls = []
    def fetch(planet, jd):
        calls.append((planet, jd))
        return payload(body=viewing.BODY_IDS[planet])
    monkeypatch.setattr(viewing, 'fetch_response', fetch)
    with ser(tmp_path) as source:
        first = viewing.viewing_geometry(source, 'saturn')
        assert viewing.viewing_geometry(source, 'saturn') == first
        assert len(calls) == 1
        viewing.viewing_geometry(source, 'jupiter')
        assert len(calls) == 2
        cache = Path(str(source.path)+'.planetrecon-viewing.json')
        saved = json.loads(cache.read_text())
        saved['key']['epoch_jd_utc'] += 1
        cache.write_text(json.dumps(saved))
        viewing.viewing_geometry(source, 'jupiter')
        assert len(calls) == 3
        monkeypatch.setattr(viewing, 'fetch_response', lambda *a: pytest.fail('cached lookup must work offline'))
        viewing.viewing_geometry(source, 'jupiter')


def test_manual_override_avoids_network_and_missing_auto_view_refuses_saturn(tmp_path, monkeypatch):
    cfg = ReconstructionConfig(geometry_mode='saturn', sub_obs_lat_rad=0.)
    monkeypatch.setattr(viewing, 'fetch_response', lambda *a: pytest.fail('manual geometry needs no network'))
    with ser(tmp_path) as source:
        assert viewing.resolve_viewing(source, cfg) == (cfg, None)
        def offline(*a):
            raise OSError('offline')
        monkeypatch.setattr(viewing, 'fetch_response', offline)
        with pytest.raises(ValueError, match='Automatic planetary viewing latitude unavailable'):
            viewing.resolve_viewing(source, replace(cfg, sub_obs_lat_rad=None))
        effective, report = viewing.resolve_viewing(source, replace(cfg, sub_obs_lat_rad=None), strict=False)
        assert effective.sub_obs_lat_rad is None and report['status'] == 'unavailable'


def test_discovery_keeps_auto_latitude_blank_and_gui_shows_source(tmp_path, monkeypatch):
    from planetrecon.geometry.discovery import discover_geometry
    from planetrecon.pipeline.preprocess import screen_source
    from planetrecon.gui.app import create_app
    from planetrecon.gui.controls import ConfigControls
    monkeypatch.setattr(viewing, 'fetch_response', lambda *a: payload())
    cfg = ReconstructionConfig(geometry_mode='saturn', rotation_planet='saturn')
    with ser(tmp_path) as source:
        report = discover_geometry(source, cfg, screen_source(source, cfg))
    assert 'sub_obs_lat_rad' not in report['suggestions']
    assert report['viewing_geometry']['origin'] == 'jpl_horizons_geocentric'
    assert any('Automatic Saturn viewing latitude' in note for note in report['notes'])
    app = create_app(['viewing-geometry'])
    controls = ConfigControls(cfg)
    try:
        controls.prefill_geometry(report)
        assert controls.configuration().sub_obs_lat_rad is None
        assert 'JPL Horizons' in controls.geometry_estimate_label.text()
    finally:
        controls.close()
        app.processEvents()


def test_auto_saturn_stack_matches_manual_view_and_resumes(tmp_path, monkeypatch):
    from test_ring_registration import config
    from planetrecon.geometry.globe import GlobeParams
    from planetrecon.geometry.pose import FramePose
    from planetrecon.geometry.rings import RingParams
    from planetrecon.geometry.saturn import render_saturn
    from planetrecon.pipeline.baseline import stack_source
    monkeypatch.setattr(viewing, 'fetch_response', lambda *a: payload())
    lat = viewing.parse_response(payload(), 'saturn', JD)['sub_obs_lat_rad']
    globe = GlobeParams(20, flattening=.1, sub_obs_lat_rad=lat, surface_rate_rad_s=.3)
    frames = np.array([render_saturn(112, 112, FramePose(t, 0., 56, 56), globe, RingParams(26, 42),
                        lambda lon, b: 1+.2*np.cos(5*lon)*np.cos(3*b))*1000 for t in range(3)], dtype='u2')
    automatic = replace(config(), sub_obs_lat_rad=None)
    with ser(tmp_path, frames) as source:
        expected = stack_source(source, replace(automatic, sub_obs_lat_rad=lat))
        actual = stack_source(source, automatic, state_checkpoint=tmp_path/'state.npz')
        with np.load(tmp_path/'state.npz', allow_pickle=False) as state:
            saved_cfg = ReconstructionConfig.from_dict(json.loads(str(state['metadata']))['identity']['config'])
        assert saved_cfg.sub_obs_lat_rad is None
        resumed = stack_source(source, saved_cfg, resume_from=tmp_path/'state.npz')
    assert actual.n_used == 3
    np.testing.assert_array_equal(actual.image, expected.image)
    np.testing.assert_array_equal(actual.coverage, expected.coverage)
    np.testing.assert_array_equal(actual.image, resumed.image)
    assert actual.provenance['geometry']['viewing_geometry']['sub_obs_lat_rad'] == lat


def test_gui_starts_worker_with_optional_latitude_blank(monkeypatch):
    import planetrecon.gui.app as gui
    from test_ring_registration import config
    app = gui.create_app(['auto-viewing-run'])
    win = gui.MainWindow(config=replace(config(), sub_obs_lat_rad=None))
    win.path = Path('saturn.ser')
    captured = []
    def start(path, cfg, **kwargs):
        captured.append(cfg)
        raise ValueError('worker reached')
    monkeypatch.setattr(gui, 'start_stack_job', start)
    try:
        win._run()
        assert len(captured) == 1 and captured[0].sub_obs_lat_rad is None
        assert 'worker reached' in win.error.text()
    finally:
        win._shutdown()
        win.window.close()
        app.processEvents()


def test_selecting_planet_clears_old_equator_on_prefill_but_preserves_manual_latitude():
    from planetrecon.gui.app import create_app
    from planetrecon.gui.controls import ConfigControls
    app = create_app(['viewing-preset-change'])
    controls = ConfigControls(ReconstructionConfig(geometry_mode='surface'))
    try:
        controls.prefill_geometry({'suggestions': {'sub_obs_lat_rad': 0., 'surface_rate_rad_s': .02,
                                                   'field_rate_rad_s': .003, 'flattening': .1}})
        selector = controls.fields['rotation_planet']
        selector.setCurrentIndex(selector.findData('jupiter'))
        cfg = controls.configuration()
        assert cfg.sub_obs_lat_rad is None and cfg.field_rate_rad_s is None
        assert cfg.surface_rate_rad_s is None and cfg.flattening == 0.
        latitude = controls.fields['sub_obs_lat_rad']
        latitude.setText('15')
        latitude.textEdited.emit('15')
        selector.setCurrentIndex(selector.findData('saturn'))
        assert controls.configuration().sub_obs_lat_rad == pytest.approx(math.radians(15))
    finally:
        controls.close()
        app.processEvents()


def test_preprocessing_view_is_invalidated_by_changed_planet_or_utc(tmp_path):
    from planetrecon.pipeline.preprocess import screen_source
    from planetrecon.pipeline.preprocess_cache import cache_report
    cfg = ReconstructionConfig(geometry_mode='surface', rotation_planet='saturn')
    with ser(tmp_path) as source:
        selection = screen_source(source, cfg)
        selection.summary['geometry_estimate'] = {
            'suggestions': {'surface_rate_rad_s': .1},
            'viewing_geometry': viewing.parse_response(payload(), 'saturn', JD)}
        assert cache_report(selection, config=cfg, source=source)['geometry_estimate'].get('applicable', True)
        changed = cache_report(selection, config=replace(cfg, rotation_planet='jupiter'), source=source)
        assert changed['geometry_estimate']['applicable'] is False
        selection.summary['geometry_estimate']['viewing_geometry']['epoch_jd_utc'] -= 1
        changed = cache_report(selection, config=cfg, source=source)
        assert changed['geometry_estimate']['applicable'] is False
        assert changed['accepted'] == int(selection.accepted.sum())
