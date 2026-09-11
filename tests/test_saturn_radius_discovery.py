"""Resolved ring boundaries must reach the next run as detector-pixel geometry."""
from dataclasses import replace
import json

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from planetrecon.geometry.discovery import discover_geometry
from planetrecon.geometry.globe import GlobeParams, render_globe_texture
from planetrecon.geometry.pose import FramePose
from planetrecon.geometry.rings import RingParams
from planetrecon.geometry.saturn import render_saturn
from planetrecon.geometry.saturn_fit import fit_saturn_geometry
from planetrecon.io.source import ArraySource
from planetrecon.pipeline.preprocess import screen_source
from planetrecon.reconstruction import ReconstructionConfig


def scene(latitude=.18, angle=.4, scale=1., field=0.):
    globe = GlobeParams(35*scale, flattening=.1, sub_obs_lat_rad=latitude, pole_pa_rad=angle)
    image = render_saturn(round(192*scale), round(224*scale),
                          FramePose(0, field, 112*scale, 96*scale), globe,
                          RingParams(46*scale, 78*scale), lambda lon, lat: 1+.1*np.cos(6*lat))
    return gaussian_filter(image, .8)


@pytest.mark.parametrize('latitude,angle,scale', [(.18, 0, 1), (-.18, .7, 1),
    (.45, .4, 1), (.65, 1.1, 1), (.25, -.6, .65), (.25, 1.5, 1.5)])
def test_separate_radii_with_seeing_noise_rotation_and_sampling(latitude, angle, scale):
    image = scene(latitude, angle, scale)
    image += np.random.default_rng(17).normal(0, .001, image.shape)
    result = fit_saturn_geometry(image)
    assert result['ok'], result
    for key, truth in [('radius', 35), ('ring_inner', 46), ('ring_outer', 78)]:
        assert result[key] == pytest.approx(truth*scale, abs=1.7)
    assert result['cx'] == pytest.approx(112*scale, abs=1.)
    assert result['cy'] == pytest.approx(96*scale, abs=1.)
    assert np.sin(result['pa_rad']-angle) == pytest.approx(0., abs=.025)
    assert result['opening_rad'] == pytest.approx(abs(latitude), abs=.035)
    assert 'sub_obs_lat_rad' not in result


@pytest.mark.parametrize('kind', ['edge_on', 'clipped', 'globe_only', 'blank', 'invalid'])
def test_no_radii_invented_without_resolved_ring_edges(kind):
    if kind == 'edge_on':
        image = scene(latitude=0.)
    elif kind == 'clipped':
        image = scene()[:, 60:]
    elif kind == 'globe_only':
        image = render_globe_texture(192, 224, 112, 96, 0, GlobeParams(35), 0,
                                     lambda lon, lat: 1+.1*np.cos(6*lat))
    else:
        image = np.full((100, 100), 0. if kind == 'blank' else np.nan)
    result = fit_saturn_geometry(image)
    assert not result['ok']
    assert result['ring_inner'] is None and result['ring_outer'] is None


def recording(colour='mono', rate=0.):
    times = np.linspace(0, 180, 36)
    frames = np.stack([scene(field=rate*t) for t in times])*1000
    if colour == 'RGGB':
        frames[:, ::2, ::2] *= 1.2
        frames[:, 1::2, 1::2] *= .7
    elif colour == 'RGB':
        frames = frames[..., None]*np.array([1.2, 1., .7])
    source = ArraySource(frames, color_mode=colour, timestamps=times)
    config = ReconstructionConfig(geometry_mode='saturn', rotation_planet='saturn',
                                  sub_obs_lat_rad=.18, reference_epoch_s=90., threads=2)
    return source, config


@pytest.mark.parametrize('colour', ['mono', 'RGGB', 'RGB'])
@pytest.mark.parametrize('rate', [0., .0004, -.0004])
def test_preprocess_measures_radii_and_field_rate_in_detector_units(colour, rate):
    source, config = recording(colour, rate)
    report = discover_geometry(source, config, screen_source(source, config))
    assert report['saturn_geometry']['status'] == 'estimated', report
    values = report['suggestions']
    assert values['equatorial_radius_px'] == pytest.approx(35., abs=1.5)
    assert values['ring_inner_radius_px'] == pytest.approx(46., abs=2.)
    assert values['ring_outer_radius_px'] == pytest.approx(78., abs=1.5)
    assert values['field_rate_rad_s'] == pytest.approx(rate, abs=3e-5)
    assert values['pole_pa_rad'] == pytest.approx(.4-rate*90, abs=.015)
    assert 'surface_rate_rad_s' not in values
    assert 'sub_obs_lat_rad' not in values
    config = replace(config, **values)
    config.require_motion_parameters()
    json.dumps(report, allow_nan=False)


def test_cache_gui_prefill_manual_values_and_actual_saturn_run(tmp_path):
    from planetrecon.gui.app import create_app
    from planetrecon.gui.controls import ConfigControls
    from planetrecon.pipeline.preprocess_cache import preprocess_source, cache_report, load_cache
    from planetrecon.pipeline.baseline import stack_source
    source, config = recording('RGGB')
    selection = preprocess_source(source, config, cache_path=tmp_path/'cache.npz')
    report = cache_report(selection, config=config, source=source)['geometry_estimate']
    app = create_app(['saturn-radius-test'])
    controls = ConfigControls(config)
    try:
        controls.prefill_geometry(report, allow_prefill=False)
        assert controls.configuration().ring_inner_radius_px is None
        controls.prefill_geometry(report)
        next_config = controls.configuration()
        assert next_config.ring_inner_radius_px == pytest.approx(46., abs=2.)
        assert next_config.flattening > 0.
        restored, cache = load_cache(source, next_config, path=tmp_path/'cache.npz')
        assert restored is not None
        assert cache['geometry_estimate'].get('applicable', True)
        assert 'flattening' in cache['geometry_estimate']['suggestions']
        repeated = discover_geometry(source, next_config, restored)
        controls.prefill_geometry(repeated)
        assert controls.configuration().ring_outer_radius_px == next_config.ring_outer_radius_px
        assert controls.configuration().field_rate_rad_s is not None
        result = stack_source(source, next_config, preprocessing=restored)
        assert result.n_used > 0 and result.image.shape[-1] == 3
        assert np.isfinite(result.image).all()
        for key, value in [('ring_inner_radius_px', '48'), ('equatorial_radius_px', '34')]:
            controls.fields[key].setText(value)
            controls.fields[key].textEdited.emit(value)
        controls.prefill_geometry(report)
        assert controls.configuration().ring_inner_radius_px == 48
        controls.clear_geometry_estimate()
        assert controls.configuration().ring_inner_radius_px == 48
        assert controls.configuration().ring_outer_radius_px is None
    finally:
        controls.close()
        app.processEvents()


@pytest.mark.parametrize('mode', ['none', 'surface', 'combined', 'field'])
def test_saturn_target_detection_does_not_depend_on_stacking_mode(mode):
    source, config = recording('RGGB')
    config = replace(config, geometry_mode=mode, local_alignment=mode == 'none')
    report = discover_geometry(source, config, screen_source(source, config))
    assert report['saturn_geometry']['status'] == 'estimated'
    for key in ('equatorial_radius_px', 'ring_inner_radius_px', 'ring_outer_radius_px'):
        assert report['suggestions'][key] > 0
    assert config.geometry_mode == mode and config.local_alignment == (mode == 'none')


def test_gui_preprocess_none_then_switch_to_saturn_and_run(tmp_path, monkeypatch):
    import queue
    import threading
    from types import SimpleNamespace
    from planetrecon.gui.app import create_app, MainWindow
    from planetrecon.io.ser import write_ser, SERSource, NAME_TO_COLOR
    from planetrecon.jobs import _worker_run
    from planetrecon.pipeline.preprocess_cache import load_cache
    from planetrecon.pipeline.baseline import stack_source

    source, config = recording('RGGB')
    path = write_ser(tmp_path/'colour-Sat.ser', source._frames.astype('u2'),
                     color_id=NAME_TO_COLOR['RGGB'], timestamps=np.arange(36)*50000000)
    config = replace(config, geometry_mode='none', local_alignment=True, stack_percent=50)
    app = create_app(['saturn-mode-switch'])
    window = MainWindow(path, config)
    dispatched = []

    def start(path, cfg, **options):
        dispatched.append(cfg)
        events = queue.Queue()
        if options.get('preprocess_only'):
            _worker_run('saturn-gui', str(path), cfg.to_dict(), events, threading.Event(), None,
                        preprocess_only=True, auto_output_epoch=options['auto_output_epoch'])
        return SimpleNamespace(job_id='saturn-gui', snapshot_request=threading.Event(),
                               poll=lambda: list(events.queue), close=lambda: None)

    monkeypatch.setattr('planetrecon.gui.app.start_stack_job', start)
    try:
        window._preprocess()
        assert dispatched[-1].geometry_mode == 'none' and dispatched[-1].local_alignment
        window._poll()
        assert not window.error.text()
        assert window.preprocessing_info['geometry_estimate']['saturn_geometry']['status'] == 'estimated'
        # The inactive tab retains estimates without enabling rings in a None run.
        assert window.controls.configuration().ring_inner_radius_px is None
        window.controls.fields['geometry_mode'].setCurrentText('saturn')
        window._run()
        assert not window.error.text()
        actual = dispatched[-1]
        actual.require_motion_parameters()
        assert actual.geometry_mode == 'saturn' and not actual.local_alignment
        assert actual.ring_outer_radius_px == pytest.approx(78, abs=2)
        with SERSource(path) as capture:
            selection, report = load_cache(capture, actual)
            assert report['geometry_estimate'].get('applicable', True)
            assert selection.summary['geometry_analysis_mode'] == 'none'
            result = stack_source(capture, actual, preprocessing=selection)
            assert result.n_used > 0 and result.image.shape[-1] == 3
    finally:
        window._finish_job()
        window.window.close()
        app.processEvents()
