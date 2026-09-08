from dataclasses import replace
import numpy as np
import pytest

from planetrecon.geometry.shape import estimate_flattening
from planetrecon.geometry.globe import GlobeParams, render_globe_texture
from planetrecon.io.source import ArraySource
from planetrecon.pipeline.preprocess import screen_source, FrameSelection
from planetrecon.pipeline.preprocess_cache import preprocess_source, cache_report
from planetrecon.reconstruction import ReconstructionConfig


@pytest.mark.parametrize('latitude', [0., .3, -.6])
@pytest.mark.parametrize('flattening', [0., .07, .15])
def test_flattening_from_independently_rendered_oblate_outline(latitude, flattening):
    globe = GlobeParams(85., flattening=flattening, sub_obs_lat_rad=latitude, pole_pa_rad=.42)
    frame = render_globe_texture(224, 256, 128, 112, 0., globe, 0., lambda lon, lat: np.full_like(lon, 100.), limb_weight=False)
    cfg = ReconstructionConfig(device='cpu', threads=2, sub_obs_lat_rad=latitude)
    source = ArraySource(np.stack([frame]*16), bit_depth=32)
    report = estimate_flattening(screen_source(source, cfg), cfg)
    assert report['status'] == 'estimated'
    assert report['suggestions']['flattening'] == pytest.approx(flattening, abs=.018)


def selection(ratio):
    metrics = np.tile([1., 100., ratio*100., .2], (16, 1))
    return FrameSelection(np.ones(16, dtype=bool), metrics, {})


def test_unknown_latitude_is_labelled_and_ring_phase_poleon_are_not_prefilled():
    cfg = ReconstructionConfig(device='cpu', threads=2)
    result = estimate_flattening(selection(.93), cfg)
    assert result['status'] == 'apparent'
    assert result['suggestions']['flattening'] == pytest.approx(.07)
    assert any('equator-on' in note for note in result['notes'])
    for settings, ratio in [(replace(cfg, geometry_mode='saturn'), .8), (cfg, .6),
                             (replace(cfg, sub_obs_lat_rad=1.55), .99),
                             (replace(cfg, sub_obs_lat_rad=1.2), .8)]:
        report = estimate_flattening(selection(ratio), settings)
        assert report['status'] == 'unresolved' and not report['suggestions']


def test_cached_flattening_is_not_used_as_saturn_globe_shape():
    globe = GlobeParams(40., flattening=.1)
    frame = render_globe_texture(112, 128, 64, 56, 0., globe, 0., lambda lon, lat: np.full_like(lon, 100.), limb_weight=False)
    cfg = ReconstructionConfig(device='cpu', threads=2)
    selected = preprocess_source(ArraySource(np.stack([frame]*16)), cfg)
    assert 'flattening' in selected.summary['geometry_estimate']['suggestions']
    report = cache_report(selected, config=replace(cfg, geometry_mode='saturn'))
    assert 'flattening' not in report['geometry_estimate']['suggestions']


def test_flattening_prefill_and_manual_edits():
    pytest.importorskip('PySide6')
    from planetrecon.gui.app import create_app
    from planetrecon.gui.controls import ConfigControls
    app = create_app(['shape-test'])
    controls = ConfigControls(ReconstructionConfig())
    estimate = {'suggestions': {'flattening': .075}}
    try:
        controls.prefill_geometry(estimate)
        assert controls.configuration().flattening == .075
        controls.prefill_geometry({'suggestions': {}})
        assert controls.configuration().flattening == 0.
        controls.fields['flattening'].setText('0.04')
        controls.fields['flattening'].textEdited.emit('0.04')
        controls.prefill_geometry(estimate)
        assert controls.configuration().flattening == .04
    finally:
        controls.close()
        app.processEvents()
