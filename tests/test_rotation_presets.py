"""Published rotation priors supply angular motion without claiming a texture fit."""
from dataclasses import replace
import math

import numpy as np
import pytest

from planetrecon.geometry.rotation import PERIOD_DAYS, rotation_preset
from planetrecon.reconstruction import ReconstructionConfig


@pytest.mark.parametrize('planet', list(PERIOD_DAYS))
def test_period_rate_direction_and_pixel_scale(planet):
    from planetrecon.geometry.globe import GlobeParams, body_to_sky
    small = rotation_preset(planet, radius_px=50.)
    large = rotation_preset(planet, radius_px=100.)
    rate = small['surface_rate_rad_s']
    assert rate * PERIOD_DAYS[planet] * 86400 == pytest.approx(2*math.pi)
    assert rate < 0 if planet in ('venus', 'uranus') else rate > 0
    assert large['surface_rate_rad_s'] == rate
    assert large['equator_on_speed_px_s'] == 2*small['equator_on_speed_px_s']
    assert rotation_preset(planet, reverse=True)['surface_rate_rad_s'] == -rate
    for radius in (50., 100.):
        globe = GlobeParams(radius, surface_rate_rad_s=rate)
        x, y, valid, _ = body_to_sky(0., 0., globe, 360.)
        assert valid
        assert x == pytest.approx(-radius*math.sin(rate*360), abs=1e-12)


def test_preset_and_explicit_zero_override_are_distinct_and_serializable():
    cfg = ReconstructionConfig(geometry_mode='surface', rotation_planet='jupiter')
    assert cfg.effective_surface_rate() == rotation_preset('jupiter')['surface_rate_rad_s']
    cfg.require_motion_parameters()
    assert ReconstructionConfig.from_dict(cfg.to_dict()) == cfg
    assert replace(cfg, surface_rate_rad_s=0., reverse_rotation=True).effective_surface_rate() == 0.
    assert replace(cfg, surface_rate_rad_s=.01, reverse_rotation=True).effective_surface_rate() == .01
    with pytest.raises(ValueError, match='rotation_planet'):
        replace(cfg, rotation_planet='unknown')
    with pytest.raises(ValueError, match='reverse_rotation'):
        replace(cfg, reverse_rotation=1)


def test_preset_does_not_invent_saturn_viewing_geometry_or_seconds():
    from planetrecon.io.source import ArraySource
    from planetrecon.pipeline.geometry_stack import prepare_geometry
    cfg = ReconstructionConfig(geometry_mode='saturn', rotation_planet='saturn')
    assert 'surface_rate_rad_s' not in cfg.missing_motion_parameters()
    assert 'sub_obs_lat_rad' in cfg.missing_motion_parameters()
    with pytest.raises(ValueError, match='Signed observer latitude'):
        cfg.require_motion_parameters()
    source = ArraySource(np.ones((3, 40, 40)))
    with pytest.raises(ValueError, match='measured timestamps'):
        prepare_geometry(source, replace(cfg, geometry_mode='surface'))


@pytest.mark.parametrize('reverse', [False, True])
def test_surface_stack_matches_explicit_published_rate_and_resumes(tmp_path, reverse):
    from test_globe_registration import sphere, config
    from planetrecon.io.ser import SERSource, write_ser
    from planetrecon.pipeline.baseline import stack_source
    rate = rotation_preset('jupiter', reverse=reverse)['surface_rate_rad_s']
    times = np.arange(5)*60.
    frames = np.array([sphere(t, rate=rate)*100 for t in times]).astype('u2')
    path = write_ser(tmp_path/'jupiter.ser', frames)
    explicit = replace(config(rate=rate), cadence_s=60.)
    preset = replace(explicit, surface_rate_rad_s=None, rotation_planet='jupiter', reverse_rotation=reverse)
    state = tmp_path/'state.npz'
    with SERSource(path) as source:
        expected = stack_source(source, explicit)
        actual = stack_source(source, preset, state_checkpoint=state)
        resumed = stack_source(source, preset, resume_from=state)
        for field in ('image', 'coverage', 'validity'):
            np.testing.assert_array_equal(getattr(actual, field), getattr(expected, field))
            np.testing.assert_array_equal(getattr(actual, field), getattr(resumed, field))
        geometry = actual.provenance['geometry']
        assert geometry['surface_origin'] == 'planet_preset'
        assert geometry['surface_rotation_preset']['equatorial_radius_px'] == 42
        assert actual.provenance['surface_rotation_preset']['used_for_surface_rate']
        with pytest.raises(ValueError, match='identity/configuration mismatch'):
            stack_source(source, replace(preset, reverse_rotation=not reverse), resume_from=state)


def test_gui_preset_survives_unresolved_preprocessing_and_preserves_manual_override():
    from planetrecon.gui.app import create_app
    from planetrecon.gui.controls import ConfigControls
    app = create_app(['planet-rotation'])
    controls = ConfigControls(ReconstructionConfig(geometry_mode='surface', equatorial_radius_px=50.))
    try:
        controls.prefill_geometry({'suggestions': {'surface_rate_rad_s': .003}})
        chooser = controls.fields['rotation_planet']
        chooser.setCurrentIndex(chooser.findData('saturn'))
        assert controls.fields['surface_rate_rad_s'].text() == ''
        controls.prefill_geometry({'suggestions': {}})
        controls.prefill_geometry({'suggestions': {'surface_rate_rad_s': .004}})
        cfg = controls.configuration()
        assert cfg.surface_rate_rad_s is None
        assert cfg.effective_surface_rate() == rotation_preset('saturn')['surface_rate_rad_s']
        assert 'px/s' in controls.rotation_label.text()
        controls.fields['reverse_rotation'].setChecked(True)
        assert controls.configuration().effective_surface_rate() == -cfg.effective_surface_rate()
        controls.fields['surface_rate_rad_s'].setText('0')
        controls.fields['surface_rate_rad_s'].textEdited.emit('0')
        chooser.setCurrentIndex(chooser.findData('mars'))
        assert controls.configuration().effective_surface_rate() == 0.
        assert 'overrides this preset' in controls.rotation_label.text()
    finally:
        controls.close()
        app.processEvents()


def test_cli_accepts_planet_and_reverse_options(monkeypatch, tmp_path):
    import planetrecon.cli as cli
    import planetrecon.pipeline.baseline as baseline
    from planetrecon.io.ser import write_ser
    seen = []
    def capture(source, config, **kwargs):
        seen.append(config)
        raise ValueError('captured config')
    monkeypatch.setattr(baseline, 'stack_source', capture)
    path = write_ser(tmp_path/'in.ser', np.ones((4, 12, 12), dtype='u2'))
    with pytest.raises(ValueError, match='captured config'):
        cli.main(['stack', '--path', str(path), '--out', str(tmp_path/'out'),
                  '--geometry', 'surface', '--rotation-planet', 'saturn', '--reverse-rotation'])
    assert seen[0].effective_surface_rate() == -rotation_preset('saturn')['surface_rate_rad_s']


def test_saturn_bands_can_stack_with_preset_when_texture_spin_is_unresolved():
    from test_ring_registration import config
    from planetrecon.geometry.discovery import discover_geometry
    from planetrecon.geometry.globe import GlobeParams
    from planetrecon.geometry.pose import FramePose
    from planetrecon.geometry.rings import RingParams
    from planetrecon.geometry.saturn import render_saturn
    from planetrecon.io.source import ArraySource
    from planetrecon.pipeline.baseline import stack_source
    from planetrecon.pipeline.preprocess import screen_source

    cfg = replace(config(), surface_rate_rad_s=None, rotation_planet='saturn')
    globe = GlobeParams(20, flattening=.1, sub_obs_lat_rad=.4,
                        surface_rate_rad_s=cfg.effective_surface_rate())
    rings = RingParams(26, 42, transmission=.35)
    times = np.linspace(0, 360, 16)
    frames = np.array([render_saturn(112, 112, FramePose(t, 0., 56, 56), globe, rings,
                       lambda lon, lat: 1+.2*np.cos(5*lat)) for t in times])
    source = ArraySource(frames, bit_depth=32, timestamps=times)
    estimate = discover_geometry(source, cfg, screen_source(source, cfg))
    assert not estimate.get('surface_resolved', False)
    actual = stack_source(source, cfg)
    explicit = stack_source(source, replace(cfg, rotation_planet=None,
                                            surface_rate_rad_s=cfg.effective_surface_rate()))
    assert actual.n_used == 16 and not actual.incomplete
    assert actual.provenance['geometry']['surface_origin'] == 'planet_preset'
    np.testing.assert_array_equal(actual.image, explicit.image)
    np.testing.assert_array_equal(actual.coverage, explicit.coverage)
