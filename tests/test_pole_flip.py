"""Correcting north/south ambiguity updates the actual next-run geometry."""
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from test_w14 import gui


def test_flip_preserves_manual_direction_through_preprocessing(gui):
    _, window = gui
    controls = window.controls
    estimate = {'suggestions': {'pole_pa_rad': np.deg2rad(12.5)}}
    controls.prefill_geometry(estimate)
    controls.flip_pole_button.click()
    assert controls.configuration().pole_pa_rad == pytest.approx(np.deg2rad(192.5))
    assert 'pole_pa_rad' in controls.geometry_manual
    controls.prefill_geometry(estimate)
    controls.clear_geometry_estimate()
    assert controls.configuration().pole_pa_rad == pytest.approx(np.deg2rad(192.5))
    controls.flip_pole_button.click()
    assert controls.configuration().pole_pa_rad == pytest.approx(np.deg2rad(12.5))
    assert controls.flip_pole_button.toolTip()


@pytest.mark.parametrize('text', ['', 'invalid', 'nan', 'inf'])
def test_invalid_pole_does_not_crash_or_become_manual_override(gui, text):
    _, window = gui
    controls = window.controls
    controls.fields['pole_pa_rad'].setText(text)
    controls.flip_pole_button.click()
    assert controls.fields['pole_pa_rad'].text() == text
    assert 'pole_pa_rad' not in controls.geometry_manual
    assert 'finite pole position angle' in controls.fields['geometry_mode'].toolTip()


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
