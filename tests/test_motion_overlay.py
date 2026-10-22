"""Predicted motion arrows follow the stack's own motion model."""
import numpy as np
import pytest

from planetrecon.geometry.globe import GlobeParams
from planetrecon.geometry.model import OblateGlobeModel, FieldOnlyModel
from planetrecon.geometry.pose import FramePose
from planetrecon.gui.motion import predicted_motion
from planetrecon.reconstruction import ReconstructionConfig


def config(**kwargs):
    return ReconstructionConfig(**{'geometry_mode': 'surface', 'field_center_x': 200., 'field_center_y': 150.,
                                   'equatorial_radius_px': 80., 'surface_rate_rad_s': 1.7e-4,
                                   'sub_obs_lat_rad': .3, 'pole_pa_rad': .4, **kwargs})


def tangent_at_middle(points):
    k = len(points) // 2
    return points[k+1] - points[k-1], points[k]


@pytest.mark.parametrize('rate', [1.7e-4, -1.7e-4])
@pytest.mark.parametrize('lat', [.3, -.3, .02, -.02])  # Nearly edge-on views are drawn tilted.
def test_surface_arrow_runs_the_way_the_globe_model_moves_features(rate, lat):
    cfg = config(surface_rate_rad_s=rate, sub_obs_lat_rad=lat)
    (arrow,), notes = predicted_motion(cfg)
    assert arrow.kind == 'surface' and not notes
    direction, (x, y) = tangent_at_middle(arrow.points)
    assert np.hypot(x - 200, y - 150) < 80  # On the disc.
    globe = GlobeParams(80., pole_pa_rad=.4, sub_obs_lat_rad=lat, surface_rate_rad_s=rate)
    dt = 10.
    later_x, later_y, valid = OblateGlobeModel(globe, apply_field=False).src_to_ref(
        np.array([x]), np.array([y]), FramePose(0., 0., 200., 150.), FramePose(dt, 0., 200., 150.))
    assert valid[0] and np.dot(direction, [later_x[0] - x, later_y[0] - y]) > 0


def test_reversed_rotation_reverses_the_preset_arrow():
    forward, _ = predicted_motion(config(surface_rate_rad_s=None, rotation_planet='saturn'))
    backward, _ = predicted_motion(config(surface_rate_rad_s=None, rotation_planet='saturn', reverse_rotation=True))
    np.testing.assert_allclose(forward[0].points, backward[0].points[::-1])


@pytest.mark.parametrize('rate', [2e-4, -2e-4])
def test_field_arrow_turns_with_the_field_and_stays_on_the_detector(rate):
    cfg = config(geometry_mode='saturn', field_rate_rad_s=rate, ring_inner_radius_px=100.,
                 ring_outer_radius_px=140.)
    arrows, _ = predicted_motion(cfg, shape=(300, 400))
    field = next(a for a in arrows if a.kind == 'field')
    assert (field.points[:, 1] >= 0).all() and (field.points[:, 1] <= 300).all()
    direction, (x, y) = tangent_at_middle(field.points)
    later_x, later_y, _ = FieldOnlyModel().src_to_ref(np.array([x]), np.array([y]),
        FramePose(0., 0., 200., 150.), FramePose(10., rate*10., 200., 150.))
    assert np.dot(direction, [later_x[0] - x, later_y[0] - y]) > 0


def test_notes_explain_missing_predictions():
    assert predicted_motion(config(geometry_mode='none'))[0] == []
    assert 'centre and radius' in predicted_motion(config(field_center_x=None))[1][0]
    arrows, notes = predicted_motion(config(geometry_mode='combined'))
    assert [a.kind for a in arrows] == ['surface'] and 'estimated when stacking' in notes[0]
    assert predicted_motion(config(surface_rate_rad_s=0.))[0] == []


def test_image_tab_draws_arrows_after_preprocessing(monkeypatch):
    monkeypatch.setenv('QT_QPA_PLATFORM', 'offscreen')
    from planetrecon.gui.app import MainWindow, create_app
    app = create_app([])
    win = MainWindow(config=config(local_alignment=True))
    try:
        win.input_image = np.full((300, 400), 100.)
        win.input_stride = 1
        win.view.setCurrentText('Input')
        win.zoom.setCurrentText('100%')
        win.black.setValue(0.); win.white.setValue(1000.)

        def drawn():
            win._draw()
            return win.image_label.pixmap().toImage()
        plain = drawn()
        win.preprocessing_info = {'status': 'ready'}
        arrows = drawn()
        assert arrows != plain and 'Predicted motion: surface' in win.histogram.text()
        win.motion_check.setChecked(False)
        assert drawn() == plain
    finally:
        win.window.close()
