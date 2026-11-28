"""Preview arrows point to the moons a stack found and stacked on their own positions."""
import numpy as np
import pytest

from planetrecon.gui.moon_markers import moon_markers, moon_positions


def scene(h=200, w=300, planet=(150, 100)):
    y, x = np.indices((h, w)) + .5
    return np.where(np.hypot(x - planet[0], y - planet[1]) < 40, 1000., 10.)


def provenance(*positions, final=True):
    entries = [{'output_position_px': [y, x], 'planet_registered_fwhm_px': 4.} for y, x in positions]
    moons = {'enabled': True, 'stacked': len(entries)}
    moons['moons' if final else 'positions_px'] = entries
    return {'moons': moons}


@pytest.mark.parametrize('final', [True, False])
def test_arrows_point_at_each_moon_from_away_from_the_planet(final):
    image = scene()
    markers, notes = moon_markers(provenance((100., 240.), (30., 150.), final=final), image)
    assert [m.label for m in markers] == ['moon 1', 'moon 2'] and 'Moons: 2' in notes[0]
    for marker, (mx, my) in zip(markers, [(240.5, 100.5), (150.5, 30.5)]):
        assert marker.position == (mx, my)  # Pixel indices become detector pixel centres.
        tail, tip = marker.points
        towards = np.subtract((mx, my), tip)
        along = np.subtract(tip, tail)
        assert np.dot(towards, along) > 0 and abs(towards[0]*along[1] - towards[1]*along[0]) < 1e-6 * np.linalg.norm(along)**2
        assert 6. <= np.hypot(*towards) < 12  # Stops short of the moon's light.
        # Approaches from the side away from the planet.
        assert np.hypot(tail[0] - 150, tail[1] - 100) > np.hypot(mx - 150, my - 100)


def test_arrows_near_the_frame_edge_turn_to_stay_inside():
    image = scene()
    (marker,), _ = moon_markers(provenance((100., 296.)), image)  # Right edge, outward is off the frame.
    tail, tip = marker.points
    assert 0 <= tail[0] <= 300 and 0 <= tail[1] <= 200
    assert np.hypot(*np.subtract(marker.position, tip)) < 12


def test_strided_previews_place_arrows_in_detector_pixels():
    image = scene()
    full, _ = moon_markers(provenance((100., 240.)), image)
    strided, _ = moon_markers(provenance((100., 240.)), image[::2, ::2], stride=2)
    np.testing.assert_allclose(strided[0].points, full[0].points, atol=2.)


def test_no_moons_or_skipped_moon_stacking():
    assert moon_markers({}, scene()) == ([], [])
    markers, notes = moon_markers({'moons': {'enabled': True, 'skipped': 'not available with planet rotation models'}},
                                  scene())
    assert markers == [] and 'planet rotation models' in notes[0]
    assert moon_positions({'moons': {'enabled': False}}) == []


def test_progress_previews_record_moon_positions(tmp_path):
    from planetrecon.io.ser import SERSource
    from planetrecon.pipeline.baseline import stack_source
    from planetrecon.pipeline.preprocess_cache import preprocess_source
    from test_moons import BRIGHT, capture, config
    previews = []
    with SERSource(capture(tmp_path)) as source:
        preprocess_source(source, config())
        final = stack_source(source, config(), on_event=lambda result, info: hasattr(result, 'image') and previews.append(result.provenance))
    early = previews[0]['moons']['positions_px']
    assert len(early) == 2 and final.provenance['moons']['positions_px'] == early
    assert [m['output_position_px'] for m in final.provenance['moons']['moons']] == \
        [m['output_position_px'] for m in early]
    assert min(np.hypot(*np.subtract(p['output_position_px'], BRIGHT)) for p in early) < 2.  # Planet-registered.


def test_result_view_draws_moon_arrows(monkeypatch):
    monkeypatch.setenv('QT_QPA_PLATFORM', 'offscreen')
    from planetrecon.gui.app import MainWindow, create_app
    from planetrecon.reconstruction import ReconstructionConfig
    from planetrecon.result import ReconstructionResult
    create_app([])
    win = MainWindow(config=ReconstructionConfig(device='cpu'))
    try:
        image = scene()
        win.preview = ReconstructionResult(image=image, coverage=np.ones_like(image), validity=np.ones(image.shape, bool),
                                           units='adu', channel_order='mono', backend='cpu', precision='float64', stage='final',
                                           incomplete=False, provenance=provenance((100., 240.)))
        win.view.setCurrentText('Result')
        win.zoom.setCurrentText('100%')
        win.black.setValue(0.); win.white.setValue(1000.)

        def drawn():
            win._draw()
            return win.image_label.pixmap().toImage()
        arrows = drawn()
        green = arrows.pixelColor(int(np.mean([255, 262])), 100)  # Along the arrow, right of the moon.
        assert 'Moons: 1 stacked' in win.histogram.text()
        win.moons_check.setChecked(False)
        plain = drawn()
        assert arrows != plain and 'Moons:' not in win.histogram.text()
        assert green.green() > 200 and green.red() < 200
    finally:
        win.window.close()
