"""Camera drift must not smear Saturn's independently classified layers."""

from dataclasses import replace

import numpy as np
import pytest

from planetrecon.detector import cfa_labels
from planetrecon.geometry.globe import GlobeParams
from planetrecon.geometry.pose import FramePose
from planetrecon.geometry.rings import RingParams
from planetrecon.geometry.saturn import render_saturn
from planetrecon.io.source import ArraySource
from planetrecon.pipeline.baseline import stack_source
from planetrecon.reconstruction import ReconstructionConfig


def capture(colour, offsets):
    globe = GlobeParams(equatorial_radius_px=20, flattening=.1, sub_obs_lat_rad=.4)
    rings = RingParams(26, 42, transmission=0)
    image = render_saturn(112, 112, FramePose(0, 0, 56, 56), globe, rings,
        lambda lon, lat: 1 + .2*np.cos(5*lon)*np.cos(3*lat))
    if colour != 'mono':
        labels = cfa_labels(112, 112, colour)
        image *= sum((labels == name)*value for name, value in zip('RGB', (.8, 1., .6)))
    frames = np.array([np.roll(image, (dy, dx), axis=(0, 1)) for dx, dy in offsets])
    return ArraySource(frames, color_mode=colour, bit_depth=32,
                       timestamps=np.arange(len(frames), dtype=float))


def settings(**kwargs):
    return ReconstructionConfig(device='cpu', threads=2, frame_preselection=False,
        geometry_mode='saturn', field_center_x=56, field_center_y=56,
        equatorial_radius_px=20, flattening=.1, sub_obs_lat_rad=.4,
        ring_inner_radius_px=26, ring_outer_radius_px=42, ring_transmission=0,
        surface_rate_rad_s=0, field_rate_rad_s=0, **kwargs)


@pytest.mark.parametrize('colour', ['mono', 'RGGB', 'GRBG', 'GBRG', 'BGGR'])
def test_saturn_camera_drift_preserves_globe_and_ring_image(colour):
    offsets = [(0, 0), (2, -4), (-4, 2), (4, 4), (-2, -2)]
    result = stack_source(capture(colour, offsets), settings())
    stationary = stack_source(capture(colour, [(0, 0)]*len(offsets)), settings())
    mask = stationary.validity & (stationary.image > .05)
    error = np.sqrt(np.mean((result.image[mask] - stationary.image[mask])**2))
    assert result.n_used == len(offsets) and result.n_rejected == 0
    assert error < 1e-10
    for layer in ('globe', 'ring'):
        np.testing.assert_allclose(result.layer_coverage[layer],
                                   stationary.layer_coverage[layer], atol=1e-10)


def test_explicit_moon_tracks_keep_fixed_detector_centres():
    result = stack_source(capture('mono', [(0, 0), (2, -4)]),
        settings(moon_x=10, moon_y=10, moon_radius_px=2))
    assert result.provenance['geometry']['registration'] == 'fixed centre (Saturn detector tracks)'


def test_static_saturn_uses_existing_camera_shift_limit():
    result = stack_source(capture('mono', [(0, 0), (4, 4)]), settings(max_shift_px=2))
    assert result.n_used == 1 and result.n_rejected == 1


@pytest.mark.parametrize('colour', ['mono', 'RGGB'])
def test_saturn_auto_field_rate_recognizes_camera_drift(colour):
    src = capture(colour, [(0, 0), (2, -4), (-4, 2), (4, 4), (-2, -2)])
    inferred = stack_source(src, replace(settings(), field_rate_rad_s=None))
    known = stack_source(src, settings())
    assert inferred.provenance['geometry']['field_rate_rad_s'] == 0
    assert inferred.n_used == known.n_used == 5
    np.testing.assert_array_equal(inferred.image, known.image)
    np.testing.assert_array_equal(inferred.coverage, known.coverage)
