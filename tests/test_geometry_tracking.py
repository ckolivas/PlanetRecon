"""Geometry must retain registration and real colour support."""
from dataclasses import replace

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter, shift

from planetrecon.detector import cfa_labels
from planetrecon.geometry.globe import GlobeParams, render_globe_texture
from planetrecon.io.source import ArraySource
from planetrecon.pipeline.baseline import stack_source
from planetrecon.pipeline.preprocess_cache import preprocess_source
from planetrecon.reconstruction import ReconstructionConfig


@pytest.mark.parametrize('color', ['mono', 'RGB', 'RGGB'])
@pytest.mark.parametrize('preprocessed', [False, True])
def test_explicit_zero_surface_rate_matches_registered_stack(color, preprocessed):
    y, x = np.indices((80, 96))
    scene = gaussian_filter((((x-48)/25)**2 + ((y-40)/22)**2 < 1)*
                            (100+15*np.cos(y/3)+10*np.cos(x/4)), .8)
    offsets = [(0, 0), (2.2, -3.7), (-1.3, 4.1), (3.7, 1.2)]*4
    planes = np.stack([shift(scene, d, order=3, mode='constant') for d in offsets])
    if color == 'RGB':
        planes = planes[..., None]*[.8, 1., .6]
    elif color == 'RGGB':
        labels = cfa_labels(*scene.shape, color)
        planes *= np.where(labels == 'R', .8, np.where(labels == 'B', .6, 1.))
    source = ArraySource(planes, color_mode=color, bit_depth=32, timestamps=np.arange(len(planes)))
    cfg = ReconstructionConfig(device='cpu', threads=2, field_center_x=48.5, field_center_y=40.5,
        equatorial_radius_px=25., pole_pa_rad=.3, flattening=.1, sub_obs_lat_rad=.1)
    selection = preprocess_source(source, cfg) if preprocessed else None
    baseline = stack_source(source, cfg, preprocessing=selection)
    surface = stack_source(source, replace(cfg, geometry_mode='surface', surface_rate_rad_s=0.), preprocessing=selection)
    assert surface.n_used == baseline.n_used
    np.testing.assert_allclose(surface.image, baseline.image, atol=1e-9)
    np.testing.assert_allclose(surface.coverage, baseline.coverage, atol=1e-9)
    np.testing.assert_array_equal(surface.validity, baseline.validity)


@pytest.mark.parametrize('rate', [-.04, .04])
def test_translation_tracking_preserves_signed_surface_compensation(rate):
    size, radius = 96, 31.
    globe = GlobeParams(radius, surface_rate_rad_s=rate, pole_pa_rad=.25)
    def texture(lon, lat):
        return 100 + 20*np.cos(12*lon)*np.cos(3*lat) + 15*np.sin(14*lat+2*lon)
    times = np.arange(12.)
    frames = [render_globe_texture(size, size, 48+3*np.sin(t), 48+2*np.sin(t*.7),
              0., globe, t, texture, apply_field=False, limb_weight=False) for t in times]
    source = ArraySource(np.stack(frames), bit_depth=32, timestamps=times)
    cfg = ReconstructionConfig(device='cpu', threads=2, frame_preselection=False,
        geometry_mode='surface', field_center_x=48., field_center_y=48.,
        equatorial_radius_px=radius, pole_pa_rad=.25, surface_rate_rad_s=rate)
    result = stack_source(source, cfg)
    ordinary = stack_source(source, replace(cfg, geometry_mode='none'))
    y, x = np.indices((size, size))
    inner = (x+.5-48)**2+(y+.5-48)**2 < (.65*radius)**2
    error = np.sqrt(np.mean((result.image[inner]-frames[0][inner])**2))
    uncorrected = np.sqrt(np.mean((ordinary.image[inner]-frames[0][inner])**2))
    # Compare with the same resampling without jitter: bilinear reconstruction
    # itself attenuates this fine texture, even with exactly known poses.
    steady = [render_globe_texture(size, size, 48., 48., 0., globe, t, texture,
              apply_field=False, limb_weight=False) for t in times]
    control = stack_source(ArraySource(np.stack(steady), bit_depth=32, timestamps=times), cfg)
    control_error = np.sqrt(np.mean((control.image[inner]-frames[0][inner])**2))
    assert error < control_error + 1.
    assert error < .4*uncorrected


def test_surface_tracking_obeys_maximum_shift():
    y, x = np.indices((64, 64))
    scene = np.exp(-((x-32)**2+(y-32)**2)/50)*100
    frames = np.stack([scene, shift(scene, (0, 8), order=1)])
    src = ArraySource(frames, bit_depth=32, timestamps=np.arange(2.))
    cfg = ReconstructionConfig(device='cpu', threads=2, geometry_mode='surface',
        field_center_x=32., field_center_y=32., equatorial_radius_px=15., max_shift_px=4., surface_rate_rad_s=0.)
    result = stack_source(src, cfg)
    assert result.n_used == 1 and result.n_rejected == 1


def test_roundoff_does_not_create_spurious_cfa_coverage():
    from planetrecon.geometry.warp import bilinear_push
    y, x = np.indices((12, 14), dtype=float)
    red = cfa_labels(12, 14, 'RGGB') == 'R'
    samples = np.full(red.shape, 30.)
    accum, coverage = bilinear_push(samples, y+1e-12, x-1e-12, red.shape, valid=red)
    np.testing.assert_array_equal(coverage, red.astype(float))
    np.testing.assert_array_equal(accum, 30*red)
    # Real subpixel displacement must still spread a sample's support.
    _, fractional = bilinear_push(samples, y+.25, x+.25, red.shape, valid=red)
    assert np.count_nonzero(fractional) > red.sum()
