"""The globe's centre and radius come from its limb, not from where its light is brightest."""
import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from planetrecon.geometry.fit import fit_disc_ellipse, fit_globe_disc, fit_limb_ellipse


def globe(shape=(200, 220), centre=(104.3, 97.6), a=70., b=65.5, pa=.9, dim=.45, blur=2.5, noise=.5, seed=0):
    """A limb-darkened oblate globe, dimmer towards one equatorial end (phase), blurred by seeing."""
    h, w = shape
    y, x = np.indices(shape, dtype=float) + .5
    dx, dy = x - centre[0], y - centre[1]
    u = (dx*np.cos(pa) + dy*np.sin(pa)) / a
    v = (-dx*np.sin(pa) + dy*np.cos(pa)) / b
    r2 = u*u + v*v
    mu = np.sqrt(np.clip(1 - r2, 0, 1))
    image = np.where(r2 < 1, 200 * mu**.5 * (1 - dim*np.clip(-u, 0, 1)**1.5) * (1 + .1*np.sin(9*v)), 0.)
    image = gaussian_filter(image, blur) + 3
    return image + np.random.default_rng(seed).normal(0, noise, shape)


@pytest.mark.parametrize('pa', [-.4, .9, 2.])
def test_limb_fit_ignores_a_dim_equatorial_end(pa):
    image = globe(pa=pa)
    moments, limb = fit_disc_ellipse(image), fit_limb_ellipse(image)
    assert np.hypot(moments['cx'] - 104.3, moments['cy'] - 97.6) > 2  # The brightness centroid is pulled sideways.
    assert limb['ok'] and limb['n_edges'] > 150
    assert limb['cx'] == pytest.approx(104.3, abs=.15) and limb['cy'] == pytest.approx(97.6, abs=.15)
    # Blur moves a limb-darkened limb's steepest edge inwards by the same amount all round.
    assert limb['semi_major'] - 70 == pytest.approx(limb['semi_minor'] - 65.5, abs=.3)
    assert -1.6 < limb['semi_major'] - 70 < 0
    assert np.cos(2*(limb['pa_rad'] - pa)) > np.cos(2*.03)
    assert moments['semi_major'] < 64  # Moments fall well inside a limb-darkened limb.


def test_limb_fit_uses_rays_that_end_at_the_frame():
    """A disc filling most of the frame still has every limb point inside it."""
    image = globe(shape=(150, 150), centre=(75.2, 74.6), a=68., b=64., dim=0.)
    limb = fit_limb_ellipse(image)
    assert limb['n_edges'] > 170
    assert limb['cx'] == pytest.approx(75.2, abs=.15) and limb['cy'] == pytest.approx(74.6, abs=.15)


def test_a_limb_cut_by_the_frame_is_not_fitted_there():
    image = globe(shape=(200, 160), centre=(140., 100.), a=60., b=57., dim=0.)  # Right limb beyond the frame.
    limb = fit_globe_disc(image)
    assert limb['method'] == 'limb edges'
    assert limb['cx'] == pytest.approx(140., abs=.3) and limb['cy'] == pytest.approx(100., abs=.3)


def test_featureless_or_tiny_targets_fall_back_to_moments():
    tiny = np.zeros((40, 40))
    tiny[18:21, 18:21] = 100.
    assert fit_globe_disc(tiny)['method'] == 'brightness moments'
    assert fit_globe_disc(np.zeros((40, 40)))['ok'] is False


@pytest.mark.parametrize('offset, warned', [(0., False), (3., True)])
def test_surface_stack_warns_when_the_globe_centre_is_off_the_limb(offset, warned):
    from planetrecon.geometry.globe import GlobeParams, render_globe_texture
    from planetrecon.io.source import ArraySource
    from planetrecon.pipeline.baseline import stack_source
    from planetrecon.reconstruction import ReconstructionConfig
    globe_ = GlobeParams(40, pole_pa_rad=.4, surface_rate_rad_s=.002, sub_obs_lat_rad=0.)
    times = np.linspace(0, 30, 12)
    texture = lambda lon, lat: 100 + 20*np.sin(15*lon + 3*lat) + 12*np.cos(17*lat - 8*lon)
    frames = np.stack([gaussian_filter(render_globe_texture(112, 128, 64, 56, 0., globe_, t, texture), 1.) for t in times])
    cfg = ReconstructionConfig(device='cpu', threads=2, geometry_mode='surface', frame_preselection=False,
                               field_center_x=64 + offset, field_center_y=56, equatorial_radius_px=40,
                               surface_rate_rad_s=.002, pole_pa_rad=.4, sub_obs_lat_rad=0.)
    result = stack_source(ArraySource(frames, timestamps=times), cfg)
    assert result.provenance['geometry']['limb_centre_offset_px'] == pytest.approx(offset, abs=.3)
    assert any('globe_centre_off_limb' in w for w in result.warnings) == warned
