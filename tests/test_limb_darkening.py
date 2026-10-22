"""Limb-darkening compensation keeps fixed limb darkening still while globe light moves."""
from dataclasses import replace

import numpy as np
import pytest

from planetrecon.geometry.coords import detector_to_sky, detector_xy_grids
from planetrecon.geometry.globe import GlobeParams, globe_hit
from planetrecon.geometry.limb import fit_limb_darkening, ld_ratio
from planetrecon.geometry.model import OblateGlobeModel
from planetrecon.geometry.pose import FramePose
from planetrecon.io.source import ArraySource
from planetrecon.pipeline.baseline import stack_source
from planetrecon.reconstruction import ReconstructionConfig

POWER = .7
SIZE, CENTRE, RADIUS = 128, 64., 42.


def zonal_globe(power=POWER, latitude=.15, rings=None):
    """A globe with limb darkening mu ** power and latitude belts only: spin changes nothing.

    ``rings`` (RingParams) adds uniform rings outside the globe for Saturn drift tracking.
    """
    globe = GlobeParams(RADIUS, sub_obs_lat_rad=latitude)
    x, y = detector_xy_grids(SIZE, SIZE)
    sky = detector_to_sky(x, y, CENTRE, CENTRE)
    _lon, lat, visible, mu, _t = globe_hit(*sky, globe, 0.)
    belts = 1 + .15*np.cos(9*np.nan_to_num(lat))
    image = np.where(visible, 1000*belts*mu**power, 0.)
    if rings is not None:
        from planetrecon.geometry.saturn import classify_layers
        info = classify_layers(*sky, globe, rings, 0.)
        radius = np.nan_to_num(info['ring_radius'])
        image = np.where(info['on_ring'] & ~visible, 400 + 300*np.cos(radius/2.)**2, image)
    return image, mu


def test_fit_recovers_the_power_despite_belts():
    image, _ = zonal_globe()
    globe = GlobeParams(RADIUS, sub_obs_lat_rad=.15)
    fit = fit_limb_darkening(image, globe, CENTRE, CENTRE)
    assert fit['origin'] == 'fitted' and fit['pixels'] > 1000
    assert fit['power'] == pytest.approx(POWER, abs=.01)


def test_gain_is_one_without_spin_or_power():
    model = OblateGlobeModel(GlobeParams(RADIUS, surface_rate_rad_s=.01), apply_field=False, ld_power=0.)
    x, y = detector_xy_grids(SIZE, SIZE)
    *_, gain = model.src_to_ref(x, y, FramePose(3., 0., CENTRE, CENTRE), FramePose(0., 0., CENTRE, CENTRE), gain=True)
    assert np.all(gain == 1.)
    np.testing.assert_allclose(ld_ratio(np.array([.1, .5]), np.array([.05, .25]), 1.), [1., 2.])


@pytest.mark.parametrize('mode', ['surface', 'saturn'])
def test_spin_with_late_frames_leaves_no_rind(mode):
    from planetrecon.geometry.rings import RingParams
    image, mu = zonal_globe(rings=RingParams(50., 62.) if mode == 'saturn' else None)
    times = np.array([0., 7., 8., 9., 10.])   # weighted well after the reference time of 5 s
    source = ArraySource(np.repeat(image[None], len(times), 0), bit_depth=32, timestamps=times)
    cfg = ReconstructionConfig(device='cpu', threads=2, frame_preselection=False, motion_reference='best',
                               geometry_mode=mode, field_center_x=CENTRE, field_center_y=CENTRE,
                               equatorial_radius_px=RADIUS, sub_obs_lat_rad=.15, surface_rate_rad_s=.012,
                               field_rate_rad_s=0., reference_epoch_s=5.)
    if mode == 'saturn':
        cfg = replace(cfg, ring_inner_radius_px=50., ring_outer_radius_px=62.)
    compensated = stack_source(source, cfg)
    plain = stack_source(source, replace(cfg, limb_darkening_power=0.))
    assert compensated.provenance['geometry']['limb_darkening']['power'] == pytest.approx(POWER, abs=.02)
    # Where spin moves light (Saturn's motion fades out within mu 0.4 of the limb).
    region = (mu > .45) & (mu < .95)
    error = lambda result: np.sqrt(np.mean((result.image[region] - image[region]) ** 2))
    assert error(compensated) < .15 * error(plain)
    assert error(compensated) < .002 * image[region].mean()
