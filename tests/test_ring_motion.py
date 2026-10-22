"""Saturn's rings turn with the planet: spokes are sharpened, uniform rings and fixed light preserved."""
import numpy as np
import pytest

from planetrecon.geometry.coords import detector_to_sky, detector_xy_grids
from planetrecon.geometry.globe import GlobeParams
from planetrecon.geometry.pose import FramePose
from planetrecon.geometry.rings import RingParams
from planetrecon.geometry.saturn import SaturnSceneModel, classify_layers, default_ring_profile, render_saturn
from planetrecon.io.source import ArraySource
from planetrecon.pipeline.baseline import stack_source
from planetrecon.geometry.fit import field_rate_fit
from planetrecon.reconstruction import ReconstructionConfig

from scipy.ndimage import gaussian_filter

# Real proportions (1211_8: globe 96 px, rings 148-220 px) with spokes 20 degrees in
# from the ansae, moving about 5 px over the capture as the 1211_8 spoke did.
SIZE, CENTRE, RADIUS, INNER, OUTER = 200, 100., 30., 46., 69.
RATE, LATITUDE = .06, .3
TIMES = np.arange(4.)
REFERENCE = 1.5
DRIFT = [(.37, -.21), (-.44, .13), (.18, .41), (-.29, -.36)]
SEEING_PX = 1.5


def globe():
    return GlobeParams(RADIUS, flattening=.1, sub_obs_lat_rad=LATITUDE, surface_rate_rad_s=RATE,
                       reference_epoch_s=REFERENCE)


def rings(**kwargs):
    return RingParams(INNER, OUTER, transmission=.35, **kwargs)


def uniform(radius, azimuth):
    return default_ring_profile(radius, azimuth, INNER, OUTER)


def spokes(radius, azimuth):
    """Dark spokes across the inner ring, a few degrees wide (azimuth 0 faces the observer; ansae at +-90)."""
    dip = sum(np.exp(-.5*(np.angle(np.exp(1j*(azimuth - centre)))/.03)**2)
              for centre in (np.pi/2 - .35, -np.pi/2 + .35, np.pi/2 + .4))
    return uniform(radius, azimuth)*(1 - .5*dip*(radius < 58))


def frame(t, ring_tex, dx=0., dy=0.):
    image = render_saturn(SIZE, SIZE, FramePose(t, 0., CENTRE + dx, CENTRE + dy), globe(), rings(),
                          lambda lon, lat: 1 + .1*np.cos(3*lat), ring_tex=ring_tex, turning_rings=True)
    return gaussian_filter(image, SEEING_PX)


def capture(ring_tex):
    return ArraySource(np.array([frame(t, ring_tex, *d) for t, d in zip(TIMES, DRIFT)]), bit_depth=32,
                       timestamps=TIMES)


def config():
    return ReconstructionConfig(device='cpu', threads=2, frame_preselection=False, motion_reference='best',
        geometry_mode='saturn', field_center_x=CENTRE, field_center_y=CENTRE, equatorial_radius_px=RADIUS,
        flattening=.1, sub_obs_lat_rad=LATITUDE, ring_inner_radius_px=INNER, ring_outer_radius_px=OUTER,
        ring_transmission=.35, surface_rate_rad_s=RATE, field_rate_rad_s=0, reference_epoch_s=REFERENCE)


def exposed_rings():
    x, y = detector_xy_grids(SIZE, SIZE)
    info = classify_layers(*detector_to_sky(x, y, CENTRE, CENTRE), globe(), rings(), REFERENCE)
    return info['on_ring'] & ~info['on_globe']


def errors(monkeypatch, ring_tex):
    """Exposed-ring RMS error against the scene at the reference time, rings turning and held still."""
    truth = frame(REFERENCE, ring_tex, *DRIFT[0])   # output geometry is the first (reference) frame's
    region = exposed_rings()
    turning = stack_source(capture(ring_tex), config())
    assert 'ring_motion' in turning.provenance['geometry']
    original = SaturnSceneModel.__init__
    monkeypatch.setattr(SaturnSceneModel, '__init__',
                        lambda self, *args, **kwargs: original(self, *args, **{**kwargs, 'ring_motion': False}))
    fixed = stack_source(capture(ring_tex), config())
    assert turning.n_used == fixed.n_used == len(TIMES)
    error = lambda result: np.sqrt(np.mean((result.image[region] - truth[region])**2))
    return error(turning), error(fixed)


def test_spokes_turning_with_the_planet_stay_sharp(monkeypatch):
    turning, fixed = errors(monkeypatch, spokes)
    assert turning < .75*fixed


def test_uniform_rings_are_preserved(monkeypatch):
    # Turning only changes where samples land; bilinear resampling costs about the same.
    turning, fixed = errors(monkeypatch, uniform)
    assert turning < 1.25*fixed


def test_motion_weight_fades_at_the_globe_its_shadow_and_beyond_the_rings():
    model = SaturnSceneModel(globe(), rings(sun_lon_rad=.5, sun_lat_rad=LATITUDE))
    x, y = detector_xy_grids(SIZE, SIZE)
    sx, sy = detector_to_sky(x, y, CENTRE, CENTRE)
    weight = model.ring_motion_weight(sx, sy)
    info = classify_layers(sx, sy, globe(), model.rings, REFERENCE)
    assert np.all(weight[info['on_globe']] == 0) and np.all(weight[info['globe_shadow']] == 0)
    assert np.all(weight[np.hypot(sx, sy) > OUTER + 13] == 0)
    assert weight.max() == 1


@pytest.mark.parametrize('dt', [-2., 2.5])
def test_distant_frames_do_not_fold_rings_at_the_limb(dt):
    # A frame minutes from the output epoch moves ring points several px (here 7-10 in plane);
    # a short fade at the globe's silhouette compressed them into a seam.
    model = SaturnSceneModel(globe(), rings())
    x, y = detector_xy_grids(SIZE, SIZE)
    dx, dy, _ = model.src_to_ref(x, y, FramePose(REFERENCE + dt, 0., CENTRE, CENTRE), FramePose(REFERENCE, 0., CENTRE, CENTRE))
    det = np.gradient(dx, axis=1)*np.gradient(dy, axis=0) - np.gradient(dx, axis=0)*np.gradient(dy, axis=1)
    weight = model.ring_motion_weight(*detector_to_sky(x, y, CENTRE, CENTRE))
    fade = (weight > 0) & (weight < 1)
    assert np.hypot(dx - x, dy - y)[weight == 1].max() > 2.5
    assert np.all(np.abs(det[fade] - 1) < .3)


def test_rings_turn_the_same_way_as_the_globe():
    model = SaturnSceneModel(globe(), rings())
    src, ref = FramePose(0., 0., CENTRE, CENTRE), FramePose(1., 0., CENTRE, CENTRE)
    dx, _, _ = model.src_to_ref(np.array([CENTRE]), np.array([CENTRE]), src, ref)
    x, y = detector_xy_grids(SIZE, SIZE)
    info = classify_layers(*detector_to_sky(x, y, CENTRE, CENTRE), globe(), model.rings, 0.)
    moved, _, _ = model.src_to_ref(x, y, src, ref)
    # Away from the ansae the motion is mostly sideways: compare its horizontal sense.
    weight = model.ring_motion_weight(*detector_to_sky(x, y, CENTRE, CENTRE))
    side = (weight > .99) & (np.abs(x - CENTRE) < .7*OUTER)
    near, far = info['near_ring'] & side, info['far_ring'] & side
    assert near.any() and far.any()
    centre = np.sign(dx[0] - CENTRE)
    assert np.all(np.sign((moved - x)[near]) == centre)
    assert np.all(np.sign((moved - x)[far]) == -centre)


def test_field_rate_fit_ignores_insignificant_rotation():
    times = np.array([88., 178., 315., 355.])
    rate, fit = field_rate_fit(times, np.array([.0152, 0., .0108, .0094]))   # 2026-10-21 12:10.4 samples
    assert rate == 0. and 'applied' in fit
    rate, _ = field_rate_fit(times, 4e-5*times + np.array([1e-4, -1e-4, 1e-4, -1e-4]))
    assert rate == pytest.approx(4e-5, rel=.05)
    rate, _ = field_rate_fit(times[:2], np.array([0., .009]))
    assert rate == pytest.approx(1e-4)


try:
    import torch
    has_cuda = torch.cuda.is_available()
except ImportError:
    has_cuda = False
cuda = pytest.mark.skipif(not has_cuda, reason='CUDA device unavailable')


@cuda
@pytest.mark.parametrize('fused', [True, False])
@pytest.mark.parametrize('mode', ['saturn', 'combined'])
def test_cuda_motion_and_gain_match_cpu(fused, mode):
    from planetrecon.backends.torch_globe import TorchGlobeWarp
    from planetrecon.backends.torch_saturn import TorchSaturnWarp
    from planetrecon.geometry.model import OblateGlobeModel, render_observed
    if mode == 'saturn':
        model = SaturnSceneModel(globe(), rings(sun_lon_rad=.5, sun_lat_rad=LATITUDE), ld_power=.7,
                                 field_angle0_rad=.1)
        warp = TorchSaturnWarp((SIZE, SIZE), model)
    else:
        model = OblateGlobeModel(globe(), ld_power=.7)
        warp = TorchGlobeWarp((SIZE, SIZE), model)
    if not fused:
        warp.fused = None
    srcs = [FramePose(0., .02, 100.3, 99.6), FramePose(3., -.01, 99.2, 100.4)]
    ref = FramePose(REFERENCE, .1, 100., 100.)
    x, y = detector_xy_grids(SIZE, SIZE)
    jitter = np.random.default_rng(3).uniform(-.4, .4, (2, SIZE, SIZE))
    def check(actual, expected):
        np.testing.assert_array_equal(actual[2].cpu(), expected[2])
        for a, b in zip(actual[:2], expected[:2]):
            np.testing.assert_allclose(a.cpu(), b, atol=2e-11, rtol=0)
        np.testing.assert_allclose(actual[3].cpu(), expected[3], atol=1e-12, rtol=0)
    batch = warp.map(srcs, ref, gain=True)
    for k, src in enumerate(srcs):
        expected = model.src_to_ref(x, y, src, ref, gain=True)
        check(warp.map(src, ref, gain=True), expected)
        check([a[k] for a in batch], expected)
        explicit = warp.map(src, ref, x=warp.tensor(x + jitter[0]), y=warp.tensor(y + jitter[1]), gain=True)
        check(explicit, model.src_to_ref(x + jitter[0], y + jitter[1], src, ref, gain=True))
        image = np.random.default_rng(k).normal(size=(SIZE, SIZE, 3))
        np.testing.assert_allclose(warp.render(image, src, ref), render_observed(image, model, src, ref),
                                   atol=1e-10, rtol=0)
