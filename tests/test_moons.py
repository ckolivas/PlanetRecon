from dataclasses import replace

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter, shift as ndshift

from planetrecon.io.ser import SERSource, write_ser
from planetrecon.pipeline import moons
from planetrecon.pipeline.baseline import stack_source
from planetrecon.pipeline.preprocess_cache import preprocess_source
from planetrecon.reconstruction import ReconstructionConfig

H, W = 110, 280
N = 240
BRIGHT = (30., 150.)
FAINT = (74., 172.)


def config(**kwargs):
    return ReconstructionConfig(**{**dict(device='cpu', threads=2, batch_frames=16, local_alignment=False,
                                          frame_selection_mode='frame_count'), **kwargs})


def spot(y, x, amplitude, sigma=1.2):
    yy, xx = np.indices((H, W))
    return amplitude * np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2 * sigma ** 2))


def capture(tmp_path, *, tilt=1.5, faint=12., seed=3, kind='mono', hot=None):
    """Planet with texture; a bright and a faint moon, each tilted by its own seeing and drifting."""
    rng = np.random.default_rng(seed)
    yy, xx = np.indices((H, W))
    disc = ((xx - 50) / 22) ** 2 + ((yy - 48) / 20) ** 2 < 1
    planet = gaussian_filter(disc * (3000 + 600 * np.cos(yy / 2.5) + 300 * np.cos(xx / 3.)), 1.)
    times = np.arange(N) / 100.  # 100 frames per second
    drift = 6. * (times - times.mean()) / np.ptp(times)  # Bright moon moves 6 px in x over the capture.
    frames = []
    for k in range(N):
        g = rng.normal(0, 1.5, 2)
        scene = ndshift(planet, g, order=1)
        t = rng.normal(0, tilt, 2)
        scene = scene + spot(BRIGHT[0] + g[0] + t[0], BRIGHT[1] + drift[k] + g[1] + t[1], 400.)
        scene = scene + spot(FAINT[0] + g[0] + .7 * t[0], FAINT[1] + g[1] + .7 * t[1], faint)
        frame = scene + 100 + rng.normal(0, 20, scene.shape)
        if hot is not None:
            frame[hot] += 3000.
        frames.append(np.clip(frame, 0, 65535).astype('u2'))
    ticks = (times * 1e7).astype('<i8') + 638000000000000000
    frames = np.stack(frames)
    if kind == 'rgb':
        frames = np.stack([frames, frames, (frames * .8).astype('u2')], -1)
        return write_ser(tmp_path / 'moons.ser', frames, color_id=100, timestamps=ticks)
    if kind == 'bayer':
        frames[:, ::2, ::2] = (frames[:, ::2, ::2] * .8).astype('u2')
        return write_ser(tmp_path / 'moons.ser', frames, color_id=8, timestamps=ticks)
    return write_ser(tmp_path / 'moons.ser', frames, timestamps=ticks)


@pytest.fixture(scope='module')
def stacks(tmp_path_factory):
    path = capture(tmp_path_factory.mktemp('moons'))
    with SERSource(path) as source:
        preprocess_source(source, config())
        on = stack_source(source, config())
        off = stack_source(source, config(moon_stacking=False))
    return path, on, off


def peak(image, y, x):
    return float(image[int(y) - 4:int(y) + 5, int(x) - 4:int(x) + 5].max() - np.median(image[:12, 100:]))


def flux(image, y, x, r=12):
    patch = image[int(y) - r:int(y) + r + 1, int(x) - r:int(x) + r + 1]
    return float((patch - np.median(image[:12, 100:])).sum())


def test_bright_moon_follows_its_own_tilt_and_keeps_its_flux(stacks):
    _, on, off = stacks
    detail = on.provenance['moons']
    assert detail['enabled'] and detail['stacked'] == 2
    bright = min(detail['moons'], key=lambda m: abs(m['detected_at'][0] - BRIGHT[0]))
    assert bright['noise_measurement'].startswith('injected')
    assert bright['gain'] > .9 and 1.1 < bright['seeing_jitter_px'] < 1.9
    # Drift is 6 px per 2.39 s along x, measured relative to the planet.
    assert bright['drift_px_per_min'][1] == pytest.approx(6. / (N - 1) * 100 * 60, rel=.15)
    # Output position is the midpoint of the drift: where the planet-registered moon is centred.
    assert np.allclose(bright['output_position_px'], moons.centroid(off.image, *BRIGHT, 3., 14), atol=.3)
    assert peak(on.image, *BRIGHT) > 1.8 * peak(off.image, *BRIGHT)
    assert flux(on.image, *BRIGHT) == pytest.approx(flux(off.image, *BRIGHT), rel=.03)
    assert bright['stacked_fwhm_px'] < .7 * bright['planet_registered_fwhm_px']


def test_faint_moon_follows_only_the_share_of_its_tilt_above_its_noise(stacks):
    _, on, _ = stacks
    found = on.provenance['moons']['moons']
    faint = min(found, key=lambda m: abs(m['detected_at'][0] - FAINT[0]))
    bright = min(found, key=lambda m: abs(m['detected_at'][0] - BRIGHT[0]))
    # True tilt 0.7 x 1.5 = 1.05 px per axis, about as large as its centroid noise.
    assert .7 < faint['seeing_jitter_px'] < 1.4 and .6 < faint['centroid_noise_px'] < 1.6
    # The bright moon's measured tilt carries the shared part, free of the faint moon's noise;
    # only part of the rest follows the faint moon's own centroid.
    assert faint['shared_with_neighbours_px'] > .3
    assert faint['gain'] < .65 < bright['gain']


def test_only_the_moon_windows_change(stacks):
    _, on, off = stacks
    changed = np.argwhere(on.image != off.image)
    centres = np.array([m['output_position_px'] for m in on.provenance['moons']['moons']])
    distance = np.min(np.hypot(*(changed[:, None, :] - centres[None]).transpose(2, 0, 1)), axis=1)
    assert distance.max() <= np.sqrt(2) * moons.window_radius(max(
        m['planet_registered_fwhm_px'] for m in on.provenance['moons']['moons'])) + 1


def test_resume_continues_the_moon_sums(stacks, tmp_path):
    path, on, _ = stacks
    with SERSource(path) as source:
        stop = False

        def event(result, info):
            nonlocal stop
            if result.stage != 'setup' and info['n_processed'] >= 100:
                stop = True
        checkpoint = tmp_path / 'resume.npz'
        partial = stack_source(source, config(), should_cancel=lambda: stop, on_event=event,
                               state_checkpoint=checkpoint)
        assert partial.incomplete
        resumed = stack_source(source, config(), resume_from=checkpoint)
        np.testing.assert_allclose(resumed.image, on.image, rtol=1e-12, atol=1e-9)
        assert resumed.provenance['moons']['moons'] == on.provenance['moons']['moons']
        with pytest.raises(ValueError, match='identity|configuration|mismatch'):
            stack_source(source, config(moon_stacking=False), resume_from=checkpoint)


def test_sources_in_the_glow_or_by_alignment_points_keep_the_planet_registration():
    yy, xx = np.indices((H, W))
    planet = 3000. * ((((xx - 50) / 22) ** 2 + ((yy - 48) / 20) ** 2) < 1)
    image = gaussian_filter(planet, 6.) + 100 + spot(48, 78, 300., 2.) + spot(30, 150, 300., 2.) \
        + spot(70, 120, 300., 2.)
    image = image + np.random.default_rng(0).normal(0, .5, image.shape)
    found, excluded, blank = moons.detect(image, np.ones(image.shape, bool), points=[(66, 112)], point_reach=24)
    assert [(m['y'], m['x']) for m in found] == [(30, 150)]
    reasons = {(e['y'], e['x']): e['reason'] for e in excluded}
    assert reasons[(70, 120)] == 'within reach of an alignment point'
    assert not any(abs(m['y'] - 48) < 3 and abs(m['x'] - 78) < 3 for m in found)
    assert blank is not None


def test_centroid_finds_a_shifted_spot_and_rejects_empty_sky():
    rng = np.random.default_rng(1)
    image = spot(40.3, 60.7, 200.) + rng.normal(0, 1, (H, W))
    y, x = moons.centroid(image, 39., 62., 1.5, 9)
    assert abs(y - 40.3) < .05 and abs(x - 60.7) < .05
    assert moons.centroid(np.zeros((H, W)), 40., 60., 1.5, 9) is None
    assert moons.centroid(image, 3., 60., 1.5, 9) is None  # Window leaves the frame.


def test_cli_and_config_carry_the_option(capsys):
    from planetrecon.cli import main
    assert ReconstructionConfig().moon_stacking is True
    assert ReconstructionConfig.from_dict(replace(config(), moon_stacking=False).to_dict()).moon_stacking is False
    with pytest.raises(ValueError, match='moon_stacking'):
        ReconstructionConfig(moon_stacking=1)
    with pytest.raises(SystemExit) as stop:
        main(['stack', '--help'])
    assert stop.value.code == 0
    assert '--no-moon-stacking' in capsys.readouterr().out


def moon_gain(image, off, y, x):
    if image.ndim == 3:
        image, off = image.mean(-1), off.mean(-1)
    return peak(image, y, x) / peak(off, y, x)


@pytest.mark.parametrize('kind', ['rgb', 'bayer'])
def test_colour_captures_stack_their_moons(tmp_path, kind):
    with SERSource(capture(tmp_path, kind=kind)) as source:
        preprocess_source(source, config())
        on = stack_source(source, config())
        off = stack_source(source, config(moon_stacking=False))
    assert on.image.shape == off.image.shape == (H, W, 3)
    assert on.provenance['moons']['stacked'] >= 1
    assert moon_gain(on.image, off.image, *BRIGHT) > 1.6
    assert np.isfinite(on.image).all()


@pytest.mark.parametrize('device', ['cpu', 'gpu'])
def test_moons_with_alignment_points(tmp_path, device):
    if device == 'gpu':
        torch = pytest.importorskip('torch')
        if not torch.cuda.is_available():
            pytest.skip('CUDA unavailable')
    cfg = config(device=device, local_alignment=True, ap_size=32, stack_percent=50)
    with SERSource(capture(tmp_path)) as source:
        preprocess_source(source, cfg)
        on = stack_source(source, cfg)
        off = stack_source(source, replace(cfg, moon_stacking=False))
    assert on.provenance['local_alignment']['points'] > 0
    assert on.backend == ('cuda' if device == 'gpu' else 'cpu')
    detail = on.provenance['moons']
    bright = min(detail['moons'], key=lambda m: abs(m['detected_at'][0] - BRIGHT[0]))
    assert bright['gain'] > .9
    # Only the frames the global selection keeps contribute outside the alignment points.
    assert bright['frames_following_measured_tilt'] + bright['frames_on_drift_model'] <= N // 2 + 1
    assert moon_gain(on.image, off.image, *BRIGHT) > 1.6
    # Every planet feature keeps its alignment-point registration.
    assert not any(m['detected_at'][1] < 110 for m in detail['moons'])


def test_hot_pixel_fixed_on_the_detector_is_not_stacked_as_a_moon(tmp_path):
    with SERSource(capture(tmp_path, hot=(55, 230))) as source:
        preprocess_source(source, config())
        on = stack_source(source, config())
    detail = on.provenance['moons']
    assert not any(abs(m['detected_at'][0] - 55) < 3 and abs(m['detected_at'][1] - 230) < 3 for m in detail['moons'])
    assert any(d['reason'].startswith('fixed on the detector') for d in detail['too_faint'])
