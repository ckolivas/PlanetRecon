"""Frequency-domain lucky imaging from local-quality rank bins."""
from dataclasses import replace

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter, shift as ndshift

from planetrecon.io.ser import NAME_TO_COLOR, SERSource, write_ser
from planetrecon.pipeline import fourier_weights
from planetrecon.pipeline.baseline import stack_source
from planetrecon.pipeline.preprocess_cache import preprocess_source
from planetrecon.reconstruction import ReconstructionConfig

SHAPE = (128, 160)


def truth():
    yy, xx = np.indices(SHAPE)
    disc = gaussian_filter(((((xx-80)/58)**2 + ((yy-64)/46)**2) < 1).astype(float), 1.)
    texture = gaussian_filter(np.random.default_rng(31).normal(size=SHAPE), .9)
    return disc*(2000 + 900*texture/texture.std())


def capture(path, colour='mono', n=320):
    """Frames whose blur varies widely, with small shifts and noise."""
    rng = np.random.default_rng(5)
    scene = truth()
    frames = []
    for blur in rng.uniform(.4, 2.6, n):
        frame = ndshift(gaussian_filter(scene, blur), rng.uniform(-1.5, 1.5, 2), order=3, mode='nearest')
        # Per-frame noise comparable to real planetary video (signal-to-noise near 10).
        frames.append(frame + rng.normal(scale=260., size=SHAPE))
    frames = np.clip(np.asarray(frames) + 1000, 0, 65535)
    if colour == 'RGGB':
        frames[:, ::2, ::2] *= .8
        frames[:, 1::2, 1::2] *= .6
    return write_ser(path, frames.astype('u2'), color_id=NAME_TO_COLOR[colour])


def config(**kwargs):
    return ReconstructionConfig(**{'device': 'cpu', 'threads': 2, 'batch_frames': 16, 'local_alignment': True,
                                   'ap_size': 32, 'stack_percent': 100, **kwargs})


def test_band_filters_partition_unity():
    for shape in ((466, 1124), (63, 64), (5, 7)):
        filters = fourier_weights.band_filters(shape)
        np.testing.assert_allclose(filters.sum(0), 1., atol=1e-15)
        assert filters.min() >= 0


def test_equal_weights_reproduce_the_ordinary_stack():
    rng = np.random.default_rng(2)
    bin_weight = rng.uniform(.5, 2., (8, 40, 50, 3))
    bin_accum = bin_weight*rng.normal(100, 10, (8, 40, 50, 3))
    bin_weight[:, :3] = 0.
    ordinary = np.divide(bin_accum.sum(0), bin_weight.sum(0), out=np.zeros((40, 50, 3)),
                         where=bin_weight.sum(0) > 0)
    W = np.full((4, len(fourier_weights.BAND_CENTRES)), 3.)
    np.testing.assert_allclose(fourier_weights.combine(bin_accum, bin_weight, W), ordinary, rtol=0, atol=1e-10)
    first = fourier_weights.combine(bin_accum, bin_weight, W, half=0)
    np.testing.assert_allclose(first, np.divide(bin_accum[::2].sum(0), bin_weight[::2].sum(0),
                               out=np.zeros((40, 50, 3)), where=bin_weight[::2].sum(0) > 0), atol=1e-10)


def test_monotone_fit_pools_violators():
    np.testing.assert_allclose(fourier_weights.monotone([3., 1., 2., 0.]), [3., 1.5, 1.5, 0.])
    np.testing.assert_allclose(fourier_weights.monotone([1., 2.]), [1.5, 1.5])


def ring(c):
    return int(round(c / .01))


def coherent_gain(image, reference, low, high):
    """Signal of ``image`` coherent with ``reference`` relative to the reference's, over a ring range."""
    taper = fourier_weights.window(reference)
    spectrum = fourier_weights._spectra(taper)
    a, b = spectrum(image), spectrum(reference)
    filters = fourier_weights.band_filters(taper.shape)[ring(low):ring(high)+1].sum(0)
    return np.sum(filters*np.real(a*np.conj(b))) / np.sum(filters*np.abs(b)**2)


@pytest.mark.parametrize('colour', ['mono', 'RGGB'])
def test_fourier_weighting_concentrates_fine_detail_on_the_sharpest_frames(tmp_path, colour):
    path = capture(tmp_path/'blur.ser', colour)
    with SERSource(path) as source:
        preprocess_source(source, config())
        ordinary = stack_source(source, config(fourier_weighting=False))
        # p = 1 is the matched filter: the best measured signal-to-noise at every ring.
        fourier = stack_source(source, config(fourier_weighting=True, fourier_power=1.))
        sharper = stack_source(source, config())
    detail = fourier.provenance['fourier_weighting']
    T = np.asarray(detail['transfer'], dtype=float)
    # Coarse rings transfer equally; fine rings come from the best-ranked bins.
    assert abs(T[:, ring(.02)] - 1).max() < .1
    assert T[0, ring(.25)] > 3*max(T[-1, ring(.25)], 1e-3)
    W = np.asarray(detail['weights'])
    # Fine rings draw on the best bins; coarse rings use every bin, without favouring the best.
    assert W[0, ring(.25)] > 10*W[-1, ring(.25)]
    assert 0 < W[0, ring(.02)] <= W[-1, ring(.02)]
    # More fine detail than the ordinary stack, at better signal-to-noise.
    assert coherent_gain(fourier.image, ordinary.image, .2, .3) > 1.3
    snr, plain = np.asarray(detail['ring_snr']), np.asarray(detail['ordinary_ring_snr'])
    assert snr[ring(.25):ring(.35)].sum() > 1.25*plain[ring(.25):ring(.35)].sum()
    # The default power concentrates fine detail further on the sharpest frames.
    assert coherent_gain(sharper.image, ordinary.image, .2, .3) > coherent_gain(fourier.image, ordinary.image, .2, .3)
    np.testing.assert_allclose(fourier.coverage, ordinary.coverage, rtol=1e-12)
    assert 'fourier_weighting' not in ordinary.provenance


def test_fourier_checkpoint_resumes_and_binds_settings(tmp_path):
    path = capture(tmp_path/'resume.ser', 'RGGB', n=64)
    cfg = config(fourier_weighting=True)
    with SERSource(path) as source:
        preprocess_source(source, cfg)
        whole = stack_source(source, cfg)
        stop = False

        def event(result, info):
            nonlocal stop
            stop = stop or (result.stage not in ('setup', 'cache_ready') and info['n_processed'] >= 20)
        checkpoint = tmp_path/'state.npz'
        partial = stack_source(source, cfg, on_event=event, should_cancel=lambda: stop, state_checkpoint=checkpoint)
        assert partial.incomplete
        resumed = stack_source(source, cfg, resume_from=checkpoint)
        np.testing.assert_array_equal(resumed.image, whole.image)
        for other in (replace(cfg, fourier_power=2.), replace(cfg, fourier_weighting=False)):
            with pytest.raises(ValueError, match='identity|configuration'):
                stack_source(source, other, resume_from=checkpoint)


def test_fourier_weighting_is_the_default_for_every_alignment_point_stack():
    assert config().fourier_active and config().fourier_power == 4.
    assert not ReconstructionConfig().fourier_active  # Global alignment only.
    assert config(geometry_mode='surface').fourier_active  # Motion models too.
    assert not config(fourier_weighting=False).fourier_active
    for value in (0., 9., float('nan'), '1'):
        with pytest.raises(ValueError, match='fourier_power'):
            config(fourier_weighting=True, fourier_power=value)


def test_settings_saved_before_fourier_weighting_take_the_full_default(monkeypatch):
    monkeypatch.setenv('QT_QPA_PLATFORM', 'offscreen')
    from PySide6.QtWidgets import QApplication
    from planetrecon.gui.controls import ConfigControls
    app = QApplication.instance() or QApplication([])
    controls = ConfigControls(ReconstructionConfig(local_alignment=True))
    try:
        controls.restore_settings({'fields': {'stack_percent': 50, 'ap_size': 48}})
        assert controls.fields['stack_percent'].value() == 100 and controls.fields['ap_size'].value() == 48
        controls.restore_settings({'fields': {'stack_percent': 50, 'fourier_weighting': False}})
        assert controls.fields['stack_percent'].value() == 50
        assert not controls.configuration().fourier_weighting and not controls.fields['fourier_power'].isEnabled()
    finally:
        controls.close()


@pytest.mark.hardware
def test_cuda_fourier_weighting_matches_cpu(tmp_path):
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    path = capture(tmp_path/'cuda.ser', 'RGGB', n=160)
    with SERSource(path) as source:
        preprocess_source(source, config())
        cpu = stack_source(source, config(fourier_weighting=True))
        gpu = stack_source(source, config(fourier_weighting=True, device='gpu'))
    assert gpu.backend == 'cuda'
    np.testing.assert_allclose(gpu.image, cpu.image, rtol=1e-6, atol=1e-6*np.abs(cpu.image).max())


GLOBE = (128, 160)


def rotating_globe(path, colour='mono', n=320, rate=.002):
    """A rotating globe with fine surface texture; per-frame blur, small drift and noise."""
    from scipy.ndimage import map_coordinates
    rng = np.random.default_rng(11)
    surface = gaussian_filter(rng.normal(size=(256, 512)), 1.2, mode='wrap')
    surface = surface / surface.std()
    y, x = np.indices(GLOBE, dtype=float)
    frames = []
    for i, blur in enumerate(rng.uniform(.4, 2.4, n)):
        sx, sy = x + .5 - 80, 64 - (y + .5)
        inside = sx*sx + sy*sy < 46**2
        z = np.sqrt(np.maximum(46**2 - sx*sx - sy*sy, 0.))
        lon = (np.arctan2(sx, z) + rate*i) % (2*np.pi)
        lat = np.arcsin(np.clip(sy/46, -1, 1))
        texture = map_coordinates(surface, [(lat/np.pi + .5)*255, lon/(2*np.pi)*512], order=1, mode='grid-wrap')
        scene = np.where(inside, 2000 + 900*texture, 0.)
        frame = ndshift(gaussian_filter(scene, blur), rng.uniform(-1, 1, 2), order=1, mode='nearest')
        frames.append(frame + rng.normal(scale=260., size=GLOBE))
    frames = np.clip(np.asarray(frames) + 1000, 0, 65535)
    if colour == 'RGGB':
        frames[:, ::2, ::2] *= .8
        frames[:, 1::2, 1::2] *= .6
    return write_ser(path, frames.astype('u2'), color_id=NAME_TO_COLOR[colour],
                     timestamps=np.arange(n)*10_000_000)  # One second apart.


def globe_config(**kwargs):
    return config(**{'geometry_mode': 'surface', 'field_center_x': 80., 'field_center_y': 64.,
                     'equatorial_radius_px': 46., 'surface_rate_rad_s': .002, 'sub_obs_lat_rad': 0.,
                     'motion_reference': 'best', 'batch_frames': 24, **kwargs})


@pytest.mark.parametrize('colour', ['mono', 'RGGB'])
def test_fourier_weighting_with_a_motion_model(tmp_path, monkeypatch, colour):
    path = rotating_globe(tmp_path/'globe.ser', colour)
    with SERSource(path) as source:
        preprocess_source(source, globe_config())
        ordinary = stack_source(source, globe_config(fourier_weighting=False))
        matched = stack_source(source, globe_config(fourier_power=1.))
        sharper = stack_source(source, globe_config())
        # Equal weights must reproduce the ordinary motion stack: the rank bins
        # hold exactly the samples the ordinary accumulators receive.
        monkeypatch.setattr(fourier_weights, 'weights', lambda T, V, D, power: np.ones_like(T))
        equal = stack_source(source, globe_config())
    assert ordinary.provenance['geometry']['surface_rate_rad_s'] == .002
    np.testing.assert_allclose(equal.image, ordinary.image, rtol=0, atol=1e-9*np.abs(ordinary.image).max())
    detail = matched.provenance['fourier_weighting']
    T = np.asarray(detail['transfer'], dtype=float)
    assert T[0, ring(.2)] > 2*max(T[-1, ring(.2)], 1e-3)
    # This capture is limited by registration more than noise at fine scales,
    # so p = 1 cannot gain much signal-to-noise there, but must not lose it.
    snr, plain = np.asarray(detail['ring_snr']), np.asarray(detail['ordinary_ring_snr'])
    assert snr[ring(.15):ring(.3)].sum() > .95*plain[ring(.15):ring(.3)].sum()
    gains = [coherent_gain(r.image, ordinary.image, .15, .3) for r in (matched, sharper)]
    assert 1. < gains[0] < gains[1]
    assert 'Fourier power 4' in sharper.provenance['stack_quality']['weighting']


def test_motion_model_fourier_checkpoint_resumes(tmp_path):
    path = rotating_globe(tmp_path/'resume.ser', 'RGGB', n=96)
    cfg = globe_config()
    with SERSource(path) as source:
        preprocess_source(source, cfg)
        whole = stack_source(source, cfg)
        stop = False

        def event(result, info):
            nonlocal stop
            stop = stop or (result.stage not in ('setup', 'cache_ready') and info['n_processed'] >= 40)
        checkpoint = tmp_path/'state.npz'
        partial = stack_source(source, cfg, on_event=event, should_cancel=lambda: stop, state_checkpoint=checkpoint)
        assert partial.incomplete
        resumed = stack_source(source, cfg, resume_from=checkpoint)
        np.testing.assert_allclose(resumed.image, whole.image, rtol=0, atol=1e-9*np.abs(whole.image).max())
        with pytest.raises(ValueError, match='identity|configuration'):
            stack_source(source, replace(cfg, fourier_power=2.), resume_from=checkpoint)


@pytest.mark.hardware
@pytest.mark.parametrize('colour', ['mono', 'RGGB'])
def test_cuda_motion_model_fourier_matches_cpu(tmp_path, colour):
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    path = rotating_globe(tmp_path/'cuda.ser', colour, n=96)
    with SERSource(path) as source:
        preprocess_source(source, globe_config())
        cpu = stack_source(source, globe_config())
        gpu = stack_source(source, globe_config(device='gpu'))
    assert gpu.backend == 'cuda' and 'fourier_weighting' in gpu.provenance
    np.testing.assert_allclose(gpu.image, cpu.image, rtol=1e-5, atol=1e-5*np.abs(cpu.image).max())
