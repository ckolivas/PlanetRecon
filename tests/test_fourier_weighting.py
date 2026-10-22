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
        ordinary = stack_source(source, config())
        fourier = stack_source(source, config(fourier_weighting=True))
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


def test_fourier_weighting_requires_alignment_points():
    with pytest.raises(ValueError, match='alignment points'):
        ReconstructionConfig(fourier_weighting=True)
    for value in (0., 9., float('nan'), '1'):
        with pytest.raises(ValueError, match='fourier_power'):
            config(fourier_weighting=True, fourier_power=value)


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
