"""Optional transparency compensation of preprocessing quality."""
from dataclasses import replace

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr

from planetrecon.io.ser import SERSource, write_ser
from planetrecon.pipeline.preprocess import (FrameSelection, LAPLACIAN_NOISE_GAIN, N_MEASUREMENTS, measure_frame,
                                             noise_residual, relative_brightness, transparency_compensated)
from planetrecon.pipeline.preprocess_cache import (effective_selection, load_cache, preprocess_source, quality_plot_data,
                                                   save_cache, selection_digest)
from planetrecon.pipeline.baseline import stack_source
from planetrecon.reconstruction import ReconstructionConfig


def config(**kw):
    return ReconstructionConfig(device='cpu', threads=2, batch_frames=16, **kw)


def scene(shape=(96, 112)):
    y, x = np.indices(shape)
    disc = ((x - shape[1] / 2) / 34) ** 2 + ((y - shape[0] / 2) / 30) ** 2 < 1
    texture = 1 + .25 * np.sin(y / 2.1) + .15 * np.cos((x + y) / 3.3)
    return disc * texture


def frames(n=96, seed=3, cloudy=True, seeing=(2., 5.)):
    """Faint sky plus a soft-limbed disc of 30 e- through seeing and transparency, with photon noise.

    Photon noise is then about 40% of the quality score, as in real captures.
    """
    rng = np.random.default_rng(seed)
    blur = rng.uniform(*seeing, n)
    transparency = np.where(np.arange(n) % 2 & cloudy, rng.uniform(.45, .8, n), 1.)
    base = gaussian_filter(scene(), 1.5)
    out = [rng.poisson(1 + 30 * g * gaussian_filter(base, b)).astype(float) for b, g in zip(blur, transparency)]
    return np.stack(out), blur, transparency


def selection_of(measurements):
    return FrameSelection(np.ones(len(measurements), bool), measurements,
                          {'rejected_indices_by_reason': {k: [] for k in ('invalid', 'saturated', 'no_target',
                                                                           'clipped_target')},
                           'n_measured': len(measurements)})


def test_compensated_quality_ranks_by_seeing_not_transparency():
    stack, blur, transparency = frames()
    m = np.array([measure_frame(f, 'mono')[:-1] for f in stack])
    assert m.shape[1] == N_MEASUREMENTS
    raw = spearmanr(-blur, m[:, 0]).statistic
    compensated = transparency_compensated(selection_of(m))
    fixed = spearmanr(-blur, compensated.measurements[:, 0]).statistic
    assert fixed > .95 and fixed > raw + .3
    # Without the noise term, brightness normalisation alone favours dim frames.
    dim = transparency < 1
    contrast = m[:, 4] - m[:, 5]
    normalised = m[:, 0] / contrast ** 2
    def bias(q):  # Dim frames' scores against bright frames at matching seeing.
        trend = np.polyfit(blur[~dim], np.log(q[~dim]), 2)
        return float(np.median(np.log(q[dim]) - np.polyval(trend, blur[dim])))
    assert abs(bias(compensated.measurements[:, 0])) < .05 and bias(normalised) > .15
    # Brightness relative to the median follows the transparency.
    assert spearmanr(relative_brightness(selection_of(m)), transparency).statistic > .8


def test_noise_measurement_matches_the_quality_filter_on_white_noise():
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 3., (200, 200))
    residual = noise_residual(noise)
    assert np.mean(residual ** 2) / 36 == pytest.approx(9., rel=.03)
    from scipy.ndimage import laplace
    energy = np.mean(laplace(gaussian_filter(noise, 1., mode='nearest'), mode='nearest')[8:-8, 8:-8] ** 2)
    assert energy == pytest.approx(9. * LAPLACIAN_NOISE_GAIN, rel=.05)


def test_compensation_rescreens_a_view_and_leaves_the_cache(tmp_path):
    stack, *_ = frames(64, cloudy=False, seeing=(3., 3.6))
    # A typical frame through cloud: screened out on raw quality only.
    stack[10] = np.random.default_rng(9).poisson(1 + 30 * .6 * gaussian_filter(gaussian_filter(scene(), 1.5), 3.3))
    path = write_ser(tmp_path / 'cloud.ser', np.clip(stack, 0, 65535).astype('u2'))
    with SERSource(path) as source:
        selected = preprocess_source(source, config())
        assert selected.measurements.shape == (64, N_MEASUREMENTS)
        assert 10 in selected.summary['rejected_indices_by_reason']['low_quality']
        loaded, report = load_cache(source, config())
        view, compensated = effective_selection(loaded, report, config(transparency_compensation=True))
        assert view.accepted[10]
        assert compensated['accepted'] == int(view.accepted.sum())
        assert compensated['quality'] == len(view.summary['rejected_indices_by_reason']['low_quality'])
        assert compensated['transparency_compensation']['noise_model']['slope'] > 0
        assert view.digest == loaded.digest == selection_digest(loaded)
        np.testing.assert_array_equal(loaded.measurements, selected.measurements)
        assert effective_selection(loaded, report, config()) == (loaded, report)
        plot = quality_plot_data(view)
        assert plot['brightness'][10] == pytest.approx(.6, abs=.05)
        assert view.measurements[10, 0] == pytest.approx(np.median(view.measurements[view.accepted, 0]), rel=.15)
        # The stack ranks and screens with the same view.
        result = stack_source(source, config(transparency_compensation=True, stack_percent=100))
        assert result.provenance['preprocessing_cache']['accepted'] == compensated['accepted']


def test_caches_without_brightness_measurements_need_preprocessing_only_for_compensation(tmp_path):
    stack, *_ = frames(16)
    path = write_ser(tmp_path / 'old.ser', np.clip(stack, 0, 65535).astype('u2'))
    with SERSource(path) as source:
        selected = preprocess_source(source, config())
        old = replace(selected, measurements=selected.measurements[:, :4].copy())
        old.digest = selection_digest(old)
        save_cache(tmp_path / 'old.ser.planetrecon-preprocess.npz', old, source, config())
        loaded, report = load_cache(source, config())
        assert report['status'] == 'ready' and loaded.measurements.shape == (16, 4)
        assert quality_plot_data(loaded)['brightness'] is None
        missing, report = load_cache(source, config(transparency_compensation=True))
        assert missing is None and report['status'] == 'stale'
        assert 'Preprocess' in report['reason']


def test_thick_cloud_is_not_compensated_beyond_half_brightness():
    stack, blur, _ = frames(64, cloudy=False, seeing=(3., 3.6))
    rng = np.random.default_rng(5)
    base = gaussian_filter(scene(), 1.5)
    # Sharper than every clear frame, through cloud passing a fifth of the light.
    stack[:8] = [rng.poisson(1 + 30 * .2 * gaussian_filter(base, 2.4)) for _ in range(8)]
    m = np.array([measure_frame(f, 'mono')[:-1] for f in stack])
    view = transparency_compensated(selection_of(m))
    assert np.all(m[:8, 4] - m[:8, 5] < .3 * np.median(m[8:, 4] - m[8:, 5]))
    assert np.median(view.measurements[:8, 0]) < np.median(view.measurements[8:, 0])


def test_option_must_be_boolean():
    with pytest.raises(ValueError, match='transparency_compensation'):
        config(transparency_compensation=1)
