"""Ensure the regional acceptance gate exposes a local regression."""
import numpy as np

from tools.cfa_local_confirmation import compare_regions


def comparison_inputs():
    y, x = np.indices((96, 288), dtype=float)
    target = np.stack([30+np.sin(x/3)+np.cos(y/4), 40+np.sin((x+y)/4), 35+np.cos((x-y)/5)], axis=-1)
    noisy = target+np.random.default_rng(723).normal(0, .5, target.shape)
    return target, noisy, np.ones(target.shape, bool)


def test_candidate_with_less_noise_passes_each_region_and_uses_common_masks():
    target, ordinary, validity = comparison_inputs()
    validity[40, 45, 0] = False
    metrics, previews, mask = compare_regions(ordinary, target, target, validity)
    assert all(metrics[r]['passes_both_metrics'] for r in ('left', 'centre', 'right'))
    assert metrics['left']['common_pixels'] == 6399
    assert metrics['centre']['common_pixels'] == metrics['right']['common_pixels'] == 6400
    assert mask.sum() == 19199
    assert not mask[40, 45]
    assert not mask[:8].any() and not mask[:, 88:104].any()
    assert metrics['aggregate']['quadratic']['relative_rmse'] < 1e-12


def test_single_region_regression_fails_even_when_aggregate_improves():
    target, ordinary, validity = comparison_inputs()
    candidate = target.copy()
    candidate[:, :96] = target[:, :96]+1.1*(ordinary[:, :96]-target[:, :96])
    metrics, _, _ = compare_regions(ordinary, candidate, target, validity)
    assert not metrics['left']['passes_both_metrics']
    assert metrics['centre']['passes_both_metrics'] and metrics['right']['passes_both_metrics']
    assert metrics['aggregate']['quadratic']['relative_rmse'] < metrics['aggregate']['ordinary']['relative_rmse']
