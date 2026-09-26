"""Scientific controls for the standalone local-selection experiment."""
import numpy as np
import pytest
from scipy.ndimage import gaussian_filter, laplace

from tools.patch_stacking_experiment import top_masks, combine_patches, finalize_patches


def test_rank_per_ap_excludes_missing_data_and_breaks_ties_in_capture_order():
    quality = np.array([[4., np.nan], [4., 2.], [1., 9.], [np.inf, 8.]])
    mask = top_masks(quality, 2)
    np.testing.assert_array_equal(np.flatnonzero(mask[:, 0]), [0, 1])
    np.testing.assert_array_equal(np.flatnonzero(mask[:, 1]), [2, 3])
    with pytest.raises(ValueError, match='Insufficient'):
        top_masks(quality, 4)


def test_overlaps_preserve_absolute_brightness_and_uncovered_fallback():
    # Deliberately unequal exposures and spatially varying support: neither
    # the patch division nor the overlap should change a constant scene.
    weight = np.arange(1, 51, dtype=float).reshape(2, 5, 5)
    weight[0, 2, 2] = 0
    mean = finalize_patches(weight*137.5, weight)
    support = (weight > 0).astype(float)
    y, x = np.indices((5, 5))-2
    footprint = np.maximum(0., 1.-np.hypot(x, y)/3.)
    fallback = np.full((13, 15), 22.)
    result, coverage = combine_patches(mean, support, [[5, 5], [5, 7]], footprint, fallback)
    np.testing.assert_allclose(result[coverage >= 1], 137.5, atol=1e-12)
    edge = (coverage > 0)&(coverage < 1)
    np.testing.assert_allclose(result[edge], coverage[edge]*137.5+(1.-coverage[edge])*22., atol=1e-12)
    uniform, _ = combine_patches(mean, support, [[5, 5], [5, 7]], footprint, np.full_like(fallback,137.5))
    np.testing.assert_allclose(uniform, 137.5, atol=1e-12)
    np.testing.assert_array_equal(result[coverage == 0], 22.)


def test_local_selection_recovers_complementary_sharp_regions():
    # Known truth with alternating atmospheric blur: global selection must
    # compromise, whereas different local subsets can recover both regions.
    rng = np.random.default_rng(482)
    truth = 100.+gaussian_filter(rng.normal(size=(65, 65)), 1.)*15.
    blurred = gaussian_filter(truth, 3.)
    frames = np.empty((20, 2, 65, 65))
    for i in range(20):
        frames[i] = [truth, blurred] if i < 10 else [blurred, truth]
    quality = np.array([[np.mean(laplace(gaussian_filter(p, 2.))**2) for p in f] for f in frames])
    local = top_masks(quality, 10)
    global_mask = top_masks(quality.mean(axis=1)[:, None], 10)[:, 0]
    local_means = np.array([frames[local[:, k], k].mean(axis=0) for k in range(2)])
    global_means = frames[global_mask].mean(axis=0)
    assert np.mean((local_means-truth)**2) < 1e-8
    assert np.mean((global_means-truth)**2) > 1.


def test_shared_translation_patch_stack_equals_whole_frame_pull():
    import torch
    from planetrecon.backends.torch_circular import sample
    rng = np.random.default_rng(549)
    raw = torch.as_tensor(rng.normal(size=(25, 31)), dtype=torch.float64)
    yy, xx = torch.meshgrid(torch.arange(25), torch.arange(31), indexing='ij')
    direct = sample(raw, yy+.37, xx-.21).numpy()
    centres = np.array([[10, 11], [10, 16]])
    py, px = torch.meshgrid(torch.arange(-4, 5), torch.arange(-4, 5), indexing='ij')
    patches = np.array([sample(raw, y+py+.37, x+px-.21).numpy() for y, x in centres])
    combined, weight = combine_patches(patches, np.ones_like(patches), centres,
                                      np.ones((9, 9)), np.zeros((25, 31)))
    np.testing.assert_allclose(combined[weight > 0], direct[weight > 0], atol=1e-14)


def test_completed_patch_registration_corrects_known_translation():
    from scipy.ndimage import shift
    from planetrecon.backends.cpu import CPUBackend
    from tools.patch_stacking_experiment import register_patches
    y, x = np.indices((65, 65))
    truth = (100*np.exp(-((x-30)**2+(y-31)**2)/70.) +
             35*np.exp(-((x-41)**2+(y-39)**2)/25.))
    displaced = shift(truth, (.7, -1.2), order=3)
    result, support, offsets = register_patches(displaced[None], np.ones((1, 65, 65)),
                                                [[32, 32]], truth, CPUBackend(threads=2))
    np.testing.assert_allclose(offsets[0], [-1.2, .7], atol=.1)
    roi = np.s_[8:-8, 8:-8]
    assert np.mean((result[0][roi]-truth[roi])**2) < .02*np.mean((displaced[roi]-truth[roi])**2)


def test_two_stage_resampling_preserves_brightness_with_partial_edge_support():
    import torch
    from planetrecon.backends.torch_circular import sample
    y, x = torch.meshgrid(torch.arange(17), torch.arange(19), indexing='ij')
    support = torch.ones((17, 19), dtype=torch.float64)
    signal = support*83.2
    for dx, dy in [(1.3, -.7), (-.2, .4)]:
        signal = sample(signal, y+dy, x+dx)
        support = sample(support, y+dy, x+dx)
    valid = support > 0
    np.testing.assert_allclose((signal[valid]/support[valid]).numpy(), 83.2, atol=1e-12)
