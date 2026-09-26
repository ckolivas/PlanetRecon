"""Scientific controls for the offline split-stack diagnostic."""
import numpy as np
from scipy.ndimage import fourier_shift, gaussian_filter

from tools.stack_repeatability import align_pair, analyse_pair


def test_identical_halves_have_unit_agreement_and_zero_difference():
    image = 100 + np.random.default_rng(935).normal(size=(404, 696))
    result = analyse_pair(image, image)
    for region in result.values():
        for band in region['bands']:
            assert abs(band['correlation'] - 1) < 1e-12
            assert abs(band['difference_to_mean_power']) < 1e-12


def test_independent_noise_has_little_high_frequency_agreement():
    rng = np.random.default_rng(935)
    first, second = 100 + rng.normal(size=(2, 404, 696))
    result = analyse_pair(first, second)
    for region in result.values():
        for band in region['bands'][2:]:
            assert abs(band['correlation']) < .15


def test_translation_estimated_from_proxies_preserves_image_detail():
    rng = np.random.default_rng(935)
    y, x = np.indices((404, 696))
    scene = np.exp(-((y-202)**2 + (x-348)**2)/4000) * (
        100 + gaussian_filter(rng.normal(size=y.shape), 1)*10)
    offset = np.array([1.25, -.3])
    moved = np.fft.ifft2(fourier_shift(np.fft.fft2(scene), offset)).real
    restored, result = align_pair(scene, moved)
    np.testing.assert_allclose(result['shift_yx'], -offset, atol=.02, rtol=0)
    assert result['proxy_rms_after'] < result['proxy_rms_before']/20
    assert np.sqrt(np.mean((restored-scene)**2)) < .001
