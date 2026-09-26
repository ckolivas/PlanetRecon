import numpy as np
import pytest
from scipy.ndimage import gaussian_filter, map_coordinates

from tools.analyse_stack_transfer import coarse_displacement, fit_kernel, fit_translation


def test_recovers_crop_and_known_gaussian_on_separate_validation_region():
    truth = np.random.default_rng(4).uniform(10, 200, (60, 100))
    output = gaussian_filter(truth, 1., radius=3)[3:-3]
    np.testing.assert_array_equal(coarse_displacement(truth, output), [-3, 0])
    y, x = np.indices(output.shape)
    mask = (y > 5)&(y < 45)&(x > 5)&(x < 94)
    sy, sx = y[mask]+3, x[mask]
    train, test = sx < 50, sx >= 50
    result = fit_kernel(truth, output[mask], sy, sx, train, test)
    assert result['max_coefficient_difference_from_sigma1_gaussian'] < 1e-12
    assert result['held_out']['rms_adu'] < 1e-10


@pytest.mark.parametrize('dy,dx', [(0., 0.), (.27, -.19)])
def test_translation_model_preserves_exact_identity_and_code_scale(dy, dx):
    truth = np.random.default_rng(5).uniform(10, 200, (60, 100))
    y, x = np.indices(truth.shape)
    mask = (y > 5)&(y < 54)&(x > 5)&(x < 94)
    sy, sx = y[mask], x[mask]
    output = map_coordinates(truth, [sy+dy, sx+dx], order=3)*1.03+.4
    result = fit_translation(truth, output, sy, sx, sx < 50, sx >= 50)
    assert result['optimizer_success']
    assert result['held_out']['rms_adu'] < 1e-7
    np.testing.assert_allclose(result['parameters_dy_dx_gain_offset'], [dy, dx, 1.03, .4], atol=1e-7)
