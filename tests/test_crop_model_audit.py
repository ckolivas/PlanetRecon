import numpy as np

from tools.audit_crop_model import regional_errors


def test_boundary_control_distinguishes_interior_model_error_from_noise():
    expected = np.ones((3, 128, 128))
    model = expected.copy()
    model[:, :8] += 4.
    observed = expected+2.
    errors = regional_errors(model, expected, observed, np.full_like(expected, 4.), np.ones((128, 128)))
    assert errors['whole_crop']['model_mean_standardized_square'] == .25
    assert all(r['noise_mean_standardized_square'] == 1. for r in errors.values())
    assert errors['interior_border_8']['model_mean_standardized_square'] == 0.
    model[:, 32:96, 32:96] += 2.
    errors = regional_errors(model, expected, observed, np.full_like(expected, 4.), np.ones((128, 128)))
    assert errors['interior_border_32']['model_mean_standardized_square'] == 1.
