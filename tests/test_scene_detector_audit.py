import numpy as np
import pytest
from tools.audit_scene_detector import stack_statistics


def test_stack_bias_detects_correlated_subnoise_residuals():
    var = np.ones((100, 2, 2))
    result = stack_statistics(np.full_like(var, .2), var)
    assert result['per_frame_mean_standardized_square'] == pytest.approx(.04)
    assert result['stack_mean_standardized_square'] == pytest.approx(4.)
    residual = np.full_like(var, .2)
    residual[1::2] *= -1
    assert stack_statistics(residual, var)['stack_mean_standardized_square'] == 0.


def test_bad_variance_rejected():
    with pytest.raises(ValueError):
        stack_statistics(np.zeros((2, 2)), np.zeros((2, 2)))
