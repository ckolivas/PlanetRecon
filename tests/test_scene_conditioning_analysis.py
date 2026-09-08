import numpy as np
import pytest
from tools.scene_conditioning_analysis import fixed_residual_limit


def test_limit_bounds_the_exact_energy_certificate_not_actual_error():
    h = np.diag([.01, 1., 4.]); ridge = .01; upper = 4.
    x = np.ones(3)*10; r = np.array([0., 0., 1.])
    raw = np.linalg.norm(r)/ridge/np.linalg.norm(x)
    result = fixed_residual_limit({'ridge': ridge, 'majorizer_max': upper,
                                  'raw_certificate': {'relative_solution_error_bound': raw}})
    exact_error = np.linalg.norm(np.linalg.solve(h, r))/np.linalg.norm(x)
    energy_certificate = np.sqrt(r@np.linalg.solve(h, r)/ridge)/np.linalg.norm(x)
    assert result['fixed_residual_gap_certificate_floor'] == pytest.approx(energy_certificate)
    assert result['fixed_residual_gap_certificate_floor'] > exact_error
    assert result['threshold_ruled_out_for_fixed_residual_family']


def test_limit_below_target_is_only_a_necessary_condition():
    result = fixed_residual_limit({'ridge': 1., 'majorizer_max': 100.,
                                  'raw_certificate': {'relative_solution_error_bound': 2e-5}})
    assert result['fixed_residual_gap_certificate_floor'] == pytest.approx(2e-6)
    assert not result['threshold_ruled_out_for_fixed_residual_family']


def test_invalid_hessian_bounds_rejected():
    with pytest.raises(ValueError):
        fixed_residual_limit({'ridge': 2., 'majorizer_max': 1.,
                              'raw_certificate': {'relative_solution_error_bound': .1}})
