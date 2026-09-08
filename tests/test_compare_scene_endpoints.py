import copy
import pytest
from tools.compare_scene_endpoints import compare


def fixture():
    protocol = {'input_sha256': 'a', 'manifest_sha256': 'b', 'case': [1001, 4., 'feature'],
                'fractions': [5], 'sum_native_strength': .0003, 'margin': 64, 'cell_factor': 1,
                'budgets': [750], 'tolerance': 1e-5, 'image_tolerance': 1e-4}
    report = {'source_input_unchanged': True, 'rows': [{'fraction': 5, 'numerical_passed': True,
              'indices': [1], 'mean_ridge': .0003, 'observed_electron_sum': 5., 'runs': [
              {'maxiter': 750, 'converged': True, 'objective': 1., 'reference_certificate': {
                  'feasible': True, 'relative_solution_error_bound': 1e-6, 'objective_gap_upper_bound': 1e-4}}]}]}
    return protocol, report


def test_certified_same_objective_gap_bound_check():
    p, a = fixture(); b = copy.deepcopy(a)
    b['rows'][0]['runs'][0]['objective'] += .0001
    result = compare(a, b, p, p)
    assert result['rows'][0]['fits'][0]['objective_consistent_with_bounds']
    b['rows'][0]['runs'][0]['objective'] += .01
    assert not compare(a, b, p, p)['rows'][0]['fits'][0]['objective_consistent_with_bounds']


def test_incomplete_fit_cannot_be_an_accuracy_baseline():
    p, a = fixture(); b = copy.deepcopy(a)
    a['rows'][0]['numerical_passed'] = False
    a['rows'][0]['runs'][0] = {'maxiter': 750, 'converged': False, 'reason': 'wall budget'}
    result = compare(a, b, p, p)
    assert result['candidate_endpoints_passed'] and not result['reference_endpoints_passed']
    assert result['rows'][0]['fits'][0]['objective_consistent_with_bounds'] is None
    assert result['q3_authorized'] is False


def test_different_data_or_prior_rejects_comparison():
    p, a = fixture(); q = copy.deepcopy(p); q['sum_native_strength'] *= 2
    with pytest.raises(ValueError, match='same observations'):
        compare(a, a, p, q)
    b = copy.deepcopy(a); b['rows'][0]['indices'] = [2]
    with pytest.raises(ValueError, match='inputs differ'):
        compare(a, b, p, p)


def test_stale_pass_flag_cannot_hide_failed_certificate():
    p, a = fixture(); b = copy.deepcopy(a)
    b['rows'][0]['runs'][0]['reference_certificate']['relative_solution_error_bound'] = .1
    assert not compare(a, b, p, p)['candidate_endpoints_passed']


def test_product_caps_compare_objectives_without_claiming_equal_work():
    p, a = fixture(); q = copy.deepcopy(p); b = copy.deepcopy(a)
    q['budget_unit'] = 'Hessian products'
    fit = b['rows'][0]['runs'][0]
    fit['max_products'] = fit.pop('maxiter')
    result = compare(a, b, p, q)
    assert result['candidate_endpoints_passed'] and result['certified_pair_count'] == 1
    assert result['reference_budget_unit'] == 'outer iterations'
    assert result['candidate_budget_unit'] == 'Hessian products'
    assert not result['equal_nominal_caps_imply_equal_work']


def test_absent_certified_pairs_are_explicit():
    p, a = fixture(); b = copy.deepcopy(a)
    b['rows'][0]['runs'][0]['converged'] = False
    result = compare(a, b, p, p)
    assert result['certified_pair_count'] == 0
    assert result['rows'][0]['fits'][0]['objective_absolute_difference'] is None
