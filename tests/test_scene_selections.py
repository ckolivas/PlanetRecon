import copy
import numpy as np
import pytest
from tools.scene_selection_manifest import selection_rows
from tools.audit_scene_selections import validate_case


def manifest():
    scores, rows = selection_rows(np.ones((500, 6, 6)), 2.)
    return {'status': 'valid', 'source_input_unchanged': True, 'cases': [
        {'seed': 1001, 'dr0': 4., 'crop': 'feature', 'n_captured': 500,
         'scores': scores, 'selections': rows}]}


def test_complete_selection_case_required():
    source = manifest()
    assert len(validate_case(source, 1001, 4., 'feature')['selections']) == 5
    with pytest.raises(ValueError, match='exactly one'):
        validate_case(source, 1002, 4., 'feature')
    source['cases'] *= 2
    with pytest.raises(ValueError, match='exactly one'):
        validate_case(source, 1001, 4., 'feature')


@pytest.mark.parametrize('mutation', ['missing', 'index', 'count', 'prior', 'score'])
def test_changed_selection_contract_rejected(mutation):
    source = copy.deepcopy(manifest()); case = source['cases'][0]
    if mutation == 'missing': case['selections'].pop()
    if mutation == 'index': case['selections'][0]['indices'][0] = 0
    if mutation == 'count': case['selections'][0]['n_used'] = 24
    if mutation == 'prior': case['selections'][0]['mean_ridge'] *= 2
    if mutation == 'score': case['scores'][0] = float('nan')
    with pytest.raises(ValueError):
        validate_case(source, 1001, 4., 'feature')
