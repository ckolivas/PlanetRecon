import copy
import json
from pathlib import Path
import pytest
from tools.audit_scene_selection_family import select_case


def manifest():
    return json.loads((Path(__file__).resolve().parents[1]/'results/p2-full-selection-manifest/report.json').read_text())


def test_every_declared_case_resolves_complete_observed_selections():
    data=manifest()
    for index in range(12):
        case=select_case(data,index)
        assert [s['n_used'] for s in case['selections']]==[25,50,125,250,500]
        assert case==data['cases'][index]


@pytest.mark.parametrize('index',[-1,12,1.5])
def test_invalid_index_cannot_silently_select_another_case(index):
    with pytest.raises(ValueError,match='case index'):select_case(manifest(),index)


def test_changed_catalog_or_selection_is_rejected():
    data=manifest();data['cases'][0],data['cases'][1]=data['cases'][1],data['cases'][0]
    with pytest.raises(ValueError,match='catalog'):select_case(data,0)
    data=manifest();data['cases'][0]['selections'].pop()
    with pytest.raises(ValueError,match='all five'):select_case(data,0)


def stability():
    certificate={'feasible':True,'relative_solution_error_bound':1e-6}
    proof={'status':'valid','source_input_unchanged':True,'lower_certificate':copy.deepcopy(certificate),
           'upper_fit':{'converged':True,'solver':'extended_scene_projected_acceleration_v4','maxiter':3000,
                        'reference_certificate':copy.deepcopy(certificate)},'relative_changes':{'latent':0.,'detector':0.}}
    protocol={'caps':[1500,3000],'margin':64,'cell_factor':1,'sum_native_strength':.0003,
              'tolerance':1e-5,'image_tolerance':1e-4,'cache_bytes':4*1024**3,'upper_initialization':'zero'}
    return proof,protocol


def test_family_prerequisite_rechecks_certificates_and_stability_not_only_flags():
    from tools.audit_scene_selection_family import require_qualified_stability
    proof,protocol=stability();require_qualified_stability(proof,protocol)
    for changed in ('lower','upper','image','converged','initialization'):
        proof,protocol=stability()
        if changed=='lower':proof['lower_certificate']['relative_solution_error_bound']=float('nan')
        if changed=='upper':proof['upper_fit']['reference_certificate']['relative_solution_error_bound']=.1
        if changed=='image':proof['relative_changes']['latent']=.1
        if changed=='converged':proof['upper_fit']['converged']=False
        if changed=='initialization':protocol['upper_initialization']='lower image'
        with pytest.raises(ValueError):require_qualified_stability(proof,protocol)


@pytest.mark.parametrize('key,value',[('status','incomplete'),('source_input_unchanged',False)])
def test_family_refuses_unqualified_prerequisite(key,value):
    from tools.audit_scene_selection_family import require_qualified_stability
    proof,protocol=stability();proof[key]=value
    with pytest.raises(ValueError,match='qualified'):require_qualified_stability(proof,protocol)
