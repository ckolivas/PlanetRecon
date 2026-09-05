import copy
import json

import numpy as np
import pytest

from planetrecon.evaluate import _to_jsonable
from planetrecon.mfbd import fit_frame_alpha, select_init
from planetrecon.q2 import aggregate_q2, closure_C, _known_transfer_block
from test_q2 import _tiny_fwd, _blob


@pytest.mark.parametrize('m', [1,2])
def test_small_mode_freeze_is_exact(m):
    fwd,_ = _tiny_fwd()
    obj_f=np.fft.fft2(_blob(fwd.eval_size))
    image=np.fft.ifft2(fwd.otf(np.array([.3,.2]))*obj_f).real
    start=np.arange(m,dtype=float)*.1
    fit,loss,info=fit_frame_alpha(start,fwd,obj_f,image,1.,freeze_tip_tilt=True,return_info=True)
    np.testing.assert_array_equal(fit,start)
    assert info['nit'] == 0 and info['success'] is True
    assert info['frozen_modes'] == m and len(info['objective_trace']) == 1


def test_phase_diagnostics_and_nonfinite():
    fwd,_=_tiny_fwd(); obj_f=np.fft.fft2(_blob(fwd.eval_size))
    image=np.fft.ifft2(fwd.otf(np.array([.3,.2,.1]))*obj_f).real
    fit,loss,info=fit_frame_alpha(np.zeros(3),fwd,obj_f,image,1.,n_iter=1,return_info=True)
    assert info['nit'] <= 1 and info['nfev'] >= 1
    assert np.isfinite(info['gradient_norm']) and len(info['objective_trace']) == info['nit']+1
    with pytest.raises(ValueError,match='finite inputs'):
        fit_frame_alpha(np.array([np.nan]),fwd,obj_f,image,1.)


@pytest.mark.parametrize('vals', [(1,.2,1),(1,.2,2),(1,.2,1-1e-12),(1,float('nan'),.2),(1,None,.2)])
def test_bad_closure_denominators(vals):
    assert np.isnan(closure_C(*vals))


def record(seed=1001,c=.5,status='valid'):
    return {'seed':seed,'dr0':4.,'crops':{'feature':{'C':c,'status':status,'E_H_D':.7,
        'E_H_A1o':1.,'E_H_E2_star':.4,'E2_star':'E2b','prior_limited':False,
        'chosen_init':'zero','holdout':None}}}


def test_complete_valid_family_required():
    results=[record(s) for s in (1001,1002,1003)]
    kw={'expected_seeds':(1001,1002,1003),'expected_regimes':(4.,)}
    assert aggregate_q2(results,**kw)['4.0']['passed']
    assert not aggregate_q2(results)['4.0']['passed']
    assert not aggregate_q2(results[:2],**kw)['4.0']['passed']
    assert not aggregate_q2(results,**{**kw,'expected_regimes':(4.,7.)})['4.0']['passed']
    for value in (None,float('nan')):
        bad=copy.deepcopy(results);bad[1]['crops']['feature']['C']=value
        assert not aggregate_q2(bad,**kw)['4.0']['passed']
    bad=copy.deepcopy(results);bad[1]['crops']['feature']['status']='incomplete'
    assert not aggregate_q2(bad,**kw)['4.0']['passed']
    with pytest.raises(ValueError,match='duplicate'):
        aggregate_q2(results+[results[0]],**kw)
    with pytest.raises(ValueError,match='empty'):
        aggregate_q2([],**kw)


def test_json_types_and_array_nonfinites():
    output=_to_jsonable({'pass':True,'no':np.bool_(False),'arr':np.array([np.nan,np.inf,1]),'null':None})
    roundtrip=json.loads(json.dumps(output,allow_nan=False))
    assert roundtrip == {'pass':True,'no':False,'arr':[None,None,1.],'null':None}
    assert type(roundtrip['pass']) is bool


def test_selection_never_falls_back_from_invalid_holdout_to_training():
    results={'a':{'stages':[{'train_loss':0.,'holdout_loss':float('nan')}]},
             'b':{'stages':[{'train_loss':2.,'holdout_loss':3.}]}}
    assert select_init(results) == 'b'
    results['b']['stages'][0]['holdout_loss']=None
    with pytest.raises(ValueError,match='invalid losses'):
        select_init(results)


def test_truth_gaps_do_not_select_blind_prior(monkeypatch):
    # Perturb all truth metrics while preserving observed arrays and frozen settings.
    import planetrecon.q2 as q
    from types import SimpleNamespace
    fake=lambda *a,**k:(np.ones((4,4)),{'converged':True})
    for name in ('e2a','e2b','a1o'):
        monkeypatch.setattr(q,name,fake)
    crop=SimpleNamespace(expected=np.ones((2,4,4)),observed=np.ones((2,4,4)),
        otf=np.ones((2,4,4)),support=np.ones((4,4)),shifts=np.zeros((2,2)))
    reg=SimpleNamespace(field_e2a=lambda *a:1.,field_e2b=lambda *a:1.)
    for metric in (1.,100.):
        monkeypatch.setattr(q,'_metrics',lambda *a:{'E_H':metric})
        block=_known_transfer_block(SimpleNamespace(eval_size=4),crop,{'read_noise_e':1.},np.array([0]),reg,.2,'E2a')
        assert block['tv_mu_D'] == .2 and block['E2_star'] == 'E2a'
