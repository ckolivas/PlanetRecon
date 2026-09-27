import numpy as np
import torch

from tools.coherent_registration import SplineField
from tools.fixed_ap_decisions import baseline_robust_factors, solve_fixed, frozen_parent_observations
from tools.fractional_ap_trace import trace, observations
from tools.regularized_ap_fit import RegularizedAPFit
from tools.analytic_sampling_probe import components, detector_image
from planetrecon.pipeline.local_align import LocalRegistration


def test_frozen_irls_reproduces_original_and_ignores_rejected_positions():
    y,x=np.mgrid[12:84:12,12:108:12]
    fitter=SplineField((96,120),np.column_stack((x.ravel(),y.ravel())),spacing=16,stiffness=.01)
    rng=np.random.default_rng(2304)
    values=rng.normal(size=(x.size,2))
    confidence=rng.uniform(.2,1.,x.size)
    confidence[::4]=0
    expected,_=fitter.fit(values,confidence)
    factor=baseline_robust_factors(fitter,values,confidence)
    actual=solve_fixed(fitter,values,confidence*factor)
    np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=0)
    changed=values.copy()
    changed[confidence==0]+=100
    np.testing.assert_array_equal(solve_fixed(fitter,changed,confidence*factor),actual)


def test_frozen_parents_preserve_inputs_and_first_stage_refinement():
    previous=torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        ref=detector_image(components('resolved'),(128,160),blur=1.)
        model=RegularizedAPFit(ref,device='cpu',policies=['baseline'])
        proxy=LocalRegistration.proxy(ref+np.random.default_rng(817).normal(0,2,ref.shape))
        shift=(.2,-.3)
        stages,records=trace(model.engine,proxy,shift)
        copies=[s.clone() for s in stages]
        values,weights=frozen_parent_observations(model.engine,proxy,shift,stages,records)
        _, baseline_weights=observations(model.engine,stages,records,shift)
        shared=(weights>0)&(baseline_weights>0)
        assert np.count_nonzero(shared)>0
        np.testing.assert_array_equal(weights[shared],baseline_weights[shared])
        fresh,details=trace(model.engine,proxy,shift,refine=True)
        expected,_=observations(model.engine,fresh,details,shift)
        n=len(records[0]['measurements'])
        np.testing.assert_allclose(values[:n],expected[:n],atol=1e-12,rtol=0)
        np.testing.assert_allclose(weights[:n],details[0]['measurements'][:,4]*records[0]['accepted'],atol=0,rtol=0)
        for a,b in zip(stages,copies):
            torch.testing.assert_close(a,b,atol=0,rtol=0)
    finally:
        torch.set_num_threads(previous)
