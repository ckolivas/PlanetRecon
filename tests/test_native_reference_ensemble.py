import numpy as np
import torch
from tools.native_reference_ensemble import cohorts, build_references, mean_field


def test_reference_membership_is_disjoint_excludes_replay_and_balances_time():
    indices=np.arange(1000)*3
    quality=np.random.default_rng(9).random(1000)
    excluded=np.arange(0,1000,5)
    members=cohorts(indices,quality,excluded)
    np.testing.assert_array_equal(members,cohorts(indices,quality,excluded))
    assert members.shape==(4,64)
    assert len(np.unique(members))==256
    assert not np.intersect1d(members,excluded).size
    available=np.setdiff1d(np.arange(1000),excluded)
    best=available[np.argsort(-quality[available],kind='stable')[:256]]
    np.testing.assert_array_equal(np.sort(members,axis=0).T.ravel(),np.sort(best))


def test_pooled_reference_uses_sums_and_support_and_preserves_constant():
    class Source:
        def read_raw(self,index):
            return np.full((12,16),float(index+1))
    members=np.arange(256).reshape(4,64)
    refs,sums,support=build_references(Source(),np.arange(256),np.zeros((256,2)),members,np.zeros((12,16)))
    np.testing.assert_allclose(refs['pooled'],128.5,atol=0,rtol=0)
    np.testing.assert_array_equal(refs['pooled'],sums.sum(0)/support.sum(0))
    fields=[torch.full((2,12,16),v,dtype=torch.float64) for v in (-.3,.1,.2,.4)]
    field,stats=mean_field(fields,(0.,0.))
    torch.testing.assert_close(field,torch.full_like(field,.1),atol=1e-15,rtol=0)
    assert stats['field_guard_accepted']


def test_template_switch_retains_geometry_and_restores_baseline():
    from tools.native_reference_ensemble import NativeReferenceStates
    from tools.regularized_ap_fit import RegularizedAPFit
    from tools.analytic_sampling_probe import components,detector_image
    from planetrecon.pipeline.local_align import LocalRegistration
    previous=torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        ref=detector_image(components('resolved'),(128,160),blur=1.)
        model=RegularizedAPFit(ref,device='cpu',policies=['baseline'])
        states=NativeReferenceStates(model.engine,{'baseline':ref,'ref_a':ref+np.random.default_rng(314).normal(0,.3,ref.shape)})
        points=model.engine.points.copy()
        design=model.engine.fitter.design.copy()
        proxy=LocalRegistration.proxy(ref)
        baseline,_=model.field('baseline',model.observations(proxy,(.2,-.3)),(.2,-.3))
        states.select('ref_a')
        model.observations(proxy,(.2,-.3))
        states.select('baseline')
        restored,_=model.field('baseline',model.observations(proxy,(.2,-.3)),(.2,-.3))
        torch.testing.assert_close(restored,baseline,atol=0,rtol=0)
        np.testing.assert_array_equal(model.engine.points,points)
        assert (design != model.engine.fitter.design).nnz==0
    finally:
        torch.set_num_threads(previous)
