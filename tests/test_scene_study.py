import numpy as np
import pytest
from planetrecon.operators import SceneDetectorOperator
from tools.scene_study import CellBasisOperator, CellFFTBatch, expand_cells, prior_coefficient, select_candidate
from tools.scene_quadratic import SceneQuadratic, solve


@pytest.mark.parametrize('factor',[1,2,4])
def test_cell_flux_adjoint_prior_and_fast_normal(factor):
    rng=np.random.default_rng(72)
    base=SceneDetectorOperator((16,16),(np.ones((4,4))/16,),4,(0,0),(4,4))
    op=CellBasisOperator(base,factor)
    x=rng.normal(size=op.scene_shape);y=rng.normal(size=op.output_shape)
    assert expand_cells(x,factor).sum()==pytest.approx(x.sum())
    np.testing.assert_allclose(np.vdot(op.forward(x),y),np.vdot(x,op.adjoint(y)),atol=1e-12)
    batch=CellFFTBatch([op])
    np.testing.assert_allclose(batch.normal(x,[np.ones((4,4))]),op.adjoint(op.forward(x)),atol=1e-12)
    density=3.
    coarse=np.full(op.scene_shape,density*factor**2)
    native=expand_cells(coarse,factor)
    assert prior_coefficient(.003,3,factor)*np.sum(coarse**2)==pytest.approx(prior_coefficient(.003,3)*np.sum(native**2))


def test_photon_normalization_preserves_sum_objective():
    op=SceneDetectorOperator((4,4),(np.ones((1,1)),),1,(0,0),(4,4))
    y=np.full((4,4),2.)
    one=SceneQuadratic([op],[y],[1.],ridge=prior_coefficient(.3,1))
    two=SceneQuadratic([op,op],[y,y],[1.,1.],ridge=prior_coefficient(.3,2))
    a,_=solve(one);b,_=solve(two)
    np.testing.assert_allclose(a,2/1.3)
    np.testing.assert_allclose(b,2/1.15)
    # Independent extra observations weaken prior relative to the summed likelihood.
    assert b.mean()>a.mean()


def test_selection_ignores_assessment_and_rejects_incomplete_candidates():
    rows=[dict(sum_native_strength=s,selection_score=score,numerical_passed=True,assessment_score=0.)
          for s,score in [(0.0003,2.),(.003,1.),(.03,1.+1e-10)]]
    assert select_candidate(rows)==.03
    rows[-1]['assessment_score']=1e12
    assert select_candidate(rows)==.03
    rows[0]['numerical_passed']=False
    with pytest.raises(ValueError,match='all candidates'):
        select_candidate(rows)


def test_exact_psf_influence_domain_preserves_regularized_solution():
    rng=np.random.default_rng(902)
    psf=rng.uniform(size=(4,4));psf/=psf.sum()
    full=SceneDetectorOperator((12,12),(psf,),1,(4,4),(4,4))
    compact=SceneDetectorOperator((8,8),(psf,),1,(2,2),(4,4))
    y=rng.normal(.5,1.,(4,4))
    a,ia=solve(SceneQuadratic([full],[y],[1.],ridge=.03),tolerance=1e-8)
    b,ib=solve(SceneQuadratic([compact],[y],[1.],ridge=.03),tolerance=1e-8)
    assert ia['converged'] and ib['converged']
    np.testing.assert_allclose(a[2:10,2:10],b,atol=1e-8)
    outside=a.copy();outside[2:10,2:10]=0
    assert np.linalg.norm(outside)<=ia['absolute_solution_error_bound']+1e-12


def test_rgb_cell_detector_flux_and_generator_batch():
    base=SceneDetectorOperator((8,8,3),(np.ones((1,1)),),2,(1,1),(2,2))
    op=CellBasisOperator(base,2)
    x=np.ones(op.scene_shape)*4
    np.testing.assert_allclose(op.detector_scene(x),4.)
    batch=CellFFTBatch(o for o in [op])
    np.testing.assert_allclose(batch.forward(x)[0],op.forward(x),atol=1e-12)
