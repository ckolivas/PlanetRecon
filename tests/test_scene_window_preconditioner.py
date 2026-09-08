import numpy as np
from planetrecon.operators import SceneDetectorOperator
from tools.scene_quadratic import SceneQuadratic
from tools.scene_periodic_preconditioner import PeriodicScenePreconditioner
from tools.scene_window_preconditioner import WindowAveragedPreconditioner


def test_window_average_matches_exact_fourier_diagonal_for_complete_psf_footprints():
    rng=np.random.default_rng(910);shape=(8,8)
    h=rng.uniform(size=(3,3));h/=h.sum()
    op=SceneDetectorOperator(shape,(h,),2,(1,1),(2,2),flux=1.4)
    p=SceneQuadratic([op],[np.ones((2,2))],[np.array([[1.,2.],[3.,4.]])],ridge=.01)
    candidate=WindowAveragedPreconditioner(p,workers=1)
    original=PeriodicScenePreconditioner(p,workers=1)
    yy,xx=np.indices(shape)
    for ky in range(8):
        for kx in range(5):
            angle=2*np.pi*(ky*yy/8+kx*xx/8)
            re=np.cos(angle)/8;im=np.sin(angle)/8
            exact=np.vdot(re,p.normal(re))+np.vdot(im,p.normal(im))
            np.testing.assert_allclose(candidate.symbol[ky,kx],exact,rtol=1e-12,atol=1e-12)
    assert candidate.window_fraction==.25
    assert original.symbol[0,0] > 3*candidate.symbol[0,0]


def test_full_detector_coverage_does_not_change_original_inverse():
    op=SceneDetectorOperator((6,6),(np.ones((1,1)),),2,(0,0),(3,3))
    p=SceneQuadratic([op],[np.ones((3,3))],[1.],ridge=.01,smoothness=.02)
    a=WindowAveragedPreconditioner(p,workers=1)
    b=PeriodicScenePreconditioner(p,workers=1)
    assert a.window_fraction==1.
    np.testing.assert_array_equal(a.symbol,b.symbol)


def test_window_inverse_stays_positive_on_a_reduced_set():
    op=SceneDetectorOperator((4,4,3),(np.ones((1,1)),),1,(1,1),(2,2),cfa_pattern='RGGB')
    p=SceneQuadratic([op],[np.ones((2,2))],[1.],ridge=.01,smoothness=.02)
    inverse=WindowAveragedPreconditioner(p,workers=1)
    basis=np.eye(48).reshape((48,4,4,3))
    matrix=np.stack([inverse(x).ravel() for x in basis],axis=1)
    np.testing.assert_allclose(matrix,matrix.T,atol=1e-12)
    subset=np.arange(48)%3!=0
    assert np.linalg.eigvalsh(matrix[np.ix_(subset,subset)]).min()>0
