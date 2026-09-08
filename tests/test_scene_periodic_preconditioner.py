import numpy as np
import pytest
from types import SimpleNamespace
from planetrecon.operators import SceneDetectorOperator
from tools.scene_quadratic import SceneQuadratic
from tools.scene_periodic_preconditioner import PeriodicScenePreconditioner, _spectrum_term


@pytest.mark.parametrize('rgb', [False,True])
def test_inverse_is_symmetric_positive_on_full_and_reduced_active_sets(rgb):
    rng = np.random.default_rng(902)
    shape = (6,6,3) if rgb else (6,6)
    op = SceneDetectorOperator(shape,(np.ones((3,4))/12,),2,(0,0),(3,3),
                               cfa_pattern='RGGB' if rgb else None,
                               valid_mask=rng.uniform(size=(3,3)) > .2)
    p = SceneQuadratic([op],[rng.uniform(size=op.output_shape)],[1.],ridge=.01,smoothness=.02)
    pre = PeriodicScenePreconditioner(p,workers=2)
    n = int(np.prod(shape)); basis = np.eye(n).reshape((n,)+shape)
    inverse = np.stack([pre(e).ravel() for e in basis],axis=1)
    np.testing.assert_allclose(inverse,inverse.T,rtol=1e-12,atol=1e-12)
    assert np.linalg.eigvalsh(inverse).min() > 0
    free = rng.uniform(size=n) > .4
    assert np.linalg.eigvalsh(inverse[np.ix_(free,free)]).min() > 0
    assert pre.info()['symbol_min'] >= p.ridge


def test_periodic_single_frame_bin_one_matches_independent_dense_toroidal_inverse():
    rng = np.random.default_rng(903); shape=(4,5)
    h=rng.uniform(size=(3,2)); h/=h.sum()
    op=SceneDetectorOperator(shape,(h,),1,(0,0),shape,flux=1.7)
    p=SceneQuadratic([op],[np.ones(shape)],[2.],ridge=.03)
    pre=PeriodicScenePreconditioner(p,workers=1)
    def toroidal(x):
        return op.flux*sum(h[i,j]*np.roll(x,(i,j),axis=(0,1)) for i,j in np.ndindex(h.shape))
    a=np.stack([toroidal(e.reshape(shape)).ravel() for e in np.eye(20)],axis=1)
    rhs=rng.normal(size=shape)
    expected=np.linalg.solve(a.T@a/2+.03*np.eye(20),rhs.ravel())
    np.testing.assert_allclose(pre(rhs).ravel(),expected,rtol=1e-12,atol=1e-12)


def test_detector_integration_precedes_squaring_and_sampling_density():
    op=SceneDetectorOperator((4,4),(np.ones((1,1)),),2,(0,0),(2,2))
    term=_spectrum_term(op,np.ones((2,2))*3)
    box=np.zeros((4,4));box[:2,:2]=1
    expected=np.abs(np.fft.rfft2(box))**2*3/4
    np.testing.assert_allclose(term,expected,atol=1e-12)


def test_oversized_kernel_folds_instead_of_truncating():
    op=SceneDetectorOperator((2,2),(np.ones((3,3))/9,),1,(0,0),(2,2))
    term=_spectrum_term(op,np.ones((2,2)))
    folded=np.array([[4,2],[2,1]])/9
    np.testing.assert_allclose(term,np.abs(np.fft.rfft2(folded))**2,atol=1e-12)


def test_unsupported_cells_and_translations_rejected():
    from tools.scene_study import CellBasisOperator
    op=SceneDetectorOperator((4,4),(np.ones((1,1)),),1,(0,0),(4,4))
    with pytest.raises(ValueError,match='native'):
        _spectrum_term(CellBasisOperator(op,2),np.ones((4,4)))
    shifted=SceneDetectorOperator((4,4),(np.ones((1,1)),),1,(0,0),(4,4),shifts_xy=((.1,0.),))
    with pytest.raises(ValueError,match='translations'): _spectrum_term(shifted,np.ones((4,4)))
