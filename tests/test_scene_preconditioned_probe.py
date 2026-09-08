import time
import numpy as np
import pytest
from tools.scene_preconditioned_probe import probe


@pytest.mark.parametrize('reduced',[False,True])
def test_dense_coupled_inverse_and_principal_system(reduced):
    rng=np.random.default_rng(413); a=rng.normal(size=(12,12)); h=a.T@a+np.eye(12)
    b=rng.normal(size=(12,12)); inverse=b.T@b+np.eye(12)
    free=np.arange(12)%3 != 0 if reduced else np.ones(12,bool)
    rhs=np.where(free,rng.normal(size=12),0.)
    x,info=probe(lambda x:h@x,rhs,lambda x:inverse@x,free=free,steps=32)
    expected=np.linalg.solve(h[np.ix_(free,free)],rhs[free])
    np.testing.assert_allclose(x[free],expected,rtol=1e-10,atol=1e-12)
    assert np.all(x[~free] == 0)
    assert np.linalg.norm((rhs-h@x)[free])/np.linalg.norm(rhs) < 1e-10
    assert info['reason']=='linear_residual_small'


def test_deadline_and_zero_residual():
    x,info=probe(lambda x:x,np.ones(3),lambda x:x,deadline=time.monotonic()-1)
    assert info['reason']=='wall_budget' and info['iterations']==0 and np.all(x==0)
    x,info=probe(lambda x:x,np.zeros(3),lambda x:x)
    assert info['reason']=='zero_residual'


def test_nonpositive_inverse_and_unsupported_residual_rejected():
    with pytest.raises(ValueError,match='positive'):
        probe(lambda x:x,np.ones(3),lambda x:-x)
    with pytest.raises(ValueError,match='supported'):
        probe(lambda x:x,np.ones(3),lambda x:x,free=np.array([1,0,1]))
