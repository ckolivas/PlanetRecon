import numpy as np
import pytest
import torch
from tools.local_deposition import deposit, inverse_coordinates, splat
from tools.native_drizzle_probe import deposit as translated_deposit


@pytest.mark.parametrize('shift',[(0.,0.),(.3,-.7),(2.,-1.)])
def test_translation_matches_independent_cpu_area_overlap(shift):
    raw=np.random.default_rng(721).normal(size=(25,31))
    field=torch.tensor(shift,dtype=torch.float64)[:,None,None].expand(2,*raw.shape)
    total,support,stats=deposit(torch.tensor(raw),field)
    expected,weights=translated_deposit(raw,shift,1.)
    np.testing.assert_allclose(total,expected,atol=1e-13,rtol=0)
    np.testing.assert_allclose(support,weights,atol=1e-13,rtol=0)
    assert stats['max_inverse_error_px']<1e-12


def test_affine_inverse_known_truth_and_constant_brightness():
    y,x=torch.meshgrid(torch.arange(40,dtype=torch.float64),torch.arange(48,dtype=torch.float64),indexing='ij')
    field=torch.stack((.05*x+.03*y+.2,-.02*x+.06*y-.3))
    qy,qx,stats=inverse_coordinates(field)
    coords=torch.stack((qx,qy))[:,5:-5,5:-5]
    matrix=torch.tensor([[1.05,.03],[-.02,1.06]],dtype=torch.float64)
    expected=torch.linalg.solve(matrix,torch.stack((x-.2,y+.3)).reshape(2,-1)).reshape(2,*x.shape)
    torch.testing.assert_close(coords,expected[:,5:-5,5:-5],atol=1e-7,rtol=0)
    total,support=splat(torch.full_like(x,17.25),qy,qx)
    torch.testing.assert_close(total[support>0]/support[support>0],torch.full_like(total[support>0],17.25),atol=1e-12,rtol=0)
    assert stats['max_inverse_error_px']<=1e-7


def test_interior_impulse_conserves_signal_and_nonfinite_field_rejected():
    y,x=torch.meshgrid(torch.arange(24,dtype=torch.float64),torch.arange(28,dtype=torch.float64),indexing='ij')
    raw=torch.zeros_like(x); raw[12,14]=5
    total,_=splat(raw,y+.21,x-.37)
    assert float(total.sum())==pytest.approx(5.,abs=1e-12)
    with pytest.raises(ValueError,match='finite'):
        inverse_coordinates(torch.full((2,24,28),float('nan')))
