"""CUDA motion inversion preserves fixed-point convergence and fallback."""
import numpy as np
import pytest
from planetrecon.pipeline.motion_local import inverse_local_coordinates

@pytest.mark.parametrize('kind', ['zero','constant','varying','divergent','nonfinite'])
def test_cuda_inverse_matches_cpu_and_failure_guard(kind):
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    y,x = np.indices((81,97),dtype=float)
    if kind == 'zero':
        shift = (np.zeros_like(x),np.zeros_like(y))
    elif kind == 'constant':
        shift = (np.full_like(x,1.2),np.full_like(y,-2.3))
    elif kind == 'varying':
        shift = (2*np.sin(x/27)*np.cos(y/30),1.5*np.cos(x/32)*np.sin(y/30))
    elif kind == 'divergent':
        shift = (10*np.sin(x),10*np.cos(y))
    else:
        shift = (np.full_like(x,np.nan),np.zeros_like(y))
    cpu = inverse_local_coordinates(shift)
    gpu = inverse_local_coordinates(shift,use_cuda=True)
    np.testing.assert_allclose(gpu,cpu,atol=5e-13,rtol=0)
    if kind in ('divergent','nonfinite'):
        np.testing.assert_array_equal(gpu,np.stack((x+.5,y+.5)))
