import numpy as np
import pytest
from planetrecon.pipeline.local_align import LocalRegistration,pull

def scene(y,x):
    return 300+30*np.cos(x/3)+35*np.sin(y/4)+20*np.sin((x+y)/7)+25*np.cos((2*x-y)/9)

@pytest.mark.parametrize('axis',[0,1])
@pytest.mark.parametrize('sign',[-1,1])
def test_smooth_local_motion_does_not_fail_at_grid_boundary(axis,sign):
    y,x=np.indices((256,288),dtype=float)
    reference=scene(y,x)
    flow=sign*2*np.cos((y if axis==0 else x)/90)
    frame=scene(y if axis==0 else y-flow,x-flow if axis==0 else x)
    displacement=LocalRegistration(reference).displacement(frame,(0.,0.))
    roi=np.s_[48:-48,48:-48]
    assert np.linalg.norm((pull(frame,displacement)-reference)[roi]) < .25*np.linalg.norm((frame-reference)[roi])
    ux,uy=displacement
    jac=(1+np.gradient(ux,axis=1))*(1+np.gradient(uy,axis=0))-np.gradient(ux,axis=0)*np.gradient(uy,axis=1)
    assert jac.min() >= .25
    # The return to global motion must be gradual, including outside grid centres.
    assert max(np.abs(np.diff(d,axis=a)).max() for d in displacement for a in (0,1)) < .2

def test_true_internal_fold_still_falls_back_to_global():
    y,x=np.indices((96,112),dtype=float)
    matcher=LocalRegistration(scene(y,x),step=1,use_cuda=True)
    matcher.texture_valid=np.ones(len(matcher.templates),dtype=bool)
    matcher.strength=np.ones(len(matcher.templates))
    py,px=np.indices((7,7))
    costs=np.array([np.exp(-.02*((py-3)**2+(px-(5 if i<len(matcher.xs)//2 else 1))**2))
                    for _ in matcher.ys for i in range(len(matcher.xs))])
    matcher._cuda_costs=lambda image:costs
    for result,expected in zip(matcher.displacement(scene(y,x),(.375,-.625)),(.375,-.625)):
        np.testing.assert_array_equal(result,np.full((96,112),expected))
