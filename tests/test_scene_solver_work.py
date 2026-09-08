import time
import pytest
from test_scene_lbfgsb import problem
from tools.scene_quadratic import solve as reference
from tools.scene_projected_newton import solve as newton
from tools.scene_solver_work import normal_work


@pytest.mark.parametrize('method',['reference','newton'])
@pytest.mark.parametrize('cap',[1,7,40])
def test_accounted_products_equal_observed_normal_calls(method,cap):
    p=problem(805);calls=0;normal=p.normal;trace=[]
    def counted(x):
        nonlocal calls
        calls+=1;return normal(x)
    p.normal=counted
    if method=='reference':
        def progress(n,x,cert):trace.append({'iteration':n})
        x,info=reference(p,maxiter=cap,callback=progress,tolerance=1e-12)
        info['trace']=trace
    else:
        x,info=newton(p,max_products=cap,tolerance=1e-12)
    result=normal_work(info)
    assert result['optimizer_normal_products']==calls
    assert all(n<calls for n in result['accepted_trace_normal_products'])


@pytest.mark.parametrize('method',['reference','newton'])
def test_deadline_before_iteration_still_counts_initial_and_final_products(method):
    p=problem(806);calls=0;normal=p.normal
    def counted(x):
        nonlocal calls
        calls+=1;return normal(x)
    p.normal=counted
    _,info=(reference if method=='reference' else newton)(p,deadline=time.monotonic()-1)
    if method=='reference':info['trace']=[]
    assert normal_work(info)['optimizer_normal_products']==calls==2


def test_unknown_solver_is_not_assigned_a_guessed_work_count():
    with pytest.raises(ValueError,match='unsupported solver'):normal_work({'solver':'future'})


def test_reference_resume_counts_only_current_attempt(tmp_path):
    from tools.scene_iteration_state import IterationCheckpoint
    p=problem(807,ridge=1e-5);calls=0;normal=p.normal
    def counted(x):
        nonlocal calls
        calls+=1;return normal(x)
    p.normal=counted
    checkpoint=IterationCheckpoint(tmp_path/'state.npz',{'control':'normal work'})
    def pause(n,x,cert):
        if n==10:raise RuntimeError('pause control')
    with pytest.raises(RuntimeError,match='pause control'):
        reference(p,maxiter=40,tolerance=1e-12,callback=pause,iteration_checkpoint=checkpoint)
    calls=0;trace=[]
    _,info=reference(p,maxiter=40,tolerance=1e-12,iteration_checkpoint=checkpoint,
                     callback=lambda n,x,cert:trace.append({'iteration':n}))
    info['trace']=trace
    assert info['resume_from_iteration']==10
    assert normal_work(info)['optimizer_normal_products']==calls
