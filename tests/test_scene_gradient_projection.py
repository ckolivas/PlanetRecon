import time
import numpy as np
import pytest
from test_scene_lbfgsb import problem
from test_scene_quadratic import dense_oracle
from tools.scene_gradient_projection import solve


@pytest.mark.parametrize('seed',[41,42,43])
@pytest.mark.parametrize('rgb',[False,True])
@pytest.mark.parametrize('positive_start',[False,True])
def test_dense_optimum_and_releasing_incorrect_bound_set(seed,rgb,positive_start):
    p = problem(seed,rgb,True)
    oracle,_ = dense_oracle(p)
    initial = np.full(p.shape,5.) if positive_start else np.zeros(p.shape)
    states = [initial]
    x,info = solve(p,x0=initial,callback=lambda row,x:states.append(x))
    assert info['converged'] and info['feasible']
    assert np.linalg.norm(x-oracle) <= info['absolute_solution_error_bound']+1e-10
    assert p.objective(x)-p.objective(oracle) <= info['objective_gap_upper_bound']+1e-10
    assert all(np.min(x) >= 0 for x in states)
    assert np.all(np.diff([p.objective(x) for x in states]) <= 1e-12)
    assert all(row['quadratic_objective_change'] < 0 for row in info['trace'])
    assert info['trace'][0]['step_kind'] == 'gradient_projection'


@pytest.mark.parametrize('rgb',[False,True])
def test_positive_coupled_inverse_preserves_dense_optimum(rgb):
    from tools.scene_window_preconditioner import WindowAveragedPreconditioner
    p=problem(141,rgb,True); oracle,_=dense_oracle(p)
    x,info=solve(p,preconditioner=WindowAveragedPreconditioner(p,workers=2))
    assert info['converged']
    assert np.linalg.norm(x-oracle) <= info['absolute_solution_error_bound']+1e-10


def test_budgets_preserve_last_accepted_scene_and_trace():
    p=problem(142,True,True,ridge=1e-5)
    for cap in [1,3,10,35]:
        states=[]
        x,info=solve(p,max_products=cap,tolerance=1e-12,callback=lambda row,x:states.append(x))
        assert info['reason']=='product_budget' and not info['converged']
        assert info['hessian_products']==cap
        np.testing.assert_array_equal(x,states[-1] if states else np.zeros(p.shape))
        assert np.min(x)>=0
    x,info=solve(p,deadline=time.monotonic()-1)
    assert info['reason']=='wall_budget' and info['n_iter']==0 and np.all(x==0)


def test_invalid_parameters_and_inverse_rejected():
    p=problem(143,ridge=1e-5)
    for kw in [{'max_products':0},{'projection_steps':np.inf},{'inner_steps':1.5},{'tolerance':0},
               {'x0':-np.ones(p.shape)}]:
        with pytest.raises(ValueError): solve(p,**kw)
    with pytest.raises(ValueError,match='positive'):
        solve(p,preconditioner=lambda r:-r)


@pytest.mark.hardware
@pytest.mark.parametrize('rgb',[False,True])
def test_cuda_candidate_meets_independent_cpu_certificate(rgb):
    import torch
    from tools.scene_fft import SceneFFTBatch
    from tools.scene_quadratic import SceneQuadratic
    from tools.scene_window_preconditioner import WindowAveragedPreconditioner
    if not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    cpu=problem(144,rgb)
    gpu=SceneQuadratic(cpu.operators,cpu.images,[1/w for w in cpu.weights],ridge=cpu.ridge,
                       smoothness=cpu.smoothness,prior=cpu.prior,batch=SceneFFTBatch(cpu.operators,device='cuda'))
    x,info=solve(gpu,preconditioner=WindowAveragedPreconditioner(gpu,workers=2))
    assert info['converged']
    assert cpu.certificate(x)['relative_solution_error_bound']<=1e-5


def test_rejected_face_direction_records_safeguard_before_callback(monkeypatch):
    import tools.scene_gradient_projection as module
    def bad_direction(normal,g,free,diagonal,callback,**kwargs):
        callback({'iteration':1,'recursive_relative_residual':1.})
        return np.where(free,g,0.),0
    monkeypatch.setattr(module,'direction',bad_direction)
    events=[]
    x,info=solve(problem(146),max_products=25,callback=lambda row,x:events.append(row))
    safeguards=[row for row in events if row['step_kind']=='projected_majorizer']
    assert safeguards and all(len(row['rejected_inner_trace'])==1 for row in safeguards)
    assert np.min(x)>=0 and all(row['quadratic_objective_change']<0 for row in events)
