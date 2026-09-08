import time
import pytest
from test_scene_lbfgsb import problem
from tools.scene_quadratic import solve as reference
from tools.scene_projected_newton import solve as newton
from tools.scene_solver_work import normal_work
from tools.scene_solver_work import required_solver_sources


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


@pytest.mark.parametrize('solvers,expected', [
    (['extended_scene_projected_acceleration_v4'], ['tools/scene_quadratic.py']),
    (['extended_scene_projected_newton_cg_v2'], ['tools/scene_projected_newton.py']),
    (['extended_scene_projected_acceleration_v4', 'extended_scene_projected_newton_cg_v2'],
     ['tools/scene_projected_newton.py', 'tools/scene_quadratic.py']),
    ([], []),
])
def test_work_requires_only_sources_of_recorded_solvers(solvers, expected):
    report = {'rows': [{'runs': [{'solver': s} for s in solvers] + [{'reason': 'deadline'}]}]}
    assert required_solver_sources(report) == expected


def test_work_source_check_rejects_unknown_solver():
    with pytest.raises(ValueError, match='unsupported solver'):
        required_solver_sources({'rows': [{'runs': [{'solver': 'unknown'}]}]})


def test_reference_only_cli_accepts_its_source_and_rejects_changed_hash(tmp_path):
    import json
    import subprocess
    import sys
    from pathlib import Path
    from tools.study_io import file_hash
    root = Path(__file__).resolve().parents[1]
    protocol = {'identities': {'dependencies': {'tools/scene_quadratic.py': file_hash(root/'tools/scene_quadratic.py')}}}
    fit = {'solver': 'extended_scene_projected_acceleration_v4', 'maxiter': 1500,
           'n_iter': 10, 'resume_from_iteration': 0, 'trace': [{'iteration': 10}]}
    report = {'source_input_unchanged': True, 'rows': [{'fraction': 5, 'runs': [fit]}]}
    (tmp_path/'report.json').write_text(json.dumps(report))
    (tmp_path/'protocol.json').write_text(json.dumps(protocol))
    output = tmp_path/'work.json'
    command = [sys.executable, str(root/'tools/scene_solver_work.py'), str(tmp_path), '--out', str(output)]
    completed = subprocess.run(command, cwd=root, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert json.loads(output.read_text())['rows'][0]['optimizer_normal_products'] == 13
    protocol['identities']['dependencies']['tools/scene_quadratic.py'] = 'changed'
    (tmp_path/'protocol.json').write_text(json.dumps(protocol))
    failed_output = tmp_path/'failed-work.json'
    completed = subprocess.run(command[:-1] + [str(failed_output)], cwd=root, capture_output=True, text=True)
    assert completed.returncode != 0 and 'tested solver source' in completed.stderr
    assert not failed_output.exists()
