from io import BytesIO
import numpy as np
import pytest
from tools.scene_iteration_state import IterationCheckpoint


def test_single_file_state_identity_and_checksum(tmp_path):
    store=IterationCheckpoint(tmp_path/'state.npz',{'input':'abc','budget':200})
    assert store.load() is None
    x=np.arange(12.).reshape(3,4);y=x-.5
    store.save(10,x,y,2.5)
    state=store.load()
    np.testing.assert_array_equal(state['x'],x);np.testing.assert_array_equal(state['y'],y)
    assert state['iteration']==10 and state['elapsed_s']==2.5
    with pytest.raises(ValueError,match='identity'):
        IterationCheckpoint(store.path,{'input':'changed','budget':200}).load()
    with pytest.raises(ValueError,match='identity'):
        IterationCheckpoint(store.path,{'input':'abc','budget':400}).load()
    with np.load(store.path) as z:parts={k:z[k] for k in z.files}
    parts['y']=parts['y']+1
    stream=BytesIO();np.savez_compressed(stream,**parts);store.path.write_bytes(stream.getvalue())
    with pytest.raises(ValueError,match='corrupt'):
        store.load()


def test_invalid_state_preserves_previous_checkpoint(tmp_path):
    store=IterationCheckpoint(tmp_path/'state.npz',{})
    store.save(1,np.ones((2,2)),np.ones((2,2)),1.)
    before=store.path.read_bytes()
    with pytest.raises(ValueError):store.save(2,-np.ones((2,2)),np.ones((2,2)),2.)
    assert store.path.read_bytes()==before


def test_interrupted_solver_resumes_iterate_and_momentum_exactly(tmp_path):
    from planetrecon.operators import SceneDetectorOperator
    from tools.scene_quadratic import SceneQuadratic,solve
    rng=np.random.default_rng(204)
    op=SceneDetectorOperator((16,16),(np.ones((3,3))/9,),1,(2,2),(12,12))
    problem=SceneQuadratic([op],[rng.normal(2,1,(12,12))],[1.],ridge=.0001)
    reference,ref=solve(problem,maxiter=500,tolerance=1e-7)
    store=IterationCheckpoint(tmp_path/'state.npz',{'input':'bound','operator':'bound','protocol':'bound'})
    def stop(n,x,info):
        if n==20:raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        solve(problem,maxiter=500,tolerance=1e-7,iteration_checkpoint=store,callback=stop)
    actual,info=solve(problem,maxiter=500,tolerance=1e-7,iteration_checkpoint=store)
    assert info['resume_from_iteration']==20 and info['n_iter']==ref['n_iter']
    assert info['converged']==ref['converged']
    np.testing.assert_array_equal(actual,reference)
    with pytest.raises(ValueError,match='identity'):
        solve(problem,maxiter=1000,tolerance=1e-7,iteration_checkpoint=store)
    with pytest.raises(ValueError,match='identity'):
        solve(problem,maxiter=500,tolerance=1e-5,iteration_checkpoint=store)
