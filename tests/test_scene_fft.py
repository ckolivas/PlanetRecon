import numpy as np
import pytest
from planetrecon.operators import SceneDetectorOperator
from tools.scene_fft import SceneFFTBatch


def operators(rgb, even):
    rng = np.random.default_rng(43)
    shape = (12, 14)+( (3,) if rgb else () )
    kernel = (4, 2) if even else (3, 5)
    psfs = [rng.uniform(size=kernel) for _ in range(3)]
    psfs = [p/p.sum() for p in psfs]
    return [SceneDetectorOperator(shape, (psfs[i], psfs[(i+1)%3]), 2, (1+i%2, 1), (3, 4),
                                  exposure_weights=(.3, .7), flux=2.1+i,
                                  valid_mask=rng.uniform(size=(3, 4)) > .2,
                                  cfa_pattern=('RGGB','BGGR','GRBG')[i] if rgb else None,
                                  cfa_offset_xy=(i%2, i//2)) for i in range(3)]


@pytest.mark.parametrize('rgb', [False, True])
@pytest.mark.parametrize('even', [False, True])
@pytest.mark.parametrize('cache', [0, 2000, 1000000])
def test_shared_fft_matches_reference_and_respects_cache(rgb, even, cache):
    ops = operators(rgb, even)
    batch = SceneFFTBatch(ops, cache_bytes=cache)
    rng = np.random.default_rng(12)
    x = rng.normal(size=ops[0].scene_shape)
    ys = [rng.normal(size=op.output_shape) for op in ops]
    weights = [rng.uniform(.2, 2., size=op.output_shape) for op in ops]
    for _ in range(2):
        for actual, op in zip(batch.forward(x), ops):
            np.testing.assert_allclose(actual, op.forward(x), atol=1e-12, rtol=1e-12)
        ref = sum(op.adjoint(y) for op, y in zip(ops, ys))
        np.testing.assert_allclose(batch.adjoint(ys), ref, atol=1e-12, rtol=1e-12)
        ref = sum(op.adjoint(w*op.forward(x)) for op, w in zip(ops, weights))
        np.testing.assert_allclose(batch.normal(x, weights), ref, atol=1e-11, rtol=1e-12)
    assert batch.cache_info()['peak_bytes'] <= cache
    if cache == 1000000:
        assert batch.cache_info()['hits'] > 0
    batch.clear_cache()
    assert batch.cache_info()['resident_bytes'] == 0


def test_unsupported_translations_rejected():
    op = SceneDetectorOperator((8, 8), (np.ones((1, 1)),), 1, (0, 0), (8, 8), shifts_xy=((.1, 0),))
    with pytest.raises(ValueError, match='translated'):
        SceneFFTBatch([op])


@pytest.mark.hardware
def test_cuda_float64_operator_parity():
    import torch
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    for rgb in (False, True):
        ops = operators(rgb, True)
        batch = SceneFFTBatch(ops, cache_bytes=1000000, device='cuda')
        rng = np.random.default_rng(63)
        x = rng.normal(size=ops[0].scene_shape)
        residuals = [rng.normal(size=o.output_shape) for o in ops]
        for y, op in zip(batch.forward(x), ops):
            np.testing.assert_allclose(y, op.forward(x), atol=1e-11, rtol=1e-11)
        np.testing.assert_allclose(batch.adjoint(residuals), sum(o.adjoint(y) for o,y in zip(ops,residuals)), atol=1e-11)
        weights = [np.ones(o.output_shape) for o in ops]
        np.testing.assert_allclose(batch.normal(x,weights), sum(o.adjoint(o.forward(x)) for o in ops), atol=1e-10)


def test_solver_objective_gradient_certificate_parity():
    from tools.scene_quadratic import SceneQuadratic, solve
    ops = operators(True, True)
    rng = np.random.default_rng(22)
    images = [rng.normal(size=o.output_shape) for o in ops]
    variances = [rng.uniform(.4, 3., size=o.output_shape) for o in ops]
    ref = SceneQuadratic(ops, images, variances, ridge=.1, smoothness=.02)
    fast = SceneQuadratic(ops, images, variances, ridge=.1, smoothness=.02, batch=SceneFFTBatch(ops))
    x = rng.uniform(size=ref.shape)
    np.testing.assert_allclose(fast.gradient(x), ref.gradient(x), atol=1e-12)
    assert fast.objective(x) == pytest.approx(ref.objective(x), rel=1e-13)
    a, ia = solve(ref, tolerance=1e-7)
    b, ib = solve(fast, tolerance=1e-7)
    assert ia['converged'] and ib['converged']
    assert np.linalg.norm(a-b) <= ia['absolute_solution_error_bound']+ib['absolute_solution_error_bound']
    assert ref.certificate(b)['relative_solution_error_bound'] <= 1e-7


@pytest.mark.parametrize('pattern',[None,'RGGB','BGGR','GRBG','GBRG'])
def test_full_rgb_and_all_bayer_patterns(pattern):
    rng=np.random.default_rng(351)
    op=SceneDetectorOperator((8,10,3),(np.ones((3,4))/12,),2,(1,1),(3,4),cfa_pattern=pattern,cfa_offset_xy=(1,1))
    batch=SceneFFTBatch([op])
    x=rng.normal(size=op.scene_shape);y=rng.normal(size=op.output_shape)
    np.testing.assert_allclose(batch.forward(x)[0],op.forward(x),atol=1e-12)
    np.testing.assert_allclose(batch.adjoint([y]),op.adjoint(y),atol=1e-12)


@pytest.mark.hardware
def test_cuda_solver_checkpoint_resume_and_cpu_certificate(tmp_path):
    import torch
    from tools.scene_quadratic import SceneQuadratic,solve
    from tools.scene_iteration_state import IterationCheckpoint
    if not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    ops=operators(True,True);rng=np.random.default_rng(773)
    images=[rng.normal(1.,1.,o.output_shape) for o in ops];variances=[1.]*len(ops)
    cpu=SceneQuadratic(ops,images,variances,ridge=.005)
    gpu=SceneQuadratic(ops,images,variances,ridge=.005,batch=SceneFFTBatch(ops,device='cuda'))
    expected,ref=solve(gpu,maxiter=500,tolerance=1e-7)
    checkpoint=IterationCheckpoint(tmp_path/'gpu.npz',{'data':'fixed','operator':'fixed','runtime':'fixed'})
    def stop(n,x,info):
        if n==20:raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):solve(gpu,maxiter=500,tolerance=1e-7,iteration_checkpoint=checkpoint,callback=stop)
    actual,info=solve(gpu,maxiter=500,tolerance=1e-7,iteration_checkpoint=checkpoint)
    assert info['converged'] and ref['converged'] and info['resume_from_iteration']==20
    np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=1e-12)
    assert cpu.certificate(actual)['relative_solution_error_bound']<=1e-7
