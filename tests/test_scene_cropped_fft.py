import numpy as np
import pytest
from planetrecon.operators import SceneDetectorOperator
from tools.scene_cropped_fft import CroppedSceneFFTBatch,CroppedCellFFTBatch,crop_fft_shape
from tools.scene_study import CellBasisOperator


def cases(pattern,kernel,origins):
    rng=np.random.default_rng(716)
    shape=(16,18) if pattern=='mono' else (16,18,3)
    psfs=[rng.uniform(size=kernel) for _ in range(2)];psfs=[p/p.sum() for p in psfs]
    return [SceneDetectorOperator(shape,tuple(psfs),2,origin,(3,3),exposure_weights=(.3,.7),
                flux=2.,valid_mask=rng.random((3,3))>.2,cfa_pattern=None if pattern in ('mono','rgb') else pattern,
                cfa_offset_xy=(1,1)) for origin in origins]


@pytest.mark.parametrize('pattern',['mono','rgb','RGGB','BGGR','GRBG','GBRG'])
@pytest.mark.parametrize('kernel',[(4,6),(5,3)])
@pytest.mark.parametrize('origins',[[(2,2)],[(0,0),(6,5)]])
def test_crop_forward_adjoint_normal_against_independent_operator(pattern,kernel,origins):
    ops=cases(pattern,kernel,origins);rng=np.random.default_rng(717)
    batch=CroppedSceneFFTBatch(ops,cache_bytes=10000)
    x=rng.normal(size=ops[0].scene_shape);ys=[rng.normal(size=o.output_shape) for o in ops]
    weights=[rng.uniform(.2,2.,size=o.output_shape) for o in ops]
    for _ in range(2):
        for a,o in zip(batch.forward(x),ops): np.testing.assert_allclose(a,o.forward(x),atol=2e-12,rtol=2e-12)
        np.testing.assert_allclose(batch.adjoint(ys),sum(o.adjoint(y) for o,y in zip(ops,ys)),atol=2e-12,rtol=2e-12)
        np.testing.assert_allclose(batch.normal(x,weights),sum(o.adjoint(w*o.forward(x)) for o,w in zip(ops,weights)),atol=2e-12,rtol=2e-12)
        assert sum(np.vdot(a,y) for a,y in zip(batch.forward(x),ys))==pytest.approx(np.vdot(x,batch.adjoint(ys)),abs=2e-12)
    assert batch.cache_info()['peak_bytes']<=10000
    if origins==[(2,2)]: assert np.prod(batch.fft_shape)<np.prod(batch.full_fft_shape)
    batch.clear_cache();assert batch.cache_info()['resident_bytes']==0


def test_insufficient_padding_actually_aliases_retained_boundary_crop():
    op=SceneDetectorOperator((8,8),(np.ones((4,4))/16,),1,(0,0),(3,3))
    batch=CroppedSceneFFTBatch([op],cache_bytes=0)
    x=np.zeros((8,8));x[-1,-1]=1.
    np.testing.assert_allclose(batch.forward(x)[0],op.forward(x),atol=1e-15)
    assert batch.fft_shape==(10,10)
    batch.fft_shape=(8,8)  # Deliberately violate the derived high-fold exclusion.
    assert np.max(np.abs(batch.forward(x)[0]-op.forward(x)))>.05


@pytest.mark.parametrize('crop',[[],[((-1,3),(1,3))],[((1,20),(1,3))]])
def test_invalid_crop_bounds_rejected(crop):
    with pytest.raises(ValueError): crop_fft_shape((8,8),(4,4),crop)


@pytest.mark.hardware
@pytest.mark.parametrize('factor',[1,2])
@pytest.mark.parametrize('pattern',['mono','rgb','RGGB'])
def test_cuda_cells_and_quadratic_certificate_match_reference(factor,pattern):
    import torch
    from tools.scene_quadratic import SceneQuadratic
    if not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    rng=np.random.default_rng(718)
    ops=[CellBasisOperator(o,factor) for o in cases(pattern,(4,6),[(2,2),(3,1)])]
    images=[rng.normal(size=o.output_shape) for o in ops]
    variances=[rng.uniform(.3,2.,size=o.output_shape) for o in ops]
    cpu=SceneQuadratic(ops,images,variances,ridge=.1,smoothness=.02)
    gpu=SceneQuadratic(ops,images,variances,ridge=.1,smoothness=.02,batch=CroppedCellFFTBatch(ops,device='cuda'))
    x=rng.uniform(size=cpu.shape)
    np.testing.assert_allclose(gpu.gradient(x),cpu.gradient(x),atol=2e-12,rtol=2e-12)
    assert gpu.objective(x)==pytest.approx(cpu.objective(x),rel=2e-12)
    assert gpu.certificate(x)['relative_solution_error_bound']==pytest.approx(cpu.certificate(x)['relative_solution_error_bound'],rel=2e-12)
    from tools.scene_gradient_projection import solve
    from tools.scene_window_preconditioner import WindowAveragedPreconditioner
    if factor==1:
        inverse=WindowAveragedPreconditioner(gpu,workers=2)
    else:
        with pytest.raises(ValueError,match='native cells required'):
            WindowAveragedPreconditioner(gpu,workers=2)
        inverse=None
    fitted,info=solve(gpu,preconditioner=inverse)
    assert info['converged'] and cpu.certificate(fitted)['relative_solution_error_bound']<=1e-5


@pytest.mark.parametrize('kernel',[(1,1),(12,11)])
def test_unit_and_larger_than_scene_psfs(kernel):
    rng=np.random.default_rng(719);h=rng.uniform(size=kernel);h/=h.sum()
    op=SceneDetectorOperator((8,8),(h,),1,(2,1),(3,4))
    batch=CroppedSceneFFTBatch([op]);x=rng.normal(size=(8,8));y=rng.normal(size=(3,4))
    np.testing.assert_allclose(batch.forward(x)[0],op.forward(x),atol=1e-12)
    np.testing.assert_allclose(batch.adjoint([y]),op.adjoint(y),atol=1e-12)
