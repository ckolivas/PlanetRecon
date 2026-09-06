from dataclasses import replace
import os

import numpy as np
import pytest
from scipy.ndimage import shift as ndshift

from planetrecon.detector import cfa_accumulate, bilinear_demosaic, is_bayer
from planetrecon.backends.base import Backend, DeviceReport
from planetrecon.io.source import ArraySource
from planetrecon.pipeline.baseline import stack_source
from planetrecon.reconstruction import ReconstructionConfig


@pytest.fixture
def cuda_backend():
    if os.environ.get('PLANETRECON_TEST_GPU')!='1':pytest.skip('explicit GPU test opt-in required')
    torch=pytest.importorskip('torch')
    if not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    torch.cuda.set_per_process_memory_fraction(.15)
    from planetrecon.backends.torch_accel import TorchBackend
    return TorchBackend()


@pytest.mark.hardware
@pytest.mark.parametrize('color',['mono','RGB','RGGB','GRBG','GBRG','BGGR'])
@pytest.mark.parametrize('offset',[(0.,0.),(2.,-3.),(.375,-.625),(40.,0.)])
def test_cuda_backprojection_matches_cpu(cuda_backend,color,offset):
    rng=np.random.default_rng(45)
    raw=rng.uniform(0,65000,(11,13,3) if color=='RGB' else (11,13))
    add,weight,demo,support=cuda_backend.backproject(raw,offset,color)
    sy,sx=-offset[1],-offset[0]
    if is_bayer(color):
        a,w=cfa_accumulate(raw,offset,color)
        d=ndshift(bilinear_demosaic(raw,color),(sy,sx,0),order=1,prefilter=False,mode='grid-constant')
        np.testing.assert_allclose(demo,d,rtol=1e-13,atol=1e-9)
    else:
        offsets=(sy,sx,0) if raw.ndim==3 else (sy,sx)
        a=ndshift(raw,offsets,order=1,prefilter=False,mode='grid-constant')
        w=ndshift(np.ones_like(raw),offsets,order=1,prefilter=False,mode='grid-constant')
    np.testing.assert_allclose(add,a,rtol=1e-13,atol=1e-9)
    np.testing.assert_allclose(weight,w,rtol=1e-13,atol=1e-13)


@pytest.mark.hardware
@pytest.mark.parametrize('color',['mono','RGB','RGGB'])
def test_cuda_stack_matches_cpu(cuda_backend,color):
    rng=np.random.default_rng(44)
    shape=(24,32,3) if color=='RGB' else (24,32)
    frame=rng.integers(100,2000,shape,dtype='u2')
    frames=np.stack([frame,np.roll(frame,2,axis=0),np.roll(frame,-3,axis=1)])
    cfg=ReconstructionConfig(device='cpu',threads=2,batch_frames=2)
    cpu=stack_source(ArraySource(frames,color_mode=color),cfg)
    gpu=stack_source(ArraySource(frames,color_mode=color),replace(cfg,device='gpu'))
    assert gpu.backend=='cuda' and gpu.precision=='float64'
    np.testing.assert_array_equal(gpu.validity,cpu.validity)
    np.testing.assert_allclose(gpu.image,cpu.image,rtol=1e-12,atol=1e-10)
    np.testing.assert_allclose(gpu.coverage,cpu.coverage,rtol=1e-12,atol=1e-10)


def test_mid_job_cuda_failure_retains_sums_and_continues(monkeypatch):
    rng=np.random.default_rng(4);frames=rng.uniform(1,100,(4,12,16))
    source=ArraySource(frames)
    cfg=ReconstructionConfig(device='cpu',threads=2)
    expected=stack_source(source,cfg)
    class Failing(Backend):
        name='cuda';precision='float64';calls=0
        def backproject(self,raw,offset,color):
            self.calls+=1
            if self.calls==2:raise RuntimeError('simulated CUDA out of memory')
            offsets=(-offset[1],-offset[0])
            return (ndshift(raw,offsets,order=1,prefilter=False,mode='grid-constant'),
                ndshift(np.ones_like(raw),offsets,order=1,prefilter=False,mode='grid-constant'),None,None)
    monkeypatch.setattr('planetrecon.pipeline.baseline.select_backend',lambda *a,**k:(Failing(),DeviceReport('gpu','gpu',['cpu','gpu'],False)))
    actual=stack_source(source,replace(cfg,device='gpu'))
    np.testing.assert_array_equal(actual.image,expected.image)
    assert actual.n_used==4 and actual.backend=='cpu'
    assert actual.provenance['device_report']['execution_history']==['cuda','cpu']
    assert any('out of memory' in warning for warning in actual.warnings)


@pytest.mark.hardware
def test_real_cuda_allocator_oom_falls_back_mid_job(cuda_backend):
    import torch
    rng=np.random.default_rng(77)
    frame=rng.uniform(10,200,(64,80))
    frames=np.stack([frame]*3)
    cfg=ReconstructionConfig(device='cpu',threads=2,batch_frames=1)
    expected=stack_source(ArraySource(frames),cfg)
    def event(result,info):
        if result.n_used==1:
            torch.cuda.set_per_process_memory_fraction(.000001)
            torch.cuda.empty_cache()
    try:
        actual=stack_source(ArraySource(frames),replace(cfg,device='gpu'),on_event=event)
        assert actual.backend=='cpu' and actual.n_used==3
        assert actual.provenance['device_report']['execution_history']==['cuda','cpu']
        np.testing.assert_allclose(actual.image,expected.image,rtol=1e-12,atol=1e-10)
    finally:
        torch.cuda.set_per_process_memory_fraction(.15)
        torch.cuda.empty_cache()
