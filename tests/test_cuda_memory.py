from dataclasses import replace
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from planetrecon.backends.memory import cuda_allocation_limit
from planetrecon.io.source import ArraySource
from planetrecon.pipeline.baseline import stack_source
from planetrecon.reconstruction import ReconstructionConfig


def test_limit_restored_after_exception_and_never_relaxes_existing_cap(monkeypatch):
    fractions=[.25]
    cuda=SimpleNamespace(is_available=lambda:True,
        memory=SimpleNamespace(get_allocator_backend=lambda:'native'),
        get_device_properties=lambda device:SimpleNamespace(total_memory=1024),
        get_per_process_memory_fraction=lambda device:fractions[-1],
        set_per_process_memory_fraction=lambda value,device:fractions.append(value),
        empty_cache=lambda:None,memory_reserved=lambda device:0,
        reset_peak_memory_stats=lambda device:None,max_memory_reserved=lambda device:64)
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(cuda=cuda))
    with pytest.raises(RuntimeError,match='processing failed'):
        with cuda_allocation_limit(512) as report:
            assert report['enforced'] and report['effective_bytes']==256
            assert fractions[-1]==.25
            raise RuntimeError('processing failed')
    assert fractions[-1]==.25 and report['peak_reserved_bytes']==64


def test_unavailable_budget_falls_back_without_attempting_uncapped_gpu(monkeypatch):
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:False)))
    cfg=ReconstructionConfig(device='gpu',threads=2,max_vram_bytes=1024)
    frames=np.ones((2,8,8),dtype='u2')
    result=stack_source(ArraySource(frames),cfg)
    assert result.backend=='cpu' and result.n_used==2
    assert result.provenance['device_report']['fallback']
    assert not result.provenance['cuda_allocation_budget']['enforced']
    assert any('allocation limit unavailable' in warning for warning in result.warnings)


@pytest.mark.parametrize('value',[True,0,-1,1.5,float('inf')])
def test_invalid_cuda_limits(value):
    with pytest.raises(ValueError):ReconstructionConfig(max_vram_bytes=value)


def test_cpu_and_geometry_reject_cuda_only_setting():
    with pytest.raises(ValueError):ReconstructionConfig(device='cpu',max_vram_bytes=1024)
    with pytest.raises(ValueError):ReconstructionConfig(geometry_mode='field',max_vram_bytes=1024)
