"""CUDA/CPU transition state follows the fixed frame/map checkpoint contract."""
import numpy as np
import pytest

from test_cfa_local_resume import inputs, advance
from tools.cfa_local_resume import FixedLocalRun, make_manifest, ARRAYS


def gpu_inputs(tmp_path, color='RGGB'):
    m, raw, maps = inputs(tmp_path, color)
    options = m['options']
    manifest = make_manifest(m['source_path'], m['source_sha256'], tuple(options['shape']), color,
                             m['indices'], m['qualities'], m['raw_digests'], m['map_digests'],
                             device='gpu', chunk_rows=7, gpu_chunk_rows=17)
    return manifest, raw, maps


def test_cold_gpu_failure_checkpoint_resumes_on_cpu_without_gpu_calls(tmp_path, monkeypatch):
    manifest, raw, maps = gpu_inputs(tmp_path)
    run = FixedLocalRun(manifest)
    def fail(): raise RuntimeError('no CUDA')
    monkeypatch.setattr(run.model, '_device', fail)
    advance(run, raw, maps, 1)
    path = tmp_path/'state.npz'; run.save(path)
    resumed = FixedLocalRun.load(path, manifest)
    monkeypatch.setattr(resumed.model, '_device', lambda: pytest.fail('CPU retry touched CUDA'))
    advance(resumed, raw, maps, 4)
    assert resumed.model.execution_history == ['cpu']*4
    assert resumed.model.fallback_frame == 0
    assert resumed.finish()[0].shape == (10, 12, 3)


@pytest.mark.hardware
@pytest.mark.parametrize('color', ['RGGB', 'GRBG', 'GBRG', 'BGGR'])
@pytest.mark.parametrize('cut', [0, 1, 3, 4])
def test_cuda_checkpoint_matches_same_policy_uninterrupted_bitwise(tmp_path, color, cut):
    import torch
    if not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    manifest, raw, maps = gpu_inputs(tmp_path, color)
    full = FixedLocalRun(manifest); advance(full, raw, maps, 4)
    part = FixedLocalRun(manifest); advance(part, raw, maps, cut)
    path = tmp_path/'state.npz'; part.save(path)
    resumed = FixedLocalRun.load(path, manifest); advance(resumed, raw, maps, 4)
    assert resumed.model.execution_history == ['cuda']*4
    assert resumed.model.device_identity == full.model.device_identity
    for name in ARRAYS: np.testing.assert_array_equal(getattr(full.model, name), getattr(resumed.model, name))
    for a, b in zip(full.finish()[:3], resumed.finish()[:3]): np.testing.assert_array_equal(a, b)


@pytest.mark.hardware
@pytest.mark.parametrize('cancel_retry', [False, True])
def test_resume_after_gpu_failure_keeps_transition_and_exact_mixed_result(tmp_path, monkeypatch, cancel_retry):
    import torch
    if not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    manifest, raw, maps = gpu_inputs(tmp_path)
    run = FixedLocalRun(manifest); advance(run, raw, maps, 1)
    def fail(*args):
        if cancel_retry: run.model.should_cancel = lambda: True
        raise RuntimeError('GPU frame failed')
    monkeypatch.setattr(run.model, '_cuda_products', fail)
    if cancel_retry:
        with pytest.raises(InterruptedError): advance(run, raw, maps, 2)
        run.model.should_cancel = None
        assert run.model.n_used == 1
    else:
        advance(run, raw, maps, 2)
    path = tmp_path/'state.npz'; run.save(path)
    resumed = FixedLocalRun.load(path, manifest)
    monkeypatch.setattr(resumed.model, '_device', lambda: pytest.fail('resumed CPU continuation touched CUDA'))
    advance(resumed, raw, maps, 4)
    full = FixedLocalRun(manifest); advance(full, raw, maps, 1)
    full.model.gpu_enabled = False
    advance(full, raw, maps, 4)
    assert resumed.model.execution_history == ['cuda', 'cpu', 'cpu', 'cpu']
    assert resumed.model.fallback_frame == 1
    for name in ARRAYS: np.testing.assert_array_equal(getattr(full.model, name), getattr(resumed.model, name))
    for a, b in zip(full.finish()[:3], resumed.finish()[:3]): np.testing.assert_array_equal(a, b)


@pytest.mark.hardware
def test_changed_cuda_device_identity_is_refused_before_frame_publication(tmp_path):
    import torch
    if not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    manifest, raw, maps = gpu_inputs(tmp_path)
    run = FixedLocalRun(manifest); advance(run, raw, maps, 1)
    path = tmp_path/'state.npz'; run.save(path)
    resumed = FixedLocalRun.load(path, manifest)
    resumed.model.expected_device_identity = {**resumed.model.expected_device_identity, 'name': 'different GPU'}
    with pytest.raises(ValueError, match='device/runtime'): advance(resumed, raw, maps, 2)
    assert resumed.model.n_used == 1 and resumed.model.gpu_enabled
