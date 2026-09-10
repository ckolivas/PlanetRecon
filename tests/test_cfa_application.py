"""End-to-end optional interpolation using the application's actual local maps."""
from dataclasses import replace
import json

import numpy as np
import pytest

from test_local_alignment import capture, config
from planetrecon.io.ser import SERSource
from planetrecon.pipeline.baseline import stack_source
from planetrecon.pipeline.preprocess_cache import preprocess_source
from planetrecon.pipeline.cfa_application import LocalCFA
from planetrecon.pipeline.cfa_interpolation import LocalQuadraticAccumulator
from planetrecon.result import save_snapshot, load_snapshot


@pytest.mark.parametrize('color_id', [8, 9, 10, 11])
def test_application_matches_qualified_operator_and_preserves_baseline_selection(tmp_path, monkeypatch, color_id):
    cfg = config(local_cfa_interpolation=True, stack_percent=100)
    with SERSource(capture(tmp_path, color_id)) as source:
        preprocess_source(source, cfg)
        oracle = LocalQuadraticAccumulator(source.frame_shape(), source.color_mode())
        original = LocalCFA.add
        def collect(self, index, raw, shift, quality):
            oracle.add(raw, shift, quality)
            original(self, index, raw, shift, quality)
        monkeypatch.setattr(LocalCFA, 'add', collect)
        events = []
        result = stack_source(source, cfg, on_event=lambda r, i: events.append((r, i)))
        baseline = stack_source(source, replace(cfg, local_cfa_interpolation=False))
    expected, variance, degree, ordinary = oracle.finish()
    np.testing.assert_array_equal(result.image, expected)
    assert not result.incomplete
    assert result.n_used == baseline.n_used == oracle.n_used
    assert result.provenance['reference_index'] == baseline.provenance['reference_index']
    assert result.provenance['local_alignment'] == baseline.provenance['local_alignment']
    np.testing.assert_array_equal(result.coverage, baseline.coverage)
    for c, name in enumerate('RGB'):
        np.testing.assert_array_equal(result.layer_coverage['cfa_direct_'+name], baseline.layer_coverage['cfa_direct_'+name])
        np.testing.assert_allclose(result.layer_coverage['iid_effective_samples_'+name][result.validity[..., c]],
                                   1/variance[..., c][result.validity[..., c]])
    assert events[-1][0] is result
    assert any(i.get('progress_only') for r, i in events)
    assert any(r.provenance.get('local_cfa_interpolation', {}).get('preview') for r, i in events)
    assert not result.provenance['local_cfa_interpolation']['preview']
    path = tmp_path/'result.npz'; save_snapshot(path, result)
    restored = load_snapshot(path)
    np.testing.assert_array_equal(restored.image, result.image)
    assert restored.provenance == result.provenance
    assert set(restored.layer_coverage) == set(result.layer_coverage)


@pytest.mark.parametrize('cancel_phase', ['batch', 'moments', 'finish'])
def test_application_cancel_resume_is_exact_and_keeps_template(tmp_path, monkeypatch, cancel_phase):
    cfg = config(local_cfa_interpolation=True, stack_percent=100)
    with SERSource(capture(tmp_path, 8)) as source:
        preprocess_source(source, cfg)
        whole = stack_source(source, cfg)
        stop = False
        def event(result, info):
            nonlocal stop
            if cancel_phase == 'batch' and not info.get('progress_only') and result.n_used >= 2:
                stop = True
            if cancel_phase == 'finish' and result.stage == 'local CFA final fit':
                stop = True
        original = LocalQuadraticAccumulator.stage_moments
        def interrupted(self, *args):
            nonlocal stop
            staged = original(self, *args)
            if cancel_phase == 'moments' and self.n_used >= 2:
                stop = True
            return staged
        monkeypatch.setattr(LocalQuadraticAccumulator, 'stage_moments', interrupted)
        state = tmp_path/'state.npz'
        part = stack_source(source, cfg, should_cancel=lambda: stop, on_event=event, state_checkpoint=state)
        assert part.incomplete and part.provenance['local_cfa_interpolation']['preview']
        monkeypatch.setattr(LocalQuadraticAccumulator, 'stage_moments', original)
        def no_template(*args, **kwargs): pytest.fail('rebuilt checkpoint reference')
        monkeypatch.setattr('planetrecon.pipeline.local_align.build_template', no_template)
        resumed = stack_source(source, cfg, resume_from=state)
        np.testing.assert_array_equal(resumed.image, whole.image)
        np.testing.assert_array_equal(resumed.coverage, whole.coverage)
        assert resumed.provenance['local_cfa_interpolation'] == whole.provenance['local_cfa_interpolation']
        assert resumed.n_used == whole.n_used
        with pytest.raises(ValueError, match='identity|configuration'):
            stack_source(source, replace(cfg, stack_percent=50), resume_from=state)


@pytest.mark.parametrize('damage', ['array', 'metadata'])
def test_application_rejects_corrupt_checkpoint(tmp_path, damage):
    cfg = config(local_cfa_interpolation=True, stack_percent=100)
    with SERSource(capture(tmp_path, 8)) as source:
        preprocess_source(source, cfg)
        path = tmp_path/'state.npz'
        stack_source(source, cfg, state_checkpoint=path)
        with np.load(path, allow_pickle=False) as data:
            contents = {name: data[name] for name in data.files}
        if damage == 'array':
            contents['local_rhs'][0, 0, 0] += 1
        else:
            meta = json.loads(str(contents['metadata']))
            meta['local_cfa']['frames'][0]['map_sha256'] = '0'*64
            contents['metadata'] = json.dumps(meta)
        np.savez_compressed(path, **contents)
        with pytest.raises(ValueError, match='checksum'):
            stack_source(source, cfg, resume_from=path)


def test_candidate_refuses_incompatible_inputs(tmp_path):
    with pytest.raises(ValueError, match='local alignment'):
        replace(config(), local_alignment=False, local_cfa_interpolation=True)
    with pytest.raises(ValueError, match='linear quality'):
        config(local_cfa_interpolation=True, squared_quality_weights=True)
    with SERSource(capture(tmp_path)) as source:
        with pytest.raises(ValueError, match='Bayer'):
            stack_source(source, config(local_cfa_interpolation=True))


def test_midpoint_selection_remains_optional_and_reuses_measurements(tmp_path):
    cfg = config(local_cfa_interpolation=True, stack_percent=50)
    with SERSource(capture(tmp_path, 8)) as source:
        preprocess_source(source, cfg)
        ordinary = stack_source(source, replace(cfg, local_cfa_interpolation=False))
        result = stack_source(source, cfg)
        assert result.n_used == ordinary.n_used
        assert result.provenance['preprocessing'] == ordinary.provenance['preprocessing']


@pytest.mark.hardware
@pytest.mark.parametrize('fail_gpu', [False, True])
def test_application_cuda_resume_preserves_maps_and_moment_policy(tmp_path, monkeypatch, fail_gpu):
    import torch
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    from planetrecon.pipeline.cfa_cuda import CudaLocalAccumulator
    original = CudaLocalAccumulator._cuda_products
    def products(self, *args):
        if fail_gpu and self.n_used == 2:
            raise RuntimeError('injected moment failure')
        return original(self, *args)
    monkeypatch.setattr(CudaLocalAccumulator, '_cuda_products', products)
    cfg = replace(config(local_cfa_interpolation=True), device='gpu')
    with SERSource(capture(tmp_path, 8)) as source:
        preprocess_source(source, cfg)
        whole = stack_source(source, cfg)
        stop = False
        def event(result, info):
            nonlocal stop
            if not info.get('progress_only') and result.n_used >= 4:
                stop = True
        path = tmp_path/'state.npz'
        partial = stack_source(source, cfg, on_event=event, should_cancel=lambda: stop, state_checkpoint=path)
        assert partial.incomplete
        resumed = stack_source(source, cfg, resume_from=path)
        np.testing.assert_array_equal(resumed.image, whole.image)
        assert resumed.provenance['local_cfa_interpolation'] == whole.provenance['local_cfa_interpolation']
        history = resumed.provenance['local_cfa_interpolation']['moment_execution_history']
        assert history[:2] == ['cuda']*2
        assert history[2:] == [('cpu' if fail_gpu else 'cuda')]*(resumed.n_used-2)
