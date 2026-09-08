"""Exercise real bounded solvers and stage persistence with tiny optical scenes."""
import json
import time
from pathlib import Path
import numpy as np
from planetrecon.operators import SceneDetectorOperator
from tools.scene_study import CellBasisOperator
from tools.scene_selection_manifest import selection_rows
from tools.study_io import file_hash


class TinyObservedData:
    def __init__(self, path, indices, crop):
        self.images = {i: np.ones((2, 2)) for i in indices}
        self.variances = {i: 1. for i in indices}

    def operators(self, indices, **kwargs):
        return [CellBasisOperator(SceneDetectorOperator((2, 2), (np.ones((1, 1)),),
                                  1, (0, 0), (2, 2)), 1) for i in indices]


def inputs(tmp_path):
    path = tmp_path/'input.h5'; path.write_bytes(b'bounded synthetic operator fixture')
    scores, selections = selection_rows(np.ones((500, 6, 6)), 2.)
    manifest = {'status': 'valid', 'source_input_unchanged': True, 'cases': [
        {'seed': 1001, 'dr0': 4., 'crop': 'feature', 'n_captured': 500,
         'scores': scores, 'selections': selections, 'input_sha256': file_hash(path)}]}
    manifest_path = tmp_path/'manifest.json'; manifest_path.write_text(json.dumps(manifest))
    return path, manifest_path


def test_reference_deadline_keeps_iteration_state_then_resumes(tmp_path, monkeypatch):
    import tools.audit_scene_selections as audit
    monkeypatch.setattr(audit, 'ObservedSceneData', TinyObservedData)
    path, manifest = inputs(tmp_path)
    original = audit.solve
    def expired(problem, **kwargs):
        return original(problem, **{**kwargs, 'deadline': time.monotonic()-1})
    monkeypatch.setattr(audit, 'solve', expired)
    root = tmp_path/'study'
    failed = audit.run(path, manifest, root)
    assert failed['complete'] and failed['status'] == 'incomplete'
    assert len(list(root.glob('incomplete-*.json'))) == 4
    assert not list(root.glob('stages/*/budget-*.npz'))
    monkeypatch.setattr(audit, 'solve', original)
    resumed = audit.run(path, manifest, root, resume=True)
    assert resumed['status'] == 'valid'
    assert len(list(root.glob('stages/*/budget-*.npz'))) == 4
    assert len(list(root.glob('incomplete-*.json'))) == 4


def test_alternative_stage_resume_preserves_incomplete_computation(tmp_path, monkeypatch):
    import tools.audit_scene_lbfgsb as audit
    monkeypatch.setattr(audit, 'ObservedSceneData', TinyObservedData)
    path, manifest = inputs(tmp_path)
    original = audit.solve
    def expired(problem, **kwargs):
        return original(problem, **{**kwargs, 'deadline': time.monotonic()-1})
    monkeypatch.setattr(audit, 'solve', expired)
    root = tmp_path/'study'
    failed = audit.run(path, manifest, root)
    assert failed['complete'] and failed['status'] == 'incomplete'
    assert len(list(root.glob('stages/*/budget-*.npz'))) == 4
    def must_not_retry(*a, **k):
        raise AssertionError('incomplete recorded stages must not retry implicitly')
    monkeypatch.setattr(audit, 'solve', must_not_retry)
    resumed = audit.run(path, manifest, root, resume=True)
    assert resumed['status'] == 'incomplete' and not resumed['failures']
    assert all(not f['converged'] for row in resumed['rows'] for f in row['runs'])
