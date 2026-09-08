import json
import time
import numpy as np
import pytest

from tools.experiment_stages import StageStore
from test_full_gate1_audit import small_crop
from planetrecon.evaluate import evaluate_crop


def test_partial_crop_reuses_exact_completed_stages(tmp_path):
    store = StageStore(tmp_path/'stages', {'input': 'sha', 'budget': 1})
    calls = []
    def interrupted(name, compute):
        calls.append(name)
        if name == 'p10-A1o':
            raise KeyboardInterrupt()
        return store.run(name, compute)
    with pytest.raises(KeyboardInterrupt):
        evaluate_crop(*small_crop(), maxiter=1, require_convergence=False, stage_runner=interrupted)
    completed = list((tmp_path/'stages').glob('p*.json'))
    assert len(completed) == 5
    def resumed(name, compute):
        if name in calls[:-1]:
            compute = lambda: pytest.fail('completed solve repeated')
        return store.run(name, compute)
    result = evaluate_crop(*small_crop(), maxiter=1, require_convergence=False,
                           return_reconstructions=True, stage_runner=resumed)
    direct = evaluate_crop(*small_crop(), maxiter=1, require_convergence=False, return_reconstructions=True)
    for p, block in result.pop('_reconstructions').items():
        for key, image in block.items():
            np.testing.assert_array_equal(image, direct['_reconstructions'][p][key])
    direct.pop('_reconstructions')
    assert result == direct
    assert result['status'] == 'incomplete'


@pytest.mark.parametrize('key', ['input', 'physics', 'sampling', 'calibration', 'solver', 'protocol'])
def test_changed_dependencies_rejected(tmp_path, key):
    identity = {k: 'old' for k in ('input', 'physics', 'sampling', 'calibration', 'solver', 'protocol')}
    StageStore(tmp_path, identity)
    with pytest.raises(ValueError, match='identity mismatch'):
        StageStore(tmp_path, {**identity, key: 'new'})


def test_failure_budget_corruption_and_orphan_payload(tmp_path):
    store = StageStore(tmp_path, {'id': 1})
    def fail():
        raise RuntimeError('solver failed')
    with pytest.raises(RuntimeError):
        store.run('failed', fail)
    assert len(list(tmp_path.glob('failed.failure-*.json'))) == 1
    (tmp_path/'ok.npz').write_bytes(b'uncommitted orphan')
    store.run('ok', lambda: (np.ones((2, 2)), {'converged': False, 'n_iter': 1}))
    record = json.loads((tmp_path/'ok.json').read_text())
    assert record['solver_converged'] is False
    store.deadline = time.monotonic()-1
    store.run('ok', fail)  # completed work can be read after deadline
    with pytest.raises(TimeoutError):
        store.run('new', fail)
    (tmp_path/'ok.npz').write_bytes(b'broken')
    with pytest.raises(ValueError, match='corrupt stage'):
        store.run('ok', fail)
