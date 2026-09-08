import json
import numpy as np
import pytest
from tools.audit_scene_conditioning import frozen_checkpoint
from tools.scene_iteration_state import IterationCheckpoint


def checkpoint(tmp_path, *, iteration=142, shape=(1024, 912)):
    protocol = {'frozen': 'input, numerical source and protocol identity'}
    selection = {'fraction': 100, 'n_used': 500}
    identity = {'context': {'protocol': protocol, 'selection': selection},
                'maxiter': 750, 'scaling': 'diagonal', 'shape': [1024, 912],
                'solver': 'extended_scene_projected_acceleration_v4', 'tolerance': 1e-5}
    path = tmp_path/'state.npz'
    IterationCheckpoint(path, identity).save(iteration, np.zeros(shape), np.zeros(shape), 100.)
    return path, protocol, selection


def test_frozen_state_validates_without_modifying_bytes(tmp_path):
    from tools.study_io import file_hash
    path, protocol, selection = checkpoint(tmp_path)
    before = file_hash(path)
    state = frozen_checkpoint(path, protocol, selection)
    assert state['iteration'] == 142 and state['x'].shape == (1024, 912)
    assert file_hash(path) == before


@pytest.mark.parametrize('change', ['protocol', 'selection', 'iteration', 'shape', 'corruption'])
def test_changed_or_corrupt_checkpoint_is_not_a_frozen_control(tmp_path, change):
    path, protocol, selection = checkpoint(tmp_path, iteration=143 if change == 'iteration' else 142,
                                          shape=(2, 2) if change == 'shape' else (1024, 912))
    if change == 'protocol': protocol['frozen'] = 'different'
    if change == 'selection': selection['n_used'] = 25
    if change == 'corruption':
        with np.load(path, allow_pickle=False) as z:
            data = {k: z[k].copy() for k in z.files}
        data['x'][0, 0] = 1.
        np.savez_compressed(path, **data)
    with pytest.raises(ValueError):
        frozen_checkpoint(path, protocol, selection)
