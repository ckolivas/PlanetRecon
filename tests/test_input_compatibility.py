import hashlib
import json
import pytest
from tools.input_compatibility import BRIDGE, dependency_errors, archived_input_errors


def test_reviewed_dependency_closure_is_unchanged():
    bridge = json.loads(BRIDGE.read_text())
    assert dependency_errors(bridge) == []
    for name in ('simulate', 'validate', 'constants', 'config', 'rng', 'optics', 'atmosphere', 'hdf5io', 'provenance'):
        assert f'planetrecon/{name}.py' in bridge['dependencies']


@pytest.mark.parametrize('name', ['physics', 'sampling', 'calibration', 'validation'])
def test_dependency_change_rejected(tmp_path, name):
    path = tmp_path/(name+'.py')
    path.write_text('old')
    bridge = {'dependencies': {path.name: hashlib.sha256(path.read_bytes()).hexdigest()}}
    assert not dependency_errors(bridge, tmp_path)
    path.write_text('new')
    assert dependency_errors(bridge, tmp_path)


def test_unknown_input_is_never_certified(tmp_path):
    path = tmp_path/'not-certified.h5'
    path.write_bytes(b'unknown data')
    assert 'not an exact certified member' in archived_input_errors(path)[0]
