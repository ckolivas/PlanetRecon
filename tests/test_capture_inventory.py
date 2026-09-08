import json
import numpy as np
import pytest
from planetrecon.io.ser import write_ser, COLOR_RGGB
from tools.inventory_captures import inventory


def fixture(tmp_path):
    path = tmp_path/'private-name.ser'
    write_ser(path, np.ones((3, 4, 4), dtype=np.uint8), color_id=COLOR_RGGB,
              observer='private observer', instrument='private instrument',
              timestamps=np.array([10000000, 11000000, 12000000]))
    return {'id': 'capture-1', 'path': str(path), 'target': 'Mars'}


def test_intake_omits_private_fields_and_preserves_unknown_permissions(tmp_path):
    entry = fixture(tmp_path)
    result = inventory([entry]); record = result['records'][0]
    assert result['status'] == 'hashed_inventory' and not result['scientific_qualification']
    assert len(record['capture_sha256']) == 64
    assert record['timing']['duration_s'] == pytest.approx(.2)
    assert record['effective_colour'] == 'RGGB'
    assert set(record['permissions'].values()) == {'unknown'}
    assert record['night_group'] is None and record['camera_group'] is None
    encoded = json.dumps(result)
    assert 'private' not in encoded and str(tmp_path) not in encoded


def test_metadata_inventory_cannot_identify_pixels(tmp_path):
    result = inventory([fixture(tmp_path)], metadata_only=True)
    assert result['status'] == 'metadata_inventory'
    assert result['records'][0]['capture_sha256'] is None
    assert not result['scientific_qualification']


def test_explicit_colour_and_permissions_require_evidence(tmp_path):
    entry = fixture(tmp_path)
    entry['bayer_override'] = 'mono'
    with pytest.raises(ValueError, match='interpretation evidence'): inventory([entry])
    entry['interpretation_evidence'] = 'Owner correction for this fixture'
    assert inventory([entry])['records'][0]['effective_colour'] == 'mono'
    entry['permissions'] = {'redistribute_fixture': 'allowed'}
    with pytest.raises(ValueError, match='permission evidence'): inventory([entry])
    entry['permission_evidence'] = 'Owner authorizes this synthetic test fixture'
    assert inventory([entry])['records'][0]['permissions']['redistribute_fixture'] == 'allowed'


def test_duplicate_identifiers_rejected(tmp_path):
    entry = fixture(tmp_path)
    with pytest.raises(ValueError, match='unique'): inventory([entry, entry])


def test_file_change_during_hash_rejected(tmp_path, monkeypatch):
    import tools.inventory_captures as module
    entry = fixture(tmp_path)
    def changed(path):
        with path.open('ab') as f: f.write(b'changed')
        return '0'*64
    monkeypatch.setattr(module, 'file_hash', changed)
    with pytest.raises(ValueError, match='changed during'): inventory([entry])


def test_bad_timestamps_are_reported_without_inventing_duration(tmp_path):
    entry = fixture(tmp_path)
    write_ser(entry['path'], np.ones((4, 4, 4), dtype=np.uint8),
              timestamps=np.array([10000000, 10000000, 9000000, 12000000]))
    record = inventory([entry])['records'][0]
    assert record['timing']['status'] == 'invalid' and record['timing']['duration_s'] is None
    assert record['timestamp_diagnostics'] == {'duplicate_intervals': 1, 'reversed_intervals': 1, 'nonpositive_ticks': 0}
