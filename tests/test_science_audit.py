import json
from pathlib import Path
from planetrecon.provenance import sha256_bytes, manifest_errors


def test_bounded_audit_integrity_and_no_gate_claim():
    root=Path(__file__).resolve().parents[1]/'results/r10-development-audit'
    manifest=json.loads((root/'manifest.json').read_text())
    assert not manifest_errors(manifest)
    assert manifest['status'] == 'diagnostic' and manifest['q3_authorized'] is False
    assert len(manifest['results']) == 12
    for record in manifest['results']:
        path=root/record['path']
        assert sha256_bytes(path.read_bytes()) == record['sha256']
        case=json.loads(path.read_text())
        assert case['q2']['status'] == 'incomplete'
        assert case['E1_E2a0_relative_error'] < 1e-9
        for init in case['q2']['holdout']['inits'].values():
            assert [s['M'] for s in init['stages']] == [15,35,60]
