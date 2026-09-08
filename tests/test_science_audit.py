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


def test_separate_assessment_family_has_complete_disjoint_coverage():
    root = Path(__file__).resolve().parents[1]/'results/r10-development-assessment'
    manifest = json.loads((root/'manifest.json').read_text())
    assert not manifest_errors(manifest)
    assert manifest['status'] == 'diagnostic' and not manifest['q3_authorized']
    identities = set()
    for record in manifest['results']:
        path = root/record['path']
        assert sha256_bytes(path.read_bytes()) == record['sha256']
        case = json.loads(path.read_text())
        identities.add((case['seed'], case['dr0'], case['crop']))
        q = case['q2']
        train = q['holdout']['train_indices']
        select = q['holdout']['indices']
        assess = q['assessment']['indices']
        assert len(train) == 6 and len(select) == len(assess) == 1
        assert sorted(train+select+assess) == list(range(8))
        assert q['status'] == q['assessment']['status'] == 'incomplete'
        assert q['assessment']['chosen_init'] == q['chosen_init']
        for init in q['holdout']['inits'].values():
            assert [s['M'] for s in init['stages']] == [15, 35, 60]
    assert identities == {(s, d, c) for s in (1001, 1002, 1003)
                          for d in (4., 8.) for c in ('feature', 'bland')}
