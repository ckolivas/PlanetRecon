"""Protect the archived numerical claims and exact case coverage."""
import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / 'results'
STUDIES = ['p2-prior-selection', 'p2-domain-sensitivity', 'p2-sampling-sensitivity',
           'p2-photons-sensitivity', 'p2-prior-extension', 'p2-development-numerical-pilot']


@pytest.mark.parametrize('name', STUDIES)
def test_archived_study_bytes_and_scope(name):
    root = ROOT / name
    checks = json.loads((root / 'checksums.json').read_text())
    for path, expected in checks.items():
        assert hashlib.sha256((root / path).read_bytes()).hexdigest() == expected
    report = json.loads((root / 'report.json').read_text())
    assert report['source_input_unchanged']
    assert report['q3_authorized'] is False


def test_complete_pilot_has_independent_certificates_and_doubled_budgets():
    report = json.loads((ROOT / 'p2-development-numerical-pilot/report.json').read_text())
    assert report['complete'] and not report['failures']
    cases = report['cases']
    assert len(cases) == 12
    assert {(c['seed'], c['dr0'], c['crop']) for c in cases} == {
        (s, d, c) for s in (1001, 1002, 1003) for d in (4., 8.) for c in ('feature', 'bland')}
    for case in cases:
        assert case['numerical_passed'] and case['input_unchanged']
        assert len(set(case['indices'])) == 11 < case['n_captured'] == 500
        assert [r['maxiter'] for r in case['runs']] == [750, 1500]
        assert max(case['relative_changes'].values()) <= 1e-4
        for run in case['runs']:
            assert run['converged']
            assert run['reference_certificate']['feasible']
            assert run['reference_certificate']['relative_solution_error_bound'] <= 1e-5
