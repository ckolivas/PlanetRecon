import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]/'results'


def read_case(directory, record):
    raw = (directory/record['path']).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == record['sha256']
    if record['path'].endswith('.gz'):
        raw = gzip.decompress(raw)
        assert hashlib.sha256(raw).hexdigest() == record['uncompressed_sha256']
    return json.loads(raw)


def test_complete_bounded_family_and_refinement_are_traceable():
    parent = ROOT/'r10-joint-budget-audit-v3'
    refine = ROOT/'r10-joint-budget-refinement-v2'
    report = json.loads((parent/'report.json').read_text())
    final = json.loads((refine/'report.json').read_text())
    protocol = json.loads((refine/'protocol.json').read_text())
    assert hashlib.sha256((parent/'report.json').read_bytes()).hexdigest() == protocol['parent_report_sha256']
    assert report['source_unchanged'] and final['source_unchanged']
    assert report['source_hash'] == final['source_hash']
    assert not report['q3_authorized'] and not final['q3_authorized']
    cases = [read_case(parent, r) for r in report['cases']]
    corrected = [read_case(refine, r) for r in final['cases']]
    identity = lambda c: (c['seed'], c['dr0'], c['crop'])
    assert {identity(c) for c in cases} == {(s, d, c) for s in (1001, 1002, 1003)
                                         for d in (4., 8.) for c in ('feature', 'bland')}
    def all_starts(case):
        return all(s['convergence']['converged'] and s['object_info']['converged']
                   for b in case['budgets'] for g in (b['fit']['inits'], b['fit']['holdout']['inits'])
                   for init in g.values() for s in init['stages'])
    failures = {identity(c) for c in cases if not c['bounded_convergence_passed'] or not all_starts(c)}
    assert failures == {identity(c) for c in corrected} and len(failures) == 4
    accepted = [c for c in cases if identity(c) not in failures]+corrected
    assert len(accepted) == final['combined_passed_cases'] == 12
    assert final['complete_bounded_family_passed']
    for case in accepted:
        assert case['bounded_convergence_passed']
        assert max(case['image_relative_changes'].values()) <= 1e-3
        assert case['closure_absolute_change'] <= .01
        for budget in case['budgets']:
            fit = budget['fit']
            assert fit['status'] == fit['assessment']['status'] == 'valid'
            train, select, assess = fit['holdout']['train_indices'], fit['holdout']['indices'], fit['assessment']['indices']
            assert sorted(train+select+assess) == list(range(8))
            for block in (*fit['inits'].values(), *fit['holdout']['inits'].values()):
                assert [s['M'] for s in block['stages']] == [15, 35, 60]
                assert all(s['convergence']['converged'] and s['object_info']['converged'] for s in block['stages'])


def test_resource_forecast_is_measured_and_never_authorizes_q3():
    report = json.loads((ROOT/'r10-scientific-resource-forecast/report.json').read_text())
    assert not report['q3_authorized']
    assert [r['n_diam'] for r in report['measurements']] == [16, 32, 64]
    assert len(report['forecasts']) == 12
    for row in report['measurements']:
        assert row['process_peak_rss_bytes'] > 0
        assert all(0 < g['min_s'] <= g['median_s'] <= g['max_s'] for g in row['gradient'])
