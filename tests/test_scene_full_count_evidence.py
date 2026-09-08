import hashlib
import json
from pathlib import Path
import pytest
from tools.audit_scene_selections import validate_case

ROOT = Path(__file__).resolve().parents[1]/'results'


@pytest.mark.parametrize('name', ['p5-full-count-profile', 'p2-full-selection-manifest',
                                 'p5-parallel-reference', 'p2-selection-endpoints-reference'])
def test_full_count_archived_json_integrity(name):
    root = ROOT/name
    for path, expected in json.loads((root/'checksums.json').read_text()).items():
        assert hashlib.sha256((root/path).read_bytes()).hexdigest() == expected


def test_every_full_development_selection_present_and_consistent():
    manifest = json.loads((ROOT/'p2-full-selection-manifest/report.json').read_text())
    assert len(manifest['cases']) == 12
    for seed in (1001, 1002, 1003):
        for dr0 in (4., 8.):
            for crop in ('feature', 'bland'):
                case = validate_case(manifest, seed, dr0, crop)
                assert [r['n_used'] for r in case['selections']] == [25, 50, 125, 250, 500]
    assert manifest['q3_authorized'] is False


def test_full_count_parity_and_cache_churn_are_preserved():
    report = json.loads((ROOT/'p5-full-count-profile/report.json').read_text())
    assert report['status'] == 'valid' and report['source_input_unchanged']
    assert [r['count'] for r in report['rows']] == [11, 100, 500]
    for row in report['rows']:
        assert row['passed'] and max(row['relative_errors'].values()) <= 1e-10
        before, after = row['cache_before_normals'], row['cache_after_normals']
        assert after['peak_bytes'] <= 256*1024**2
        assert after['misses']-before['misses'] == (0 if row['count'] == 11 else 2*row['count'])
    assert report['q3_authorized'] is False


def test_parallel_reference_and_incomplete_endpoint_scope():
    parallel = json.loads((ROOT/'p5-parallel-reference/report.json').read_text())
    assert parallel['status'] == 'valid' and parallel['source_input_unchanged']
    for row in parallel['rows']:
        assert row['normal_bitwise_equal'] and row['linear_bitwise_equal']
        assert row['pending_frames']['peak_pending_frames'] <= row['workers']
    root = ROOT/'p2-selection-endpoints-reference'
    report = json.loads((root/'report.json').read_text())
    assert report['complete'] and report['status'] == 'incomplete'
    assert [r['numerical_passed'] for r in report['rows']] == [True, False]
    failed = [json.loads(p.read_text()) for p in root.glob('incomplete-100-*.json')]
    assert {r['maxiter'] for r in failed} == {750, 1500}
    assert all(not r['converged'] and r['reason'] == 'wall_budget' for r in failed)


def test_alternative_and_comparison_retain_incomplete_full_count_result():
    for name in ('p2-selection-endpoints-lbfgsb', 'p2-selection-endpoints-comparison'):
        root = ROOT/name
        for path, expected in json.loads((root/'checksums.json').read_text()).items():
            assert hashlib.sha256((root/path).read_bytes()).hexdigest() == expected
    alternative = json.loads((ROOT/'p2-selection-endpoints-lbfgsb/report.json').read_text())
    assert alternative['complete'] and alternative['status'] == 'incomplete'
    assert [r['numerical_passed'] for r in alternative['rows']] == [True, False]
    assert all(f['reason'] == 'wall_budget' for f in alternative['rows'][1]['runs'])
    comparison = json.loads((ROOT/'p2-selection-endpoints-comparison/report.json').read_text())
    assert comparison['certified_objectives_consistent']
    assert not comparison['reference_endpoints_passed'] and not comparison['candidate_endpoints_passed']
    for source, expected in comparison['source_reports'].items():
        assert hashlib.sha256((ROOT.parent/source).read_bytes()).hexdigest() == expected
    assert all(f['objective_consistent_with_bounds'] is True for f in comparison['rows'][0]['fits'])
    assert all(f['objective_consistent_with_bounds'] is None for f in comparison['rows'][1]['fits'])
