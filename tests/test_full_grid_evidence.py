import gzip
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]/'results'


def read(path):
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == '.gz' else raw)


def case(directory, entry):
    raw = (directory/entry['path']).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == entry['sha256']
    if entry['path'].endswith('.gz'):
        raw = gzip.decompress(raw)
        assert hashlib.sha256(raw).hexdigest() == entry['uncompressed_sha256']
    return json.loads(raw)


def test_full_grid_inputs_and_studies_keep_complete_separate_families():
    inputs = read(ROOT/'r10-full-grid-inputs/manifest.json')
    assert len(inputs['files']) == 30 and inputs['n_frames_per_sequence'] == 500
    assert inputs['development_seeds'] == [1001, 1002, 1003]
    assert inputs['evaluation_seeds'] == list(range(2001, 2013))
    assert all(not f['gate_errors'] for f in inputs['files'])
    assert not inputs['q3_authorized']
    identities = {f['path']: f['sha256'] for f in inputs['files']}
    expected = {(s, d, c) for s in (1001, 1002, 1003) for d in (4., 8.) for c in ('feature', 'bland')}
    for name in ('r10-full-gate1-admm-development', 'r10-full-gate1-admm-development-v2'):
        directory = ROOT/name
        report = read(directory/'report.json')
        assert report['complete'] and report['source_unchanged'] and not report['q3_authorized']
        cases = [case(directory, entry) for entry in report['cases']]
        assert {(c['seed'], c['dr0'], c['crop']) for c in cases} == expected
        for row in cases:
            assert row['input_sha256'] == identities[row['input_path']]
            assert row['input_unchanged']
            assert row['config']['n_frames'] == 500 and row['config']['eval_size'] == 128
            assert [run['maxiter'] for run in row['runs']] == [10000, 20000]
            valid = []
            for run in row['runs']:
                metrics = run['result']['metrics']
                assert set(metrics) == {'5', '10', '25', '50', '100'}
                converged = all(m['_info'][e]['converged'] for m in metrics.values() for e in ('E2a', 'A1o'))
                oracle = all(metrics[p]['_info']['E2a0']['match_pass'] for p in ('10', '100'))
                assert (run['result']['status'] == 'valid') == (converged and oracle)
                valid.append(converged and oracle)
                for p, m in metrics.items():
                    assert m['_subset']['n_captured'] == 500
                    assert m['_subset']['n_used'] == int(p)*5
            assert row['runs'][0]['result']['subsets'] == row['runs'][1]['result']['subsets']
            stable = max(row['image_relative_changes'].values()) <= 1e-3 and max(row['gap_absolute_changes'].values()) <= .01
            assert row['budget_convergence_passed'] == (all(valid) and stable)
        assert report['full_resolution_budget_convergence_passed'] == all(c['budget_convergence_passed'] for c in cases)


def test_full_noise_and_boundary_evidence_supports_the_reported_decision():
    directory = ROOT/'r10-full-noise-weight-sensitivity'
    report = read(directory/'report.json')
    assert report['complete'] and report['source_unchanged'] and report['numerical_controls_passed']
    rows = [case(directory, entry) for entry in report['cases']]
    assert len(rows) == 12
    boundary = read(ROOT/'r10-full-crop-model-audit/report.json')
    assert boundary['complete'] and boundary['source_unchanged'] and not boundary['q3_authorized']
    hashes = {(c['seed'], c['dr0'], c['crop']): c['input_sha256'] for c in boundary['cases']}
    for row in rows:
        assert row['input_sha256'] == hashes[row['seed'], row['dr0'], row['crop']]
        assert row['input_unchanged']
        for subset in row['subsets']:
            assert subset['converged'] and subset['budget_image_relative_change'] == 0
            assert .95 < subset['known_noise_mean_standardized_square'] < 1.05
            for fit in subset['weighted_fits']:
                assert fit['solver']['converged'] and fit['solver']['normal_relative_residual'] <= 1e-9
                assert fit['metrics']['E_H'] > subset['scalar_metrics']['E_H']
                assert fit['metrics']['image_rel_mse'] > subset['scalar_metrics']['image_rel_mse']
    for row in boundary['cases']:
        a, b = (row['regions'][name]['model_mean_standardized_square'] for name in ('whole_crop', 'interior_border_32'))
        assert a > b
