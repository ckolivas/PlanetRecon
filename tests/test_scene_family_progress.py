import hashlib
import json
from pathlib import Path

import pytest

from tools.scene_family_progress import inspect_progress


@pytest.fixture
def records(tmp_path):
    root = Path(__file__).resolve().parents[1]
    archive = root / 'results/p2-selection-family/case-00'
    for name in ('protocol.json', 'fraction-5.json'):
        (tmp_path / name).write_bytes((archive / name).read_bytes())
    return tmp_path, root / 'results/p2-full-selection-manifest/report.json'


def test_partial_progress_is_read_only_and_does_not_infer_liveness_or_pass(records):
    directory, manifest = records
    before = {str(p): p.read_bytes() for p in directory.rglob('*') if p.is_file()}
    result = inspect_progress(directory, manifest)
    assert result['completed_selection_records'] == 1
    assert result['selections_with_passing_recorded_checks'] == 1
    assert result['final_report_available'] is False
    assert result['reported_case_status'] is None
    assert result['process_liveness'] == 'unknown'
    assert result['case_qualification'] == 'use_final_family_summary'
    assert result['q3_authorized'] is False
    assert before == {str(p): p.read_bytes() for p in directory.rglob('*') if p.is_file()}
    for path, digest in result['record_sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest


def test_small_optimizer_bound_never_counts_as_independent_selection_pass(records):
    directory, manifest = records
    (directory / 'trace-10-1500.json').write_text(json.dumps({'trace': [
        {'iteration': 100, 'relative_solution_error_bound': 1e-8}]}))
    result = inspect_progress(directory, manifest)
    assert result['selections_with_passing_recorded_checks'] == 1
    row = result['rows'][1]
    assert row['recorded_checks_pass'] is None
    assert row['fits'][0]['last_recorded_iteration'] == 100
    assert row['fits'][0]['optimizer_distance_bound'] == 1e-8


def test_stored_stage_does_not_replace_independent_certificates(records):
    directory, manifest = records
    stage = directory / 'stages/10'
    stage.mkdir(parents=True)
    (stage / 'budget-1500.json').write_text(json.dumps({
        'status': 'completed', 'solver_converged': True, 'actual_iterations': 440}))
    result = inspect_progress(directory, manifest)
    assert result['rows'][1]['fits'][0]['stored_stage_available'] is True
    assert result['rows'][1]['recorded_checks_pass'] is None


@pytest.mark.parametrize('change', ['certificate', 'stability', 'indices'])
def test_rechecks_completed_selection_evidence(records, change):
    directory, manifest = records
    path = directory / 'fraction-5.json'
    row = json.loads(path.read_text())
    if change == 'certificate': row['runs'][0]['reference_certificate']['relative_solution_error_bound'] = .1
    if change == 'stability': row['relative_changes']['latent'] = .1
    if change == 'indices': row['indices'].reverse()
    path.write_text(json.dumps(row))
    result = inspect_progress(directory, manifest)
    assert result['completed_selection_records'] == 1
    assert result['selections_with_passing_recorded_checks'] == 0


def test_final_incomplete_report_and_failures_are_preserved(records):
    directory, manifest = records
    report = {'status': 'incomplete', 'failures': ['wall budget exhausted']}
    (directory / 'report.json').write_text(json.dumps(report))
    result = inspect_progress(directory, manifest)
    assert result['final_report_available'] is True
    assert result['reported_case_status'] == 'incomplete'
    assert result['reported_execution_failures'] == report['failures']
    assert result['case_qualification'] == 'use_final_family_summary'


@pytest.mark.parametrize('field,value', [('margin', 0), ('manifest_sha256', 'changed'), ('budgets', [750, 1500])])
def test_changed_progress_contract_is_rejected(records, field, value):
    directory, manifest = records
    path = directory / 'protocol.json'
    protocol = json.loads(path.read_text()); protocol[field] = value
    path.write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match='frozen family'):
        inspect_progress(directory, manifest)


@pytest.mark.parametrize('iteration,bound', [(1501, 1e-6), (True, 1e-6), (100, -1.), (100, float('nan'))])
def test_corrupt_progress_is_rejected(records, iteration, bound):
    directory, manifest = records
    (directory / 'trace-10-1500.json').write_text(json.dumps({'trace': [
        {'iteration': iteration, 'relative_solution_error_bound': bound}]}))
    with pytest.raises(ValueError, match='optimizer progress'):
        inspect_progress(directory, manifest)
