import copy
import json
import hashlib
from pathlib import Path

import pytest

from tools.summarize_scene_selection_family import CONTRACT, summarize


@pytest.fixture
def family():
    path = Path(__file__).resolve().parents[1] / 'results/p2-full-selection-manifest/report.json'
    manifest = json.loads(path.read_text())
    entries = []
    for index, case in enumerate(manifest['cases']):
        protocol = {**copy.deepcopy(CONTRACT), 'case_index': index,
                    'case': [case['seed'], case['dr0'], case['crop']],
                    'input_sha256': case['input_sha256'], 'manifest_sha256': 'manifest',
                    'qualification_hashes': {'proof': 'hash'}, 'identities': {'source': 'fixed'}}
        rows = []
        for selection in case['selections']:
            runs = [{'maxiter': cap, 'n_iter': 310, 'converged': True, 'tolerance': 1e-5,
                     'solver': 'extended_scene_projected_acceleration_v4',
                     'reference_certificate': {'feasible': True, 'relative_solution_error_bound': 9e-6}}
                    for cap in (1500, 3000)]
            rows.append({**copy.deepcopy(selection), 'n_captured': 500, 'numerical_passed': True,
                         'runs': runs, 'relative_changes': {'latent': 0., 'detector': 0.}})
        report = {'case_index': index, 'status': 'valid', 'complete': True,
                  'source_input_unchanged': True, 'failures': [], 'rows': rows, 'q3_authorized': False}
        entries.append((protocol, report))
    return manifest, entries


def summary(family):
    manifest, entries = family
    return summarize(manifest, entries, manifest_hash='manifest', qualification_hashes={'proof': 'hash'})


def test_requires_all_twelve_cases_and_sixty_selections(family):
    result = summary(family)
    assert result['status'] == 'valid'
    assert result['cases_passed'] == 12 and result['selections_passed'] == 60
    assert result['q3_authorized'] is False
    family[1].pop()
    result = summary(family)
    assert result['status'] == 'incomplete'
    assert result['cases_passed'] == 11 and result['selections_passed'] == 55
    assert result['cases'][-1]['status'] == 'missing'


@pytest.mark.parametrize('bound', [1.01e-5, -1., float('nan'), float('inf'), float('-inf'), None, True])
def test_stale_pass_flags_do_not_override_bad_independent_certificate(family, bound):
    family[1][0][1]['rows'][0]['runs'][0]['reference_certificate']['relative_solution_error_bound'] = bound
    result = summary(family)
    assert result['status'] == 'incomplete' and result['selections_passed'] == 59


@pytest.mark.parametrize('change', [1.01e-4, -1., float('nan'), float('inf'), None])
def test_rechecks_both_image_stability_measures(family, change):
    family[1][0][1]['rows'][0]['relative_changes']['detector'] = change
    assert summary(family)['selections_passed'] == 59


@pytest.mark.parametrize('field,value', [('maxiter', 750), ('converged', False), ('n_iter', 1501),
                                        ('solver', 'different'), ('tolerance', 1e-3)])
def test_changed_or_unconverged_fit_cannot_pass(family, field, value):
    family[1][0][1]['rows'][0]['runs'][0][field] = value
    assert summary(family)['selections_passed'] == 59


@pytest.mark.parametrize('field,value', [('margin', 0), ('input_sha256', 'changed'),
                                        ('manifest_sha256', 'changed'), ('wall_budget_s', 7200),
                                        ('qualification_hashes', {}), ('identities', {'source': 'changed'})])
def test_mixed_protocol_or_source_cannot_qualify_family(family, field, value):
    family[1][0][0][field] = value
    result = summary(family)
    assert result['status'] == 'incomplete'
    assert result['cases'][0]['failures']


def test_partial_attempt_retains_passed_selections_without_passing_case(family):
    report = family[1][0][1]
    report.update(status='incomplete', complete=False, failures=['case budget exhausted'])
    report['rows'] = report['rows'][:2]
    result = summary(family)
    assert result['status'] == 'incomplete' and result['selections_passed'] == 57
    assert result['cases'][0]['execution_failures'] == ['case budget exhausted']


@pytest.mark.parametrize('change', ['duplicate', 'reordered', 'indices', 'prior', 'missing_change', 'infeasible'])
def test_selection_substitution_and_missing_evidence_cannot_pass(family, change):
    rows = family[1][0][1]['rows']
    if change == 'duplicate': rows.append(copy.deepcopy(rows[0]))
    if change == 'reordered': rows[0], rows[1] = rows[1], rows[0]
    if change == 'indices': rows[0]['indices'].reverse()
    if change == 'prior': rows[0]['mean_ridge'] *= 2
    if change == 'missing_change': del rows[0]['relative_changes']['latent']
    if change == 'infeasible': rows[0]['runs'][0]['reference_certificate']['feasible'] = False
    assert summary(family)['status'] == 'incomplete'


def test_duplicate_case_is_rejected(family):
    family[1].append(copy.deepcopy(family[1][0]))
    with pytest.raises(ValueError, match='duplicate'):
        summary(family)


def test_cli_binds_exact_input_bytes_and_preserves_existing_summary(family, tmp_path, monkeypatch):
    from tools.summarize_scene_selection_family import main
    import sys
    root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(root)
    manifest_path = root / 'results/p2-full-selection-manifest/report.json'
    proof_paths = [Path('results/p2-reference-stability') / n for n in ('report.json', 'protocol.json')]
    protocol, report = family[1][0]
    protocol['qualification_hashes'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in proof_paths}
    protocol['manifest_sha256'] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    case_dir = tmp_path / 'case-00'
    case_dir.mkdir()
    for name, record in [('protocol.json', protocol), ('report.json', report)]:
        (case_dir / name).write_text(json.dumps(record))
    output = tmp_path / 'nested' / 'summary.json'
    monkeypatch.setattr(sys, 'argv', ['summary', '--cases', str(case_dir), '--out', str(output)])
    main()
    raw = output.read_bytes()
    result = json.loads(raw)
    assert result['cases_passed'] == 1 and result['selections_passed'] == 5
    assert result['status'] == 'incomplete'
    assert result['source_reports'][str(case_dir / 'report.json')] == hashlib.sha256((case_dir / 'report.json').read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='already exists'):
        main()
    assert output.read_bytes() == raw
