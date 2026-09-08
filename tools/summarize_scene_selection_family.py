"""Recheck the full observed numerical family without promoting partial evidence."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.audit_scene_selection_family import select_case, require_qualified_stability
from tools.study_io import file_hash, write_json


CONTRACT = {
    'fractions': [5, 10, 25, 50, 100], 'budgets': [1500, 3000],
    'budget_unit': 'iterations', 'margin': 64, 'cell_factor': 1,
    'sum_native_strength': .0003, 'tolerance': 1e-5, 'image_tolerance': 1e-4,
    'cache_bytes': 4 * 1024**3, 'headroom_bytes': 1024**3,
    'threads': 2, 'cpu_frame_workers': 8,
    'fit_budget_s': {'5': 300., '10': 300., '25': 600., '50': 900., '100': 900.},
    'wall_budget_s': 3600., 'q3_authorized': False,
}


def bounded(value, maximum):
    return (isinstance(value, (float, int)) and not isinstance(value, bool)
            and math.isfinite(value) and 0 <= value <= maximum)


def selection_passes(row, selection):
    if row.get('numerical_passed') is not True or row.get('n_captured') != 500:
        return False
    if any(row.get(k) != selection[k] for k in ('fraction', 'indices', 'n_used', 'mean_ridge')):
        return False
    runs = row.get('runs', [])
    if len(runs) != 2:
        return False
    for fit, cap in zip(runs, CONTRACT['budgets']):
        n = fit.get('n_iter')
        if (fit.get('maxiter') != cap or fit.get('converged') is not True
                or fit.get('solver') != 'extended_scene_projected_acceleration_v4'
                or fit.get('tolerance') != CONTRACT['tolerance']
                or not isinstance(n, int) or isinstance(n, bool) or not 0 <= n <= cap):
            return False
        certificate = fit.get('reference_certificate', {})
        if (certificate.get('feasible') is not True
                or not bounded(certificate.get('relative_solution_error_bound'), CONTRACT['tolerance'])):
            return False
    changes = row.get('relative_changes', {})
    return all(bounded(changes.get(k), CONTRACT['image_tolerance']) for k in ('latent', 'detector'))


def summarize(manifest, entries, *, manifest_hash, qualification_hashes):
    """Entries are (protocol, report) pairs; absent cases remain explicitly missing."""
    cases = [select_case(manifest, index) for index in range(12)]
    by_index = {}
    for protocol, report in entries:
        index = protocol.get('case_index')
        if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < 12:
            raise ValueError('invalid case index in family report')
        if index in by_index:
            raise ValueError('duplicate case report')
        by_index[index] = (protocol, report)
    # All recorded cases must share source, runtime and prerequisites. A changed
    # implementation requires a new family identity, even if each fit passed.
    identities = [p.get('identities') for p, _ in entries]
    same_identity = bool(identities and identities[0] and all(i == identities[0] for i in identities))
    rows = []
    for index, case in enumerate(cases):
        expected_case = [case['seed'], case['dr0'], case['crop']]
        result = {'case_index': index, 'case': expected_case, 'status': 'missing',
                  'selections_passed': 0, 'selections': [], 'failures': []}
        if index not in by_index:
            rows.append(result)
            continue
        protocol, report = by_index[index]
        failures = result['failures']
        if any(protocol.get(k) != v for k, v in CONTRACT.items()):
            failures.append('declared numerical or execution contract differs')
        if not same_identity:
            failures.append('source/runtime identities differ or are missing')
        if (protocol.get('case') != expected_case or protocol.get('input_sha256') != case['input_sha256']
                or protocol.get('manifest_sha256') != manifest_hash
                or protocol.get('qualification_hashes') != qualification_hashes):
            failures.append('case/input/manifest/prerequisite identity differs')
        if report.get('source_input_unchanged') is not True or report.get('case_index') != index:
            failures.append('unchanged matching case report required')
        if report.get('q3_authorized') is not False:
            failures.append('unexpected scientific authorization')
        observed = report.get('rows', [])
        fractions = [r.get('fraction') for r in observed]
        # Partial reports must still be an ordered prefix, with no duplicated or
        # substituted selections hidden by a dictionary lookup.
        if fractions != CONTRACT['fractions'][:len(fractions)]:
            failures.append('missing, duplicate or reordered selection rows')
        for selection in case['selections']:
            matches = [r for r in observed if r.get('fraction') == selection['fraction']]
            passed = (not failures and len(matches) == 1 and selection_passes(matches[0], selection))
            result['selections'].append({'fraction': selection['fraction'], 'passed': passed})
        result['selections_passed'] = sum(s['passed'] for s in result['selections'])
        result['execution_failures'] = report.get('failures', [])
        complete = (report.get('status') == 'valid' and report.get('complete') is True
                    and not result['execution_failures'] and result['selections_passed'] == 5)
        result['status'] = 'valid' if complete else 'incomplete'
        rows.append(result)
    passed = sum(r['status'] == 'valid' for r in rows)
    return {'status': 'valid' if passed == 12 else 'incomplete', 'cases_required': 12,
            'cases_present': len(entries), 'cases_passed': passed, 'selections_required': 60,
            'selections_passed': sum(r['selections_passed'] for r in rows), 'cases': rows,
            'source_runtime_identity_consistent': same_identity,
            'scope': 'Known-transfer numerical family only; no scientific prior, blind reconstruction or production qualification.',
            'q3_authorized': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=Path, nargs='+', required=True)
    parser.add_argument('--manifest', type=Path, default=Path('results/p2-full-selection-manifest/report.json'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError('summary output already exists; retain earlier evidence')
    hashes = {}
    def read_record(path):
        raw = path.read_bytes()
        hashes[str(path)] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)
    proof_paths = [Path('results/p2-reference-stability') / n for n in ('report.json', 'protocol.json')]
    proof, protocol = [read_record(p) for p in proof_paths]
    require_qualified_stability(proof, protocol)
    qualification = {str(p): hashes[str(p)] for p in proof_paths}
    manifest = read_record(args.manifest)
    entries = []
    for root in args.cases:
        paths = [root / n for n in ('protocol.json', 'report.json')]
        entries.append(tuple(read_record(p) for p in paths))
    report = summarize(manifest, entries,
                       manifest_hash=hashes[str(args.manifest)], qualification_hashes=qualification)
    report['source_reports'] = hashes
    report['analysis_sha256'] = file_hash(Path(__file__))
    if hashes != {p: file_hash(p) for p in hashes}:
        raise ValueError('report inputs changed during aggregation')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, report)
    print(json.dumps({k: report[k] for k in ('status', 'cases_passed', 'selections_passed')}))


if __name__ == '__main__':
    main()
