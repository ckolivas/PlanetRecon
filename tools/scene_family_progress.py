"""Read atomic family progress records without inferring process liveness or a pass."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.audit_scene_selection_family import select_case
from tools.summarize_scene_selection_family import CONTRACT, bounded, selection_passes


def inspect_progress(directory, manifest_path):
    directory, manifest_path = Path(directory), Path(manifest_path)
    hashes = {}

    def read(path):
        raw = path.read_bytes()
        hashes[str(path)] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    protocol = read(directory / 'protocol.json')
    manifest = read(manifest_path)
    case = select_case(manifest, protocol['case_index'])
    if (any(protocol.get(k) != v for k, v in CONTRACT.items())
            or protocol.get('case') != [case['seed'], case['dr0'], case['crop']]
            or protocol.get('input_sha256') != case['input_sha256']
            or protocol.get('manifest_sha256') != hashes[str(manifest_path)]):
        raise ValueError('progress protocol does not match the frozen family')

    rows = []
    for selection in case['selections']:
        fraction = selection['fraction']
        row_path = directory / f'fraction-{fraction}.json'
        row = read(row_path) if row_path.exists() else None
        fits = []
        for cap in CONTRACT['budgets']:
            stage_path = directory / 'stages' / str(fraction) / f'budget-{cap}.json'
            trace_path = directory / f'trace-{fraction}-{cap}.json'
            stage = read(stage_path) if stage_path.exists() else None
            trace = read(trace_path)['trace'] if trace_path.exists() else []
            last = trace[-1] if trace else None
            if last is not None and (not isinstance(last.get('iteration'), int)
                    or isinstance(last['iteration'], bool) or not 0 <= last['iteration'] <= cap
                    or not bounded(last.get('relative_solution_error_bound'), float('inf'))):
                raise ValueError('invalid optimizer progress record')
            fits.append({'cap': cap, 'stored_stage_available': stage is not None,
                         'stored_stage_status': stage.get('status') if stage else None,
                         'stored_iterations': stage.get('actual_iterations') if stage else None,
                         'last_recorded_iteration': last['iteration'] if last else None,
                         'optimizer_distance_bound': last['relative_solution_error_bound'] if last else None})
        rows.append({'fraction': fraction, 'n_used': selection['n_used'],
                     'completed_selection_record_available': row is not None,
                     'recorded_checks_pass': selection_passes(row, selection) if row else None,
                     'fits': fits})
    report_path = directory / 'report.json'
    report = read(report_path) if report_path.exists() else None
    return {
        'case_index': protocol['case_index'], 'case': protocol['case'],
        'final_report_available': report is not None,
        'reported_case_status': report.get('status') if report else None,
        'reported_execution_failures': report.get('failures') if report else None,
        'completed_selection_records': sum(r['completed_selection_record_available'] for r in rows),
        'selections_with_passing_recorded_checks': sum(r['recorded_checks_pass'] is True for r in rows),
        'selections_required': 5, 'rows': rows, 'record_sha256': hashes,
        'process_liveness': 'unknown', 'case_qualification': 'use_final_family_summary',
        'scope': ('Read-only progress from individually atomic records, which may advance during inspection. '
                  'Optimizer traces are not independent CPU certificates. No input/image checksum or '
                  'whole-study source verification is performed here. A missing final report or an old '
                  'trace does not prove that a process has stopped; check the existing execution session '
                  'before starting or resuming a study.'),
        'q3_authorized': False,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--manifest', type=Path, default=Path('results/p2-full-selection-manifest/report.json'))
    args = parser.parse_args()
    print(json.dumps(inspect_progress(args.directory, args.manifest), indent=2, allow_nan=False))
