"""Refine only failed cases from a declared joint convergence audit."""
import argparse
import json
import gzip
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def read_case(path):
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == '.gz' else raw)


def all_starts_converged(case):
    return all(stage['convergence']['converged'] and stage['object_info']['converged']
               for budget in case['budgets']
               for group in (budget['fit']['inits'], budget['fit']['holdout']['inits'])
               for init in group.values() for stage in init['stages'])


def run(parent, directory):
    import numpy as np
    from planetrecon.evaluate import load_crop, _to_jsonable
    from planetrecon.hdf5io import filename
    from planetrecon.provenance import source_hash, sha256_bytes
    from planetrecon.q2 import evaluate_q2_crop
    from planetrecon.runtime import apply_thread_limits
    from planetrecon.physics_audit import peak_rss_bytes
    apply_thread_limits(2)
    parent_report = json.loads((parent/'report.json').read_text())
    old_protocol = json.loads((parent/'protocol.json').read_text())
    cases = [r for r in parent_report['cases'] if not r['bounded_convergence_passed']
             or not all_starts_converged(read_case(parent/r['path']))]
    directory.mkdir(parents=True, exist_ok=False)
    identity = source_hash()
    protocol = {'status': 'diagnostic', 'q3_authorized': False, 'parent': str(parent),
                'parent_report_sha256': sha256_bytes((parent/'report.json').read_bytes()),
                'cases': [{k: r[k] for k in ('seed', 'dr0', 'crop')} for r in cases],
                'budgets': [{'outer': 36, 'phase': 192}, {'outer': 72, 'phase': 384}],
                'image_tolerance': old_protocol['image_stability_tolerance'],
                'closure_tolerance': old_protocol['closure_stability_tolerance'], 'source_hash': identity,
                'design': 'Refine failures in either selected-model checks or any unselected initialization/stage; same inputs, model, starts, partitions and tolerances; no Q3 or full-resolution acceptance.'}
    (directory/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    rows = []
    for case in cases:
        path = parent/'inputs'/filename(case['seed'], case['dr0'])
        previous = read_case(parent/case['path'])
        if sha256_bytes(path.read_bytes()) != previous['input_sha256']:
            raise ValueError(f'input changed: {path}')
        cfg, crop, extras = load_crop(path, case['crop'])
        runs, images = [], []
        for budget in protocol['budgets']:
            print(f"refine {case['seed']} {case['dr0']} {case['crop']} {budget}", flush=True)
            started = time.monotonic()
            fit = evaluate_q2_crop(cfg, crop, extras, holdout=True, tv_mu=0.,
                    m_grid=(15, 35, 60), outer_iters=(budget['outer'],)*3, alpha_iters=budget['phase'],
                    frame_workers=2, return_reconstructions=True)
            images.append(fit.pop('_reconstructions'))
            runs.append({'budget': budget, 'wall_s': time.monotonic()-started, 'fit': fit,
                         'cumulative_process_peak_rss_bytes': peak_rss_bytes()})
            print(f"  {fit['status']}; {runs[-1]['wall_s']:.2f}s", flush=True)
        delta = {k: float(np.linalg.norm(images[1][k]-images[0][k])/max(np.linalg.norm(images[1][k]), 1e-12)) for k in images[0]}
        cross = [float(np.linalg.norm(im['zero']-im['subset'])/max(np.linalg.norm(im['zero']), 1e-12)) for im in images]
        closure_delta = abs(runs[1]['fit']['C']-runs[0]['fit']['C'])
        passed = (all(r['fit']['status'] == 'valid' for r in runs) and max(delta.values()) <= protocol['image_tolerance']
                  and max(cross) <= protocol['image_tolerance'] and closure_delta <= protocol['closure_tolerance']
                  and all_starts_converged({'budgets': runs}))
        row = {'seed': case['seed'], 'dr0': case['dr0'], 'crop': case['crop'], 'budgets': runs,
               'image_relative_changes': delta, 'cross_start_relative_difference': cross,
               'closure_absolute_change': closure_delta, 'bounded_convergence_passed': passed}
        dest = directory/f"case-{case['seed']}-{int(case['dr0'])}-{case['crop']}.json"
        dest.write_text(json.dumps(_to_jsonable(row), indent=2, allow_nan=False)+'\n')
        rows.append({'path': dest.name, 'sha256': sha256_bytes(dest.read_bytes()),
                     'bounded_convergence_passed': passed, 'max_image_relative_change': max(delta.values())})
    unchanged = identity == source_hash()
    passed_parent = len(parent_report['cases'])-len(cases)
    report = {'status': 'diagnostic', 'q3_authorized': False, 'source_hash': identity,
              'source_unchanged': unchanged, 'cases': rows, 'previously_passed_cases': passed_parent,
              'combined_passed_cases': passed_parent+sum(r['bounded_convergence_passed'] for r in rows),
              'complete_bounded_family_passed': bool(parent_report['complete'] and parent_report['source_unchanged'] and unchanged
                                                    and all(r['bounded_convergence_passed'] for r in rows))}
    (directory/'report.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    run(args.parent, args.out)
