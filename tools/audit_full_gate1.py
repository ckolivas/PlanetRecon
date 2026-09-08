"""Compare Gate-1 solver budgets on complete, certified full-resolution families.

Generate inputs with validate-dev/eval first. Reports are immutable; a failed
study must be retained and a refinement written to a new directory.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    from planetrecon.evaluate import _to_jsonable
    from tools.experiment_stages import atomic_write
    atomic_write(path, (json.dumps(_to_jsonable(value), indent=2, allow_nan=False)+'\n').encode())


def compare(runs, images, image_tolerance, gap_tolerance):
    import numpy as np
    changes = {
        f'{p}/{name}': float(np.linalg.norm(images[1][p][name]-image)
                            / max(np.linalg.norm(images[1][p][name]), 1e-12))
        for p, block in images[0].items() for name, image in block.items()
    }
    gaps = {gap: abs(runs[1]['result'][gap]['g']-runs[0]['result'][gap]['g'])
            for gap in ('G1', 'G2', 'G3_m')}
    finite = all(np.isfinite(v) for v in (*changes.values(), *gaps.values()))
    return {
        'image_relative_changes': changes, 'gap_absolute_changes': gaps,
        'budget_convergence_passed': bool(finite and all(r['result']['status'] == 'valid' for r in runs)
            and max(changes.values()) <= image_tolerance and max(gaps.values()) <= gap_tolerance),
    }


def complete_family(rows, seeds, regimes):
    expected = {(s, d, c) for s in seeds for d in regimes for c in ('feature', 'bland')}
    actual = [(r['seed'], r['dr0'], r['crop']) for r in rows]
    return len(actual) == len(expected) and set(actual) == expected


def evaluate_case(payload):
    from planetrecon import constants as C
    from planetrecon.evaluate import evaluate_crop, load_crop
    from planetrecon.hdf5io import gate_errors
    from planetrecon.physics_audit import peak_rss_bytes
    from planetrecon.runtime import apply_thread_limits
    apply_thread_limits(2)
    path, crop_name, directory, protocol, deadline = payload
    if protocol.get('solver') == 'admm':
        # A1o delegates to estimators.e2a; evaluate_crop imported its own alias.
        # Both must use this explicitly recorded experimental candidate.
        import planetrecon.estimators as estimators
        import planetrecon.evaluate as evaluation
        from tools.quadratic_admm import solve
        estimators.e2a = evaluation.e2a = solve
    identity = file_hash(path)
    if protocol.get('input_compatibility') == 'archived-full-grid':
        from tools.input_compatibility import archived_input_errors
        errors = archived_input_errors(path)
    else:
        errors = gate_errors(path)
    if errors:
        raise ValueError(f'{path}: {errors}')
    cfg, crop, extras = load_crop(path, crop_name)
    if cfg.n_frames != C.N_FRAMES or cfg.eval_size != C.EVAL_SIZE or cfg.n_diam != C.DEFAULT_N_DIAM:
        raise ValueError('full-resolution audit requires the locked 500-frame, 128-pixel, 64-sample pupil configuration')
    started = time.monotonic()
    runs, images = [], []
    from tools.experiment_stages import StageStore
    for budget in protocol['budgets']:
        store = StageStore(directory/'stages'/f'{cfg.seed}-{int(cfg.dr0)}-{crop_name}-{budget}',
                           {'protocol': protocol, 'input_sha256': identity,
                            'crop': crop_name, 'budget': budget, 'initialization': 'default'},
                           deadline=deadline)
        t0 = time.monotonic()
        result = evaluate_crop(cfg, crop, extras, maxiter=budget,
                               require_convergence=False, return_reconstructions=True, stage_runner=store.run)
        images.append(result.pop('_reconstructions'))
        runs.append({'maxiter': budget, 'wall_s': time.monotonic()-t0, 'result': result})
    comparison = compare(runs, images, protocol['image_tolerance'], protocol['gap_tolerance'])
    unchanged = identity == file_hash(path)
    row = {'seed': cfg.seed, 'dr0': cfg.dr0, 'crop': crop_name,
           'input_path': str(path), 'input_sha256': identity, 'input_unchanged': unchanged,
           'config': asdict(cfg), 'runs': runs, **comparison,
           'wall_s': time.monotonic()-started, 'process_peak_rss_bytes': peak_rss_bytes()}
    row['budget_convergence_passed'] &= unchanged
    dest = directory/f'case-{cfg.seed}-{int(cfg.dr0)}-{crop_name}.json'
    write_json(dest, row)
    summary = {k: row[k] for k in ('seed', 'dr0', 'crop', 'budget_convergence_passed', 'wall_s')}
    summary.update(path=dest.name, sha256=file_hash(dest),
                   max_image_relative_change=max(comparison['image_relative_changes'].values()),
                   max_gap_absolute_change=max(comparison['gap_absolute_changes'].values()))
    return summary


def run(inputs, directory, family='development', workers=3, budgets=(512, 1024), solver='fista', *, resume=False, wall_budget_s=3600., background_workload='unspecified', input_compatibility='strict'):
    from planetrecon import constants as C
    from planetrecon.evaluate import REG, aggregate_family, family_paths
    from planetrecon.provenance import source_hash
    from planetrecon.rank import ranking_config_hash
    seeds = C.DEV_SEEDS if family == 'development' else C.EVAL_SEEDS
    regimes = C.MANDATORY_DR0
    if len(budgets) != 2 or budgets[0] < 1 or budgets[1] <= budgets[0]:
        raise ValueError('two positive increasing solver budgets are required')
    if not 1 <= workers <= 16:
        raise ValueError('workers must be 1–16 (two CPU threads each)')
    if wall_budget_s <= 0 or not __import__('math').isfinite(wall_budget_s):
        raise ValueError('wall budget must be finite and positive')
    directory.mkdir(parents=True, exist_ok=resume)
    identity = source_hash()
    runner_identity = file_hash(__file__)
    solver_path = Path(__file__).with_name('quadratic_admm.py') if solver == 'admm' else None
    solver_identity = None if solver_path is None else file_hash(solver_path)
    protocol = {'input_compatibility': input_compatibility,
                'compatibility_source_sha256': file_hash(Path(__file__).with_name('input_compatibility.py')),
                'compatibility_bridge_sha256': file_hash(Path(__file__).resolve().parents[1]/'results/r10-full-grid-inputs/compatibility.json'),
                'family': family, 'seeds': list(seeds), 'regimes': list(regimes),
                'crops': ['feature', 'bland'], 'n_frames': C.N_FRAMES, 'eval_size': C.EVAL_SIZE,
                'budgets': list(budgets), 'solver_tolerance': C.E2A_FISTA_TOL,
                'image_tolerance': 1e-3, 'gap_tolerance': .01,
                'regularisation': asdict(REG), 'ranking_hash': ranking_config_hash(),
                'source_hash': identity, 'runner_sha256': runner_identity,
                'solver': solver, 'solver_source_sha256': solver_identity,
                'admm_relative_solution_error_bound_tolerance': 1e-4 if solver == 'admm' else None,
                'workers': workers, 'threads_per_worker': 2,
                'wall_budget_s': wall_budget_s, 'background_workload': background_workload,
                'checkpoint_runner_sha256': file_hash(Path(__file__).with_name('experiment_stages.py')),
                'input_sha256': {str(p.resolve()): file_hash(p) for p in family_paths(inputs, seeds)}, 'q3_authorized': False,
                'design': 'Known-transfer full-resolution Gate-1 reconstruction and gap budget stability. '
                          'Fixed regularisation, rankings and physical inputs; no MFBD/model/noise qualification.'}
    protocol_path = directory/'protocol.json'
    if resume:
        if not protocol_path.exists() or json.loads(protocol_path.read_text()) != protocol:
            raise ValueError('resume protocol/input/source identity mismatch')
    else:
        write_json(protocol_path, protocol)
    started = time.monotonic()
    rows, failures = [], []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(evaluate_case, (path, crop, directory, protocol, started+wall_budget_s)): (path, crop)
                   for path in family_paths(inputs, seeds) for crop in protocol['crops']}
        for future in as_completed(pending):
            path, crop = pending[future]
            try:
                row = future.result()
                rows.append(row)
                print(f'{path.name} {crop}: passed={row["budget_convergence_passed"]}; {row["wall_s"]:.1f}s', flush=True)
            except Exception as exc:
                failures.append({'input': str(path), 'crop': crop, 'error': repr(exc)})
                print(f'{path.name} {crop}: {exc}', flush=True)
            write_json(directory/'progress.json', {'cases': rows, 'failures': failures})
    rows.sort(key=lambda r: (r['seed'], r['dr0'], r['crop']))
    complete = complete_family(rows, seeds, regimes) and not failures
    unchanged = (identity == source_hash() and runner_identity == file_hash(__file__)
                 and (solver_path is None or solver_identity == file_hash(solver_path))
                 and protocol['checkpoint_runner_sha256'] == file_hash(Path(__file__).with_name('experiment_stages.py'))
                 and protocol['compatibility_source_sha256'] == file_hash(Path(__file__).with_name('input_compatibility.py'))
                 and protocol['compatibility_bridge_sha256'] == file_hash(Path(__file__).resolve().parents[1]/'results/r10-full-grid-inputs/compatibility.json'))
    passed = complete and unchanged and all(r['budget_convergence_passed'] for r in rows)
    tables = {}
    if complete:
        for budget_index, budget in enumerate(budgets):
            for crop in protocol['crops']:
                results = []
                for row in rows:
                    if row['crop'] != crop:
                        continue
                    case = json.loads((directory/row['path']).read_text())
                    results.append({'seed': row['seed'], 'dr0': row['dr0'],
                                    'crops': {crop: case['runs'][budget_index]['result']}})
                tables[f'{budget}/{crop}'] = aggregate_family(results, crop=crop)
    report = {'status': 'valid' if passed else 'incomplete', 'q3_authorized': False,
              'complete': complete, 'source_hash': identity, 'source_unchanged': unchanged,
              'full_resolution_budget_convergence_passed': passed, 'cases': rows,
              'failures': failures, 'tables': tables, 'wall_s': time.monotonic()-started}
    write_json(directory/f'attempt-{time.time_ns()}.json', report)
    write_json(directory/'report.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--family', choices=('development', 'evaluation'), default='development')
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--budgets', type=int, nargs=2, default=(512, 1024))
    parser.add_argument('--solver', choices=('fista', 'admm'), default='fista')
    parser.add_argument('--input-compatibility', choices=('strict', 'archived-full-grid'), default='strict')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--wall-budget-s', type=float, default=3600.)
    parser.add_argument('--background-workload', default='unspecified')
    args = parser.parse_args()
    report = run(args.inputs, args.out, args.family, args.workers, tuple(args.budgets), args.solver,
                 resume=args.resume, wall_budget_s=args.wall_budget_s, background_workload=args.background_workload, input_compatibility=args.input_compatibility)
    sys.exit(0 if report['full_resolution_budget_convergence_passed'] else 1)
