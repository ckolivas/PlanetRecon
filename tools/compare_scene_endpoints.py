"""Compare same-objective endpoint certificates without promoting incomplete fits."""
import argparse
import json
import math
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.study_io import file_hash, write_json


def compare(reference, candidate, reference_protocol, candidate_protocol):
    keys = ('input_sha256', 'manifest_sha256', 'case', 'fractions', 'sum_native_strength',
            'margin', 'cell_factor', 'budgets', 'tolerance', 'image_tolerance')
    if any(reference_protocol[k] != candidate_protocol[k] for k in keys):
        raise ValueError('same observations, selections, domain, prior and numerical criteria required')
    if not reference['source_input_unchanged'] or not candidate['source_input_unchanged']:
        raise ValueError('study source/input identities changed')
    def budget_field(protocol):
        unit=protocol.get('budget_unit','outer iterations')
        if unit in ('iterations','outer iterations'): return 'maxiter'
        if isinstance(unit,str) and 'hessian products' in unit.lower(): return 'max_products'
        raise ValueError('unsupported solver budget unit')
    fields=[budget_field(p) for p in (reference_protocol,candidate_protocol)]
    rows = []
    for fraction in reference_protocol['fractions']:
        blocks = []
        for report in (reference, candidate):
            matches = [r for r in report['rows'] if r['fraction'] == fraction]
            if len(matches) > 1: raise ValueError('duplicate fraction')
            blocks.append(matches[0] if matches else None)
        a, b = blocks
        if a and b and any(a[k] != b[k] for k in ('indices', 'mean_ridge', 'observed_electron_sum')):
            raise ValueError('actual numerical inputs differ')
        def qualified(block, field):
            changes={} if block is None else block.get('relative_changes',{})
            stable=all(math.isfinite(changes.get(k,float('inf'))) and
                       0<=changes.get(k,float('inf'))<=reference_protocol['image_tolerance']
                       for k in ('latent','detector'))
            return bool(block and block['numerical_passed'] and stable
                        and [f[field] for f in block['runs']] == reference_protocol['budgets']
                        and all(f.get('converged') and f.get('reference_certificate', {}).get('feasible')
                                and f['reference_certificate']['relative_solution_error_bound'] <= reference_protocol['tolerance']
                                for f in block['runs']))
        row = {'fraction': fraction, 'reference_passed': qualified(a, fields[0]),
               'candidate_passed': qualified(b, fields[1]), 'fits': []}
        for cap in reference_protocol['budgets']:
            fits = []
            for block, field in zip(blocks, fields):
                matches = [] if block is None else [f for f in block['runs'] if f[field] == cap]
                if len(matches) > 1: raise ValueError('duplicate solver cap')
                fits.append(matches[0] if matches else None)
            fa, fb = fits
            both = all(f and f.get('converged') and f.get('reference_certificate', {}).get('feasible')
                       and f['reference_certificate']['relative_solution_error_bound'] <= reference_protocol['tolerance'] for f in fits)
            difference = abs(fa['objective']-fb['objective']) if both else None
            gap_sum = sum(f['reference_certificate']['objective_gap_upper_bound'] for f in fits) if both else None
            roundoff = 1e-10*max(1., abs(fa['objective']), abs(fb['objective'])) if both else None
            row['fits'].append({'nominal_budget': cap, 'both_independently_certified': bool(both),
                'objective_absolute_difference': difference, 'summed_objective_gap_bounds': gap_sum,
                'objective_roundoff_allowance': roundoff,
                'objective_consistent_with_bounds': None if not both else difference <= gap_sum+roundoff,
                'reference': fa, 'candidate': fb})
        rows.append(row)
    return {'rows': rows,
            'reference_budget_unit': reference_protocol.get('budget_unit', 'outer iterations'),
            'candidate_budget_unit': candidate_protocol.get('budget_unit', 'outer iterations'),
            'equal_nominal_caps_imply_equal_work': False,
            'certified_pair_count': sum(f['both_independently_certified'] for r in rows for f in r['fits']),
            'certified_objectives_consistent': all(f['objective_consistent_with_bounds'] is not False for r in rows for f in r['fits']),
            'candidate_endpoints_passed': all(r['candidate_passed'] for r in rows),
            'reference_endpoints_passed': all(r['reference_passed'] for r in rows),
            'scope': 'Same-objective numerical comparison; incomplete fits are not accuracy or speed baselines. Overlapping wall times are not isolated benchmarks.',
            'q3_authorized': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    paths = [root/name for root in (args.reference, args.candidate) for name in ('report.json', 'protocol.json')]
    a, ap, b, bp = [json.loads(p.read_text()) for p in paths]
    report = compare(a, b, ap, bp)
    report['source_reports'] = {str(p): file_hash(p) for p in paths}
    if args.out.exists(): raise ValueError('comparison output already exists')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, report)
