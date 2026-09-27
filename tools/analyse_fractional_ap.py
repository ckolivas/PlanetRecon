"""Audit fractional matching controls and the exact sharpening validation."""
import json
from pathlib import Path

import numpy as np

from tools.ap_fractional_phase import hashes, POLICIES
from tools.ap_oracle_observations import CASES
from tools.analyse_joint_saturn import sharp_identity
from tools.joint_saturn_experiment import digest
from tools.validate_local_warp_centring import image_errors, weighted_field_errors


def main():
    root = Path('out/ap-fractional-validation')
    phase_path = Path('out/ap-fractional-phase/report.json')
    phase = json.loads(phase_path.read_text())
    report = json.loads((root/'report.json').read_text())
    if (phase['hashes'] != hashes() or report['hashes'] != hashes()
            or report['source_sha256'] != digest('tools/validate_fractional_ap.py')
            or report['phase_report_sha256'] != digest(phase_path)
            or report['prior_report_sha256'] != digest('out/ap-noise-coupling/report.json')):
        raise ValueError('Experiment source or input changed')
    expected_shifts = [[x,y] for y in (-.5,-.25,0.,.25,.5) for x in (-.5,-.25,0.,.25,.5)]+[[1.,0.],[0.,1.],[-1.,-1.]]
    if ([r['shift'] for r in phase['rows']] != expected_shifts or
            phase['policies'] != {k:list(v) for k,v in POLICIES.items()} or report['count'] != 32
            or set(report['cases']) != set(CASES) or report['selected'] != phase['selected']):
        raise ValueError('Incomplete or changed controls')
    for key in POLICIES:
        actual = float(np.sqrt(np.mean([r['variants'][key]['field_vector_rmse_px']**2 for r in phase['rows'][:25]])))
        if actual != phase['summary'][key]['field_rmse_px']:
            raise ValueError('Phase summary differs')
    report['phase_screen'] = phase
    boundary_path = Path('out/ap-fractional-phase/boundaries.json')
    boundary = json.loads(boundary_path.read_text())
    if (boundary['hashes'] != hashes() or boundary['source_sha256'] != digest('tools/ap_phase_boundaries.py')
            or len(boundary['rows']) != 12):
        raise ValueError('Boundary control changed or incomplete')
    phase_rows = {tuple(r['shift']): r for r in phase['rows']}
    for row in boundary['rows']:
        for key, value in row['variants'].items():
            if not row['padded']:
                np.testing.assert_allclose(value['field_vector_rmse_px'],
                    phase_rows[tuple(row['shift'])]['variants'][key]['field_vector_rmse_px'], atol=1e-10, rtol=0)
            elif all(float(s).is_integer() for s in row['shift']):
                if value['field_vector_rmse_px'] > 1e-12:
                    raise ValueError('Exact padded integer translation not recovered')
    report['boundary_control'] = boundary
    report['boundary_report_sha256'] = digest(boundary_path)
    report['analyzer_sha256'] = digest(__file__)
    report['raw_report_sha256'] = digest(root/'report.json')
    recipe = json.loads(Path('out/saturn-coherent-field/sharpened/sharpening_report.json').read_text())
    report['sharpening'] = sharp_identity(root, recipe)
    selected = report['selected']
    keys = [f'{name}_noise{noise}' for name in ('baseline',selected) for noise in (0,2)]
    if set(report['sharpening']['images']) != {f'{case}_{key}' for case in CASES for key in keys+['target']}:
        raise ValueError('Incomplete sharpening')
    for case, values in report['cases'].items():
        if values['raw_sha256'] != digest(root/f'{case}.npz'):
            raise ValueError('Raw artifacts changed')
        with np.load(root/f'{case}.npz') as data:
            mask, truth, target = data['mask'], data['truth'], data['target']
            for key, variant in values['variants'].items():
                expected = dict(image=image_errors(data[key],target,mask),
                    clean=image_errors(data['clean_'+key],target,mask),
                    field=weighted_field_errors(data['field_'+key],truth,np.ones(32)/32,mask))
                if any(variant[k] != v for k,v in expected.items()):
                    raise ValueError('Saved metrics do not match arrays')
                stats = variant.pop('statistics')
                if len(stats) != 32:
                    raise ValueError('Incomplete fit statistics')
                if key.startswith(selected):
                    variant['guard_rejections'] = sum(not s['fit']['field_guard_accepted'] for s in stats)
                    variant['insufficient_points'] = sum(s['fit']['fallback'] for s in stats)
        values['stage_sha256'] = {}
        with np.load(root/'sharpened'/f'{case}_target_stages.npz') as data:
            sharp_target = data['sharpened'].mean(2)
        for key in keys+['target']:
            path = root/'sharpened'/f'{case}_{key}_stages.npz'
            values['stage_sha256'][key] = digest(path)
            if key != 'target':
                with np.load(path) as data:
                    values['variants'][key]['sharpened_error_linear_0_1'] = image_errors(data['sharpened'].mean(2),sharp_target,mask)
        print(case,{k:dict(field=v['field']['total_vector_rmse_px'],clean=v['clean']['rmse_adu'],
              sharp=v['sharpened_error_linear_0_1']['rmse_adu']) for k,v in values['variants'].items()},flush=True)
    report['limits'] = [
        'Matching-only interpolation changes; final detector-pixel accumulation remains bilinear in every branch.',
        'The selected policy was screened on known translations and validated on the existing fixed motion/noise controls.',
        'Noise-free integer translations can still differ through reflected-boundary proxy filtering; the self-match is a separate control.',
        'The refinement preserves original integer-score confidence/ambiguity gates and accepts only fractional NCC increases.',
        'Subpixel matching tests do not establish a reduction in Saturn grain or reveal AutoStakkert internals.',
        'Realistic noise and deforming images can expose different tradeoffs from noiseless translations.',
        'This unoptimized diagnostic performs many score evaluations; no speed or production-readiness claim.',
        'No production defaults, output filter or brightness normalization changes.']
    (root/'analysis.json').write_text(json.dumps(report,indent=2)+'\n')
    Path('results/registration/ap-fractional-phase.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
