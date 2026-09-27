"""Audit the AP oracle intervention, including the exact sharpening workflow."""
import json
from pathlib import Path

import numpy as np

from tools.ap_oracle_observations import source_hashes, NAMES, CASES
from tools.analyse_joint_saturn import sharp_identity
from tools.joint_saturn_experiment import digest
from tools.validate_local_warp_centring import image_errors, weighted_field_errors


def main():
    root = Path('out/ap-oracle-observations')
    report = json.loads((root/'report.json').read_text())
    if report['hashes'] != source_hashes() or report['variants'] != NAMES or report['frames'] != 32:
        raise ValueError('Experiment implementation or configuration changed')
    if set(report['cases']) != set(CASES) or report['seed'] != 9033:
        raise ValueError('Incomplete controls')
    if report['geometry_sha256'] != digest(root/'geometry.npz'):
        raise ValueError('AP geometry changed')
    recipe = json.loads(Path('out/saturn-coherent-field/sharpened/sharpening_report.json').read_text())
    report['sharpening'] = sharp_identity(root, recipe)
    expected = {f'{case}_{name}' for case in CASES for name in NAMES+['target']}
    if set(report['sharpening']['images']) != expected:
        raise ValueError('Incomplete exact sharpening')
    report['raw_report_sha256'] = digest(root/'report.json')
    report['analyzer_sha256'] = digest(__file__)
    for case, values in report['cases'].items():
        if values['raw_sha256'] != digest(root/f'{case}.npz') or values['quadrature_endpoint_error_adu'] > 1e-5:
            raise ValueError('Raw results changed or inadequate integration')
        with np.load(root/f'{case}.npz') as data:
            mask, inner, truth = data['mask'], data['inner'], data['truth_fields']
            if truth.shape != (32, 2, *report['shape']):
                raise ValueError('Incomplete frame sequence')
            for name, variant in values['variants'].items():
                checks = dict(image=image_errors(data[name], data['target'], mask),
                    clean_image=image_errors(data['clean_'+name], data['target'], mask),
                    versus_noisy_oracle=image_errors(data[name], data['oracle'], mask),
                    field=weighted_field_errors(data['field_'+name], truth, np.ones(32)/32, mask),
                    inner_field=weighted_field_errors(data['field_'+name], truth, np.ones(32)/32, inner))
                if any(variant[k] != v for k, v in checks.items()):
                    raise ValueError('Numerical record does not reproduce from artifacts')
                stats = variant.pop('statistics')
                if len(stats) != 32:
                    raise ValueError('Incomplete diagnostics')
                variant['fit_summary'] = dict(guard_rejections=sum(not s.get('field_guard_accepted', True) for s in stats),
                    insufficient_points=sum(bool(s.get('fallback', False)) for s in stats),
                    bypassed_taper_and_guard=sum(s.get('taper_and_guard_bypassed', False) for s in stats))
        stage_arrays = {}
        values['stage_sha256'] = {}
        for name in NAMES+['target']:
            path = root/'sharpened'/f'{case}_{name}_stages.npz'
            values['stage_sha256'][name] = digest(path)
            with np.load(path) as stages:
                stage_arrays[name] = stages['sharpened'].mean(2)
        for name, variant in values['variants'].items():
            variant['sharpened_error_linear_0_1'] = image_errors(stage_arrays[name], stage_arrays['target'], mask)
            variant['sharpened_versus_noisy_oracle'] = image_errors(stage_arrays[name], stage_arrays['oracle'], mask)
        print(case, {k: dict(field=v['field']['total_vector_rmse_px'], inner=v['inner_field']['total_vector_rmse_px'],
            clean=v['clean_image']['rmse_adu'], sharp=v['sharpened_error_linear_0_1']['rmse_adu'],
            sharp_vs_noisy_truth=v['sharpened_versus_noisy_oracle']['rmse_adu']) for k, v in values['variants'].items()}, flush=True)
    report['limits'] = [
        'Truth patch averages are exact for the fitter observation model, not necessarily the optimum translation of a deforming patch.',
        'Measured AP errors include matching and multiscale composition; this intervention does not separate those two stages.',
        'Truth-gated keeps the original frame-dependent confidence and selection; truth-all uses every configured AP with unit weight.',
        'The untapered diagnostic bypasses both taper and geometric guard and is not a deployable algorithm.',
        'Known fields are unavailable for Saturn; these controlled outcomes cannot be assigned as percentages of its noise gap.',
        'The noiseless target is the same detector-integrated frames sampled with truth, not an ideal infinite-resolution scene.',
        'Noisy-oracle comparisons retain identical noise realizations; they measure changes caused by alignment, not all image noise.',
        'Four fixed smooth-shear cases and one continuous scene do not cover atmospheric motion or PSF variability.',
        'No production changes, output filtering or brightness normalization.']
    (root/'analysis.json').write_text(json.dumps(report, indent=2)+'\n')
    Path('results/registration/ap-oracle-observations.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
