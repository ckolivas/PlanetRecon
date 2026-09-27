"""Validate the zero-noise matching and paired noise-coupling intervention."""
import json
from pathlib import Path

import numpy as np

from tools.ap_noise_coupling import hashes, CASES
from tools.analyse_joint_saturn import sharp_identity
from tools.joint_saturn_experiment import digest
from tools.validate_local_warp_centring import image_errors, weighted_field_errors


def main():
    root, prior = Path('out/ap-noise-coupling'), Path('out/ap-oracle-observations')
    report = json.loads((root/'report.json').read_text())
    if (report['hashes'] != hashes() or report['prior_report_sha256'] != digest(prior/'report.json')
            or report['geometry_sha256'] != digest(prior/'geometry.npz')
            or set(report['cases']) != set(CASES) or report['count'] != 32 or report['seeds'] != [9033, 19033]):
        raise ValueError('Experiment identity changed')
    recipe = json.loads(Path('out/saturn-coherent-field/sharpened/sharpening_report.json').read_text())
    report['sharpening'] = sharp_identity(root, recipe)
    names = ['no_noise_match', 'noisy_match', 'independent_match', 'oracle',
             'paired_same', 'paired_cross', 'paired_no_noise_match', 'paired_oracle', 'target']
    if set(report['sharpening']['images']) != {f'{case}_{name}' for case in CASES for name in names}:
        raise ValueError('Incomplete sharpening')
    report['raw_report_sha256'] = digest(root/'report.json')
    report['analyzer_sha256'] = digest(__file__)
    for case, values in report['cases'].items():
        if values['raw_sha256'] != digest(root/f'{case}.npz') or values['prior_raw_sha256'] != digest(prior/f'{case}.npz'):
            raise ValueError('Input or output arrays changed')
        with np.load(root/f'{case}.npz') as data:
            mask, inner, truth = data['mask'], data['inner'], data['truth']
            if truth.shape[0] != 32:
                raise ValueError('Incomplete frame sequence')
            for name, variant in values['matching'].items():
                expected = dict(field=weighted_field_errors(data['field_'+name], truth, np.ones(32)/32, mask),
                    inner_field=weighted_field_errors(data['field_'+name], truth, np.ones(32)/32, inner),
                    clean_stack=image_errors(data['clean_'+name], data['target'], mask))
                if any(variant[k] != v for k, v in expected.items()):
                    raise ValueError('Matching metrics differ from saved arrays')
            for name in names:
                if values['image_errors'][name] != image_errors(data[name], data['target'], mask):
                    raise ValueError('Image metrics differ from saved arrays')
            np.testing.assert_array_equal(data['pair_clean_same'], data['pair_clean_cross'])
            if values['paired_same_minus_cross'] != image_errors(data['paired_same'], data['paired_cross'], mask):
                raise ValueError('Noise coupling changed')
            values['paired_clean_max_difference_adu'] = float(np.max(abs(data['pair_clean_same']-data['pair_clean_cross'])))
        stages = {}
        values['stage_sha256'] = {}
        for name in names:
            path = root/'sharpened'/f'{case}_{name}_stages.npz'
            values['stage_sha256'][name] = digest(path)
            with np.load(path) as data:
                stages[name] = data['sharpened'].mean(2)
        values['sharpened_error_linear_0_1'] = {k: image_errors(v, stages['target'], mask) for k, v in stages.items()}
        values['sharpened_same_minus_cross'] = image_errors(stages['paired_same'], stages['paired_cross'], mask)
        values['sharpened_vs_same_source_oracle'] = {
            name: image_errors(stages[name], stages['paired_oracle' if name.startswith('paired_') else 'oracle'], mask)
            for name in names if name != 'target'}
        statistics = values.pop('statistics')
        values['fit_summary'] = {}
        for name, rows in statistics.items():
            if len(rows) != 32:
                raise ValueError('Incomplete fit diagnostics')
            values['fit_summary'][name] = dict(guard_rejections=sum(not r['field_guard_accepted'] for r in rows),
                insufficient_points=sum(r['fallback'] for r in rows))
        print(case, 'fields', {k: v['field']['total_vector_rmse_px'] for k, v in values['matching'].items()}, flush=True)
        print('  raw coupling', values['paired_same_minus_cross']['rmse_adu'], 'sharp',
              {k: values['sharpened_error_linear_0_1'][k]['rmse_adu'] for k in ('paired_same', 'paired_cross', 'paired_no_noise_match', 'paired_oracle')}, flush=True)
    report['limits'] = [
        'Zero-noise matching still includes detector integration, fractional translation, interpolation, proxy conversion and multiscale composition.',
        'AP truth is the fixed patch-average model; disagreement is not uniquely attributable to the correlation peak estimator.',
        'Noise level 2 ADU is a controlled intervention, not a measurement of Saturn sensor noise.',
        'Paired same/cross have 32 poses times two noise realizations; compare them to paired controls with the same samples.',
        'The raw pair difference cancels clean-signal geometry exactly, but separate nonlinear sharpening does not preserve that additive decomposition.',
        'A nonzero coupling establishes noise-dependent selection effects, not necessarily increased RMSE or the cause of real-capture grain.',
        'One fixed scene and two noise realizations per pose are exploratory; no population confidence interval is claimed.',
        'Production, output filtering and brightness normalization remain unchanged.']
    (root/'analysis.json').write_text(json.dumps(report, indent=2)+'\n')
    Path('results/registration/ap-noise-coupling.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
