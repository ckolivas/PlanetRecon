"""Audit and compare the frozen AP regularization replay and fresh controls."""
from collections import Counter
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import center_of_mass

from planetrecon.result import load_snapshot
from tools.alignment_noise_experiment import read_png, measurements
from tools.analyse_joint_saturn import sharp_identity
from tools.analyse_refined_registration import ring_edges, edge_sensitivity
from tools.ap_stability_screen import hashes
from tools.coherent_saturn_experiment import inputs
from tools.joint_saturn_experiment import digest, CAPTURE
from tools.validate_local_warp_centring import image_errors


def fit_summary(rows):
    result = dict(count=len(rows), reasons=dict(Counter(r.get('reason', 'none') for r in rows)),
                  guard_rejections=sum(not r['field_guard_accepted'] for r in rows),
                  insufficient_points=sum(bool(r['fallback']) for r in rows))
    for key in ('minimum_jacobian', 'maximum_residual_px'):
        result[key+'_min_p25_median_p75_max'] = np.percentile(
            [r[key] for r in rows], [0, 25, 50, 75, 100]).tolist()
    return result


def validation(recipe):
    root = Path('out/ap-stability-validation')
    report = json.loads((root/'report.json').read_text())
    if (report['hashes'] != hashes() or report['source_sha256'] != digest('tools/ap_stability_validation.py')
            or report['screen_report_sha256'] != digest('out/ap-stability-screen/report.json')
            or report['seed'] != 9033 or report['frames'] != 32):
        raise ValueError('Validation identity changed')
    report['sharpening'] = sharp_identity(root, recipe)
    for case, values in report['cases'].items():
        values['raw_sha256'] = digest(root/f'{case}.npz')
        if values['quadrature_max_error_adu'] > 1e-5:
            raise ValueError('Inaccurate control integration')
        with np.load(root/f'{case}.npz') as data:
            mask, target = data['mask'], data['noiseless_oracle']
            for name, variant in values['variants'].items():
                if (image_errors(data[name], target, mask) != variant['image'] or
                        image_errors(data['clean_'+name], target, mask) != variant['clean_image']):
                    raise ValueError('Validation artifact changed')
                variant['fit_summary'] = fit_summary(variant.pop('statistics'))
                if variant['fit_summary']['count'] != 32:
                    raise ValueError('Incomplete validation fits')
        with np.load(root/'sharpened'/f'{case}_oracle_stages.npz') as data:
            target = data['sharpened'].mean(2)
        values['stage_sha256'] = {'oracle': digest(root/'sharpened'/f'{case}_oracle_stages.npz')}
        for name, variant in values['variants'].items():
            path = root/'sharpened'/f'{case}_{name}_stages.npz'
            values['stage_sha256'][name] = digest(path)
            with np.load(path) as data:
                variant['sharpened_error_linear_0_1'] = image_errors(data['sharpened'].mean(2), target, mask)
        print(case, {k: dict(field=v['field_rmse_px'], clean=v['clean_image']['rmse_adu'],
            sharp=v['sharpened_error_linear_0_1']['rmse_adu']) for k, v in values['variants'].items()}, flush=True)
    return report


def gallery(root, paths):
    import os
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QGuiApplication, QImage, QPainter, QColor, QFont
    app = QGuiApplication.instance() or QGuiApplication([])
    canvas = QImage(2000, 1130, QImage.Format.Format_RGB32)
    canvas.fill(QColor('black'))
    painter = QPainter(canvas)
    painter.setPen(QColor('white'))
    painter.setFont(QFont('Sans', 18))
    for i, (name, label) in enumerate([('baseline', 'Coherent AP baseline'),
            ('anchor_01', 'Stronger constraints: anchor_01'), ('production', 'PR production stack'),
            ('AS_manual64', 'AutoStakkert: contextual comparison')]):
        a = read_png(paths[name])[0]
        cy, cx = np.rint(center_of_mass(np.maximum(a-.02*a.max(), 0))).astype(int)
        crop = QImage(str(paths[name])).copy(int(cx-250), int(cy-115), 500, 230)
        crop = crop.scaled(1000, 460, Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.FastTransformation)
        x, y = (i % 2)*1000, (i//2)*510
        painter.drawText(x+12, y+32, label)
        painter.drawImage(x, y+45, crop)
    painter.drawText(12, 1050, '5,738 frames; Wavelet 27/0/0/0 + Adaptive Deconvolution 15.6, Contrast Adaptive')
    painter.drawText(12, 1095, 'No added filtering or normalization; AutoStakkert has its own selection and reference')
    painter.end()
    if not canvas.save(str(root/'comparison.png')):
        raise OSError('Comparison export failed')


def main():
    root = Path('out/saturn-regularized-ap')
    report = json.loads((root/'report.json').read_text())
    manifest = json.loads((root/'manifest.json').read_text())
    _, indices, quality, _, _, input_hashes = inputs()
    versions = dict(input_hashes, **hashes())
    versions['tools/regularized_ap_replay.py'] = digest('tools/regularized_ap_replay.py')
    versions['screen_report'] = digest('out/ap-stability-screen/report.json')
    if manifest['hashes'] != versions or manifest['capture_sha256'] != digest(CAPTURE):
        raise ValueError('Replay identity changed')
    versions['capture'] = manifest['capture_sha256']
    stats = report.pop('statistics')
    if (report['hashes'] != versions or report['n_used'] != 5738 or
            manifest['frame_indices'] != indices.tolist() or
            [r['position'] for r in stats] != list(range(5738)) or
            [r['frame_index'] for r in stats] != indices.tolist()):
        raise ValueError('Incomplete or misassociated replay')
    if report['halves_sha256'] != digest(root/'halves.npz'):
        raise ValueError('Half accumulators changed')
    report['raw_report_sha256'] = digest(root/'report.json')
    report['analyzer_sha256'] = digest(__file__)
    names = report['config']['names']
    if names != ['baseline', 'anchor_01']:
        raise ValueError('Unexpected selected policy')
    report['fit_summary'] = {name: fit_summary([r['fits'][name] for r in stats]) for name in names}
    baseline = load_snapshot(root/'baseline.npz')
    prior = load_snapshot('out/saturn-coherent-field/coherent.npz')
    np.testing.assert_allclose(baseline.image, prior.image, rtol=0, atol=1e-10)
    recipe = json.loads(Path('out/saturn-coherent-field/sharpened/sharpening_report.json').read_text())
    report['sharpening'] = sharp_identity(root, recipe)
    np.testing.assert_array_equal(read_png(root/'sharpened/baseline_sharpened.png')[0],
        read_png('out/saturn-coherent-field/sharpened/coherent_sharpened.png')[0])
    report['sharpened_baseline_pixel_parity'] = True
    paths = {name: root/'sharpened'/f'{name}_sharpened.png' for name in names}
    paths.update(production=Path('2024-09-27-1154_3-CK-R-SatPs.png'),
        AS_manual64=Path('out/saturn-matched-transfer/sharpened/as_manual64_sharpened.png'))
    arrays = {name: read_png(path)[0] for name, path in paths.items()}
    report['comparison'] = {}
    for name, a in arrays.items():
        report['comparison'][name] = dict(path=str(paths[name]), sha256=digest(paths[name]),
            variation=measurements(a), ring_edges=ring_edges(a))
        print(name, 'variation', [v['highpass_percent'] for v in measurements(a).values()],
            'widths', [v['width_10_90_px'] for v in ring_edges(a).values()], flush=True)
    report['raw_halves'] = {}
    with np.load(root/'halves.npz') as data:
        for i, name in enumerate(names):
            sums, weights = data['signal'][i], data['support'][i]
            halves = np.divide(sums, weights, out=np.zeros_like(sums), where=weights>0)
            coverage = weights.sum(0)
            mean = np.divide(sums.sum(0), coverage, out=np.zeros_like(coverage), where=coverage>0)
            np.testing.assert_array_equal(mean, load_snapshot(root/f'{name}.npz').image)
            report['raw_halves'][name] = dict(full_mean=measurements(mean),
                half_difference_over_two=measurements(mean, difference=(halves[0]-halves[1])/2))
    quality = np.maximum(quality, 1e-12)
    report['half_quality_weights'] = dict(fraction=[float(q.sum()/quality.sum()) for q in (quality[::2], quality[1::2])],
        full_effective_frame_count=float(quality.sum()**2/np.sum(quality*quality)))
    a = arrays['baseline']
    centre = np.rint(center_of_mass(np.maximum(a-.02*a.max(), 0))).astype(int)
    images = dict(production_local=arrays['anchor_01'], existing_bilinear=a, AS_manual64=a)
    report['paired_edges_vs_baseline'] = {side: dict(paired_windows=v['paired_windows'],
        candidate_minus_baseline_width_px_min_median_max=v['local_minus_global_width_px_min_median_max'],
        fraction_candidate_wider=v['fraction_local_wider'])
        for side, v in edge_sensitivity(images, {k: centre for k in images}).items()}
    report['validation'] = validation(recipe)
    report['limits'] = [
        'The two replay variants share every frame, weight, reference, AP observation, global shift and final sampler.',
        'More stable fields can suppress genuine short-scale motion; synthetic validation measures that tradeoff.',
        'Fine variation includes detail and artifacts; ring widths are descriptive, not calibrated resolution.',
        'Half-difference/2 cancels shared reference and estimator biases and is not a measure of total noise.',
        'Separately sharpened halves are review artifacts, not linear estimates of full-stack noise.',
        'AS uses its own selection and reference and serves only as context.',
        'No output filtering, brightness normalization or production changes were introduced.']
    (root/'analysis.json').write_text(json.dumps(report, indent=2)+'\n')
    Path('results/registration/ap-stability-replay.json').write_text(json.dumps(report, indent=2)+'\n')
    gallery(root, paths)


if __name__ == '__main__':
    main()
