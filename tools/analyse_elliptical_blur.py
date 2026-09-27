"""Verify and score the independent PSF controls and matched Saturn pilot."""
import json
from pathlib import Path

import numpy as np

from tools.alignment_noise_experiment import read_png, measurements
from tools.analyse_joint_saturn import sharp_identity
from tools.analyse_refined_registration import ring_edges
from tools.joint_saturn_experiment import digest
from tools.validate_local_warp_centring import image_errors


def independent_global_check(real, pilot):
    from scipy.ndimage import map_coordinates
    from planetrecon.io.ser import SERSource
    from planetrecon.result import load_snapshot
    from tools.coherent_saturn_experiment import inputs
    from tools.joint_saturn_experiment import CAPTURE
    _, indices, quality, shifts, reference, hashes = inputs()
    for key, expected in hashes.items():
        if real['hashes'][key] != expected:
            raise ValueError('Pilot input changed: '+key)
    if digest(CAPTURE) != real['hashes']['capture']:
        raise ValueError('Pilot capture changed')
    positions = np.linspace(0, len(indices)-1, 128, dtype=int)
    if positions.tolist() != real['positions'] or indices[positions].tolist() != real['frame_indices']:
        raise ValueError('Pilot selection differs')
    if [row['position'] for row in real['statistics']] != positions.tolist():
        raise ValueError('Misassociated frame diagnostics')
    yy, xx = np.indices(reference.shape, dtype=float)
    total, support = np.zeros_like(reference), np.zeros_like(reference)
    with SERSource(CAPTURE) as source:
        for position in positions:
            coords = np.array([yy+shifts[position, 1], xx+shifts[position, 0]])
            weight = max(float(quality[position]), 1e-12)
            total += weight*map_coordinates(source.read_raw(int(indices[position])).astype(float),
                coords, order=1, mode='grid-constant', cval=0., prefilter=False)
            support += weight*map_coordinates(np.ones_like(reference), coords, order=1,
                mode='grid-constant', cval=0., prefilter=False)
    independent = np.divide(total, support, out=np.zeros_like(total), where=support>0)
    actual = load_snapshot(pilot/'global_only.npz').image
    np.testing.assert_allclose(actual, independent, atol=1e-10, rtol=0)
    return dict(maximum_difference_adu=float(np.max(abs(actual-independent))),
                method='Independent SciPy float64 order-1 grid-constant accumulation')


def verify_sources(report):
    for path, expected in report['hashes'].items():
        if path.startswith('tools/') and digest(path) != expected:
            raise ValueError('Experiment implementation changed: '+path)


def diagnostics(rows, name):
    fits = [r[name] for r in rows]
    return dict(heldout_acceptances=sum(r['heldout_improves'] for r in fits),
        geometric_fallbacks=sum(r['fallback'] for r in fits),
        optimizer_successes=sum(r['success'] for r in fits),
        median_heldout_ratio=float(np.median([r['heldout_loss']/max(r['global_heldout_loss'], 1e-30) for r in fits])))


def gallery(root, ablation):
    import os
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QGuiApplication, QImage, QPainter, QColor, QFont
    app = QGuiApplication.instance() or QGuiApplication([])
    paths = [root/'sharpened'/f'{name}_sharpened.png' for name in ('coherent', 'isotropic')]
    paths += [ablation/'sharpened/saturn_sharpened.png', root/'sharpened/ellipse_estimated_sharpened.png']
    labels = ['Coherent AP baseline', 'Circular-blur joint fit', 'Fixed circular reference, then motion', 'Elliptical reference, then motion fit']
    canvas = QImage(1392, 960, QImage.Format.Format_RGB32)
    canvas.fill(QColor('black'))
    painter = QPainter(canvas)
    painter.setPen(QColor('white'))
    painter.setFont(QFont('Sans', 16))
    for i, (path, label) in enumerate(zip(paths, labels)):
        image = QImage(str(path))
        if image.isNull():
            raise ValueError('Missing sharpened comparison')
        x, y = i % 2*696, i//2*450
        painter.drawText(x+12, y+30, label)
        painter.drawImage(x, y+38, image.scaled(696, 404, Qt.AspectRatioMode.IgnoreAspectRatio,
                                                Qt.TransformationMode.FastTransformation))
    painter.drawText(12, 935, 'Same 128 frames; exact user sharpening; no added output filter or normalization')
    painter.end()
    if not canvas.save(str(root/'comparison.png')):
        raise OSError('Gallery export')


def main():
    root = Path('out/elliptical-blur-controls')
    pilot = Path('out/saturn-elliptical-pilot')
    ablation = Path('out/fixed-circular-blur-controls')
    reference_recipe = json.loads(Path('out/saturn-coherent-field/sharpened/sharpening_report.json').read_text())
    controls = json.loads((root/'report.json').read_text())
    verify_sources(controls)
    controls['sharpening'] = sharp_identity(root, reference_recipe)
    fixed = json.loads((ablation/'report.json').read_text())
    verify_sources(fixed)
    if fixed['control_report_sha256'] != digest(root/'report.json') or fixed['pilot_report_sha256'] != digest(pilot/'report.json'):
        raise ValueError('Fixed-circular control inputs differ')
    fixed['sharpening'] = sharp_identity(ablation, reference_recipe)
    for scene, cases in controls['scenes'].items():
        for case, result in cases.items():
            if len(result['statistics']) != controls['frames']:
                raise ValueError('Incomplete control')
            covariance_errors = [np.asarray(s['ellipse']['covariance'])-s['true_covariance']
                                 for s in result['statistics']]
            result['ellipse_fit'] = dict(
                covariance_frobenius_rmse_px2=float(np.sqrt(np.mean(np.sum(np.array(covariance_errors)**2, axis=(1, 2))))),
                optimizer_successes=sum(s['ellipse']['success'] for s in result['statistics']))
            prefix = f'{scene}_{case}'
            with np.load(root/f'{prefix}.npz') as arrays:
                mask = arrays['mask']
                for name, values in result['variants'].items():
                    actual = image_errors(arrays['clean_'+name], arrays['noiseless_oracle'], mask)
                    np.testing.assert_allclose(actual['rmse_adu'], values['clean_image']['rmse_adu'], rtol=0, atol=0)
            with np.load(root/'sharpened'/f'{prefix}_noiseless_oracle_stages.npz') as data:
                target = data['sharpened'].mean(2)
            fixed_case = fixed['controls'][prefix]
            if len(fixed_case['statistics']) != controls['frames']:
                raise ValueError('Incomplete fixed circular control')
            with np.load(ablation/'sharpened'/f'{prefix}_stages.npz') as data:
                fixed_case['sharpened_error'] = image_errors(data['sharpened'].mean(2), target, mask)
            fixed_case['diagnostics'] = diagnostics(fixed_case['statistics'], 'fit')
            for name, values in result['variants'].items():
                with np.load(root/'sharpened'/f'{prefix}_{name}_stages.npz') as data:
                    pixels = data['sharpened'].mean(2)
                values['sharpened_error'] = image_errors(pixels, target, mask)
                if name in ('isotropic', 'ellipse_estimated', 'ellipse_known'):
                    values['diagnostics'] = diagnostics(result['statistics'], name)
            print(scene, case, {k: [round(v['field_rmse_px'], 6),
                round(v['clean_image']['rmse_adu'], 6), round(v['sharpened_error']['rmse_adu'], 6)]
                for k, v in result['variants'].items()}, flush=True)
            print('fixed circular', fixed_case['field_rmse_px'], fixed_case['clean_image']['rmse_adu'],
                  fixed_case['sharpened_error']['rmse_adu'], flush=True)
    real = json.loads((pilot/'report.json').read_text())
    verify_sources(real)
    if real['frames'] != 128 or len(real['statistics']) != 128:
        raise ValueError('Incomplete Saturn pilot')
    real['sharpening'] = sharp_identity(pilot, reference_recipe)
    real['independent_global_parity'] = independent_global_check(real, pilot)
    real['comparison'] = {}
    for name in ('global_only', 'coherent', 'isotropic', 'ellipse_estimated', 'fixed_circular'):
        path = (ablation/'sharpened/saturn_sharpened.png' if name == 'fixed_circular' else
                pilot/'sharpened'/f'{name}_sharpened.png')
        pixels = read_png(path)[0]
        values = dict(variation=measurements(pixels), ring_edges=ring_edges(pixels))
        if name in ('isotropic', 'ellipse_estimated'):
            values['diagnostics'] = diagnostics(real['statistics'], name)
        real['comparison'][name] = values
        print('Saturn', name, 'variation', [v['highpass_percent'] for v in values['variation'].values()],
              'widths', [v['width_10_90_px'] for v in values['ring_edges'].values()], flush=True)
    if [r['position'] for r in fixed['real']] != real['positions']:
        raise ValueError('Incomplete fixed-circular pilot')
    report = dict(controls=controls, saturn_pilot=real, fixed_circular=fixed, limits=[
        'Known-PSF templates use simulation truth and are diagnostic only.',
        'Noise level 2 ADU is a synthetic control parameter, not a measured camera-noise estimate.',
        'The estimated elliptical PSF is fitted at the global pose and then held fixed during motion fitting.',
        'The covariance bounds permit broader blur than the original sigma-2 circular model.',
        'The Saturn pilot uses 128 frames; no comparison with a 5738-frame AS stack is valid.',
        'Fine-scale variation and ring widths are descriptive, not calibrated noise and resolution.',
        'Better held-out residuals do not establish true geometric motion in the real capture.',
        'Controlled Gaussian PSFs do not reproduce all atmospheric or optical aberrations.'])
    Path('results/registration/elliptical-blur-probe.json').write_text(json.dumps(report, indent=2)+'\n')
    gallery(pilot, ablation)


if __name__ == '__main__':
    main()
