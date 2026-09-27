"""Verify full-count joint-fit replay and compare exact sharpened outputs."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import center_of_mass

from planetrecon.result import load_snapshot
from tools.alignment_noise_experiment import read_png, measurements
from tools.analyse_refined_registration import ring_edges, edge_sensitivity
from tools.joint_saturn_experiment import digest, POLICIES


def sharp_identity(root, previous):
    report = json.loads((root/'sharpened/sharpening_report.json').read_text())
    for key in ('wavelet', 'deconvolution', 'implementation_sha256'):
        if report[key] != previous[key]:
            raise ValueError('Sharpening recipe differs')
    for item in report['images'].values():
        for key in ('source', 'output'):
            if digest(item[key]) != item[key+'_sha256']:
                raise ValueError('Sharpening artifact changed')
    return report


def comparison_image(root, paths, centre):
    import os
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QGuiApplication, QImage, QPainter, QColor, QFont
    app = QGuiApplication.instance() or QGuiApplication([])
    names = ['production', 'AS_manual64', 'coherent', 'joint_strong', 'global', 'joint_validated']
    labels = ['PR production stack', 'AutoStakkert: contextual comparison',
              'Coherent AP fit', 'Corrected joint fit', 'Global translations only', 'Joint fit with validation']
    canvas = QImage(2000, 1640, QImage.Format.Format_RGB32)
    canvas.fill(QColor('black'))
    painter = QPainter(canvas)
    painter.setPen(QColor('white'))
    painter.setFont(QFont('Sans', 18))
    for i, (name, label) in enumerate(zip(names, labels)):
        pixels = read_png(paths[name])[0]
        cy, cx = (np.rint(center_of_mass(np.maximum(pixels-.02*pixels.max(), 0))).astype(int)
                  if name == 'AS_manual64' else centre)
        image = QImage(str(paths[name])).copy(int(cx-250), int(cy-115), 500, 230)
        image = image.scaled(1000, 460, Qt.AspectRatioMode.IgnoreAspectRatio,
                             Qt.TransformationMode.FastTransformation)
        x, y = (i % 2)*1000, (i//2)*510
        painter.drawText(x+12, y+32, label)
        painter.drawImage(x, y+45, image)
    painter.drawText(12, 1560, '5,738 frames; Wavelet 27/0/0/0 + Adaptive Deconvolution 15.6, Contrast Adaptive')
    painter.drawText(12, 1605, 'No added output filter or brightness normalization; AS has its own selection and reference')
    painter.end()
    if not canvas.save(str(root/'comparison.png')):
        raise OSError('comparison image')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--record', type=Path, required=True)
    args = p.parse_args()
    root = args.out
    report = json.loads((root/'report.json').read_text())
    if report['n_used'] != 5738 or report['config']['policies'] != POLICIES:
        raise ValueError('Full fixed selection required')
    prior = json.loads(Path('out/saturn-coherent-field/report.json').read_text())
    for key in ('reference', 'indices', 'quality', 'global_shifts', 'model_code'):
        if report['hashes'][key] != prior['hashes'][key]:
            raise ValueError('Previous inputs differ')
    stats = json.loads((root/'fit_statistics.json').read_text())
    if digest(root/'fit_statistics.json') != report['fit_statistics_sha256']:
        raise ValueError('Changed fit statistics')
    if [s['position'] for s in stats] != list(range(5738)) or [s['frame_index'] for s in stats] != report['frame_indices']:
        raise ValueError('Incomplete or misassociated frame diagnostics')
    if digest(root/'half_stacks.npz') != report['half_stacks_sha256']:
        raise ValueError('Changed half stacks')
    recipe = json.loads(Path('out/saturn-coherent-field/sharpened/sharpening_report.json').read_text())
    report['sharpening'] = sharp_identity(root, recipe)
    report['baseline_parity'] = {}
    for method, path in [('global', 'out/saturn-controlled-alignment/global_subpixel.npz'),
                         ('coherent', 'out/saturn-coherent-field/coherent.npz')]:
        actual, previous = load_snapshot(root/f'{method}.npz'), load_snapshot(path)
        np.testing.assert_allclose(actual.image, previous.image, rtol=0, atol=1e-10)
        report['baseline_parity'][method] = dict(path=path, sha256=digest(path),
            maximum_difference_adu=float(np.max(abs(actual.image-previous.image))))
    paths = {name: root/'sharpened'/f'{name}_sharpened.png' for name in POLICIES}
    paths.update(production=Path('2024-09-27-1154_3-CK-R-SatPs.png'),
                 AS_manual64=Path('out/saturn-matched-transfer/sharpened/as_manual64_sharpened.png'),
                 full_any=Path('out/saturn-refit-full/sharpened/full_any_sharpened.png'))
    arrays = {name: read_png(path)[0] for name, path in paths.items()}
    report['comparison'] = {}
    for name, pixels in arrays.items():
        report['comparison'][name] = dict(path=str(paths[name]), sha256=digest(paths[name]),
                                          variation=measurements(pixels), ring_edges=ring_edges(pixels))
        print(name, 'variation', [round(v['highpass_percent'], 6) for v in measurements(pixels).values()],
              'widths', [round(v['width_10_90_px'], 6) for v in ring_edges(pixels).values()], flush=True)
    report['raw_halves'] = {}
    with np.load(root/'half_stacks.npz') as data:
        for i, name in enumerate(POLICIES):
            sums, weights = data['signal'][i], data['support'][i]
            halves = np.divide(sums, weights, out=np.zeros_like(sums), where=weights>0)
            coverage = weights.sum(0)
            mean = np.divide(sums.sum(0), coverage, out=np.zeros_like(coverage), where=coverage>0)
            np.testing.assert_allclose(mean, load_snapshot(root/f'{name}.npz').image, rtol=0, atol=0)
            report['raw_halves'][name] = dict(full_mean=measurements(mean),
                half_difference_over_two=measurements(mean, difference=(halves[0]-halves[1])/2),
                half_median_coverage=[float(np.median(w[coverage>0])) for w in weights])
    centre = np.rint(center_of_mass(np.maximum(arrays['coherent']-.02*arrays['coherent'].max(), 0))).astype(int)
    report['paired_edges_vs_coherent'] = {}
    for name in ('joint_strong', 'joint_validated'):
        images = dict(production_local=arrays[name], existing_bilinear=arrays['coherent'], AS_manual64=arrays['coherent'])
        values = edge_sensitivity(images, {key: centre for key in images})
        report['paired_edges_vs_coherent'][name] = {side: dict(
            paired_windows=v['paired_windows'],
            candidate_minus_coherent_width_px_min_median_max=v['local_minus_global_width_px_min_median_max'],
            fraction_candidate_wider=v['fraction_local_wider']) for side, v in values.items()}
    report['fit_summary'] = {}
    for key in ('sigma', 'initial_sigma', 'iterations', 'heldout_loss', 'global_heldout_loss', 'minimum_jacobian'):
        values = np.array([s['fit'][key] for s in stats if key in s['fit']])
        report['fit_summary'][key] = dict(count=len(values), min_p25_median_p75_max=np.percentile(values, [0, 25, 50, 75, 100]).tolist())
    report['fit_summary']['residual_motion_rms_px'] = np.percentile(
        [s['residual_vector_rms_px'] for s in stats], [0, 25, 50, 75, 100]).tolist()
    report['fit_summary']['blur_lower_bound_count'] = sum(s['fit'].get('sigma', 1.) < 1e-6 for s in stats)
    report['fit_summary']['blur_upper_bound_count'] = sum(s['fit'].get('sigma', 0.) > 2.-1e-6 for s in stats)
    report['fit_summary']['guard_or_optimizer_reasons'] = dict(Counter(s['fit'].get('reason', s['fit'].get('message', '')) for s in stats))
    probe_path = Path('out/saturn-joint-blur-ceiling.json')
    probe = json.loads(probe_path.read_text())
    if (probe['model_sha256'] != report['hashes']['joint_registration']
            or probe['sampler_sha256'] != report['hashes']['spline_joint_registration']
            or probe['source_sha256'] != digest('tools/joint_blur_ceiling_probe.py')):
        raise ValueError('Blur-ceiling probe source differs')
    for key in ('reference', 'indices', 'quality', 'global_shifts'):
        if probe['hashes'][key] != report['hashes'][key]:
            raise ValueError('Blur-ceiling probe inputs differ')
    expected_positions = np.linspace(0, 5737, 6, dtype=int).tolist()
    if [f['position'] for f in probe['frames']] != expected_positions:
        raise ValueError('Incomplete blur-ceiling probe')
    for frame in probe['frames']:
        if {(v['maximum_sigma'], v['budget']) for v in frame['variants']} != {(s, n) for s in (2., 4.) for n in (100, 300)}:
            raise ValueError('Incomplete blur-ceiling comparisons')
    report['blur_ceiling_probe'] = dict(path=str(probe_path), sha256=digest(probe_path), results=probe)
    drizzle_path = Path('results/registration/native-mono-drizzle.json')
    report['native_drizzle_record'] = dict(path=str(drizzle_path), sha256=digest(drizzle_path))
    report['limits'] = [
        'The four replay methods share frame selection, weights, reference, global shifts and final bilinear sampling.',
        'AS uses its own selection and reference; it is context, not a controlled motion-estimator ablation.',
        'Fine-scale variation includes detail and artifacts; ring transition widths are not calibrated resolution.',
        'Half-difference/2 is an approximate full-mean random-component measure when halves have comparable independent noise.',
        'Both halves share reference and estimator biases; those cancel from their difference.',
        'Separately sharpened halves are review artifacts, not a linear estimate of sharpened full-stack noise.',
        'A held-out score improvement does not establish true motion under an incomplete image-formation model.',
        'This fixed-budget experimental fit is not a production speed implementation.']
    (root/'analysis.json').write_text(json.dumps(report, indent=2)+'\n')
    args.record.parent.mkdir(parents=True, exist_ok=True)
    args.record.write_text(json.dumps(report, indent=2)+'\n')
    comparison_image(root, paths, centre)


if __name__ == '__main__':
    main()
