"""Audit the Saturn fixed-decision AP factorial."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import center_of_mass

from planetrecon.result import load_snapshot
from tools.alignment_noise_experiment import read_png, measurements
from tools.analyse_joint_saturn import sharp_identity
from tools.analyse_refined_registration import ring_edges, edge_sensitivity
from tools.analyse_regularized_ap import fit_summary
from tools.joint_saturn_experiment import digest, CAPTURE, capture_identity
from tools.saturn_ap_decisions import identity, NAMES
from tools.stack_repeatability import analyse_pair


def gallery(root, centre):
    import os
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QGuiApplication, QImage, QPainter, QColor, QFont
    app = QGuiApplication.instance() or QGuiApplication([])
    canvas = QImage(2000, 1640, QImage.Format.Format_RGB32)
    canvas.fill(QColor('black'))
    painter = QPainter(canvas)
    painter.setPen(QColor('white'))
    painter.setFont(QFont('Sans', 18))
    cy, cx = centre
    labels = ['Original peaks', 'Changed positions only', 'Changed AP acceptance only',
              'Changed positions + acceptance', 'Full refinement (previous pilot)']
    for index, (name, label) in enumerate(zip(NAMES, labels)):
        path = root/'sharpened'/f'{name}_sharpened.png'
        crop = QImage(str(path)).copy(int(cx-250), int(cy-115), 500, 230)
        crop = crop.scaled(1000, 460, Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.FastTransformation)
        x, y = (index%2)*1000, (index//2)*510
        painter.drawText(x+12, y+32, label+' - 512 frames')
        painter.drawImage(x, y+45, crop)
    painter.drawText(12, 1560, 'Wavelet 27/0/0/0 + Adaptive Deconvolution 15.6, Contrast Adaptive')
    painter.drawText(12, 1605, 'Same frame IDs, reference, weights and raw sampler; no normalization or added output filter')
    painter.end()
    if not canvas.save(str(root/'comparison.png')):
        raise OSError('Comparison export failed')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--record', type=Path, required=True)
    args = p.parse_args()
    root = args.out
    report = json.loads((root/'report.json').read_text())
    manifest = json.loads((root/'manifest.json').read_text())
    _, indices, quality, _, _, selected, versions, config = identity()
    if (manifest['hashes'] != versions or manifest['config'] != config or
            manifest['capture'] != capture_identity() or manifest['capture_sha256'] != digest(CAPTURE) or
            manifest['selection_positions'] != selected.tolist() or
            manifest['frame_indices'] != indices[selected].tolist()):
        raise ValueError('Replay inputs changed')
    versions['capture'] = manifest['capture_sha256']
    stats = report.pop('statistics')
    if (report['hashes'] != versions or report['config'] != config or report['n_used'] != len(selected) or
            [r['position'] for r in stats] != list(range(len(selected))) or
            [r['selection_position'] for r in stats] != selected.tolist() or
            [r['frame_index'] for r in stats] != indices[selected].tolist()):
        raise ValueError('Incomplete or misassociated replay')
    if report['halves_sha256'] != digest(root/'halves.npz'):
        raise ValueError('Half accumulators changed')
    parity = [r['frozen_baseline_parity_max_px'] for r in stats]
    if len(parity) != 512 or max(parity) > 1e-10:
        raise ValueError('Failed frozen baseline parity')
    report['frozen_baseline_parity'] = dict(frames=len(parity), max_error_px=max(parity))
    report['ap_decisions_totals'] = {key:sum(r['ap_decisions'][key] for r in stats)
        for key in ('both','baseline_only','refined_only')}
    report['forced_guard_disagreements'] = {name: dict(
        all=sum(r['fits'][name]['forced_guard_disagreement'] for r in stats),
        accepted_but_naturally_rejected=sum(r['fits'][name]['field_guard_accepted'] and
            not r['fits'][name]['natural_guard_accepted'] for r in stats))
        for name in ('peaks_only','gates_only','peaks_gates')}
    report['raw_report_sha256'] = digest(root/'report.json')
    report['analyzer_sha256'] = digest(__file__)
    report['fit_summary'] = {name: fit_summary([r['fits'][name] for r in stats]) for name in NAMES}
    accepted = {name: np.array([r['fits'][name]['field_guard_accepted'] for r in stats]) for name in NAMES}
    a, b = accepted['baseline'], accepted['refined']
    report['paired_field_guards'] = dict(both_accepted=int(np.count_nonzero(a & b)),
        both_rejected=int(np.count_nonzero(~a & ~b)), baseline_only=int(np.count_nonzero(a & ~b)),
        refined_only=int(np.count_nonzero(~a & b)))
    recipe = json.loads(Path('out/saturn-coherent-field/sharpened/sharpening_report.json').read_text())
    report['sharpening'] = sharp_identity(root, recipe)
    if len(report['sharpening']['images']) != 15:
        raise ValueError('Expected full and half exports for both methods')
    report['comparison'], report['raw_halves'], arrays = {}, {}, {}
    with np.load(root/'halves.npz') as data:
        for i, name in enumerate(NAMES):
            sums, weights = data['signal'][i], data['support'][i]
            halves = np.divide(sums, weights, out=np.zeros_like(sums), where=weights>0)
            coverage = weights.sum(0)
            mean = np.divide(sums.sum(0), coverage, out=np.zeros_like(coverage), where=coverage>0)
            snapshot = load_snapshot(root/f'{name}.npz')
            np.testing.assert_array_equal(mean, snapshot.image)
            if snapshot.n_used != len(selected):
                raise ValueError('Wrong snapshot frame count')
            report['raw_halves'][name] = dict(full_mean=measurements(mean),
                half_difference_over_two=measurements(mean, difference=(halves[0]-halves[1])/2),
                repeatability=analyse_pair(*halves), snapshot_sha256=digest(root/f'{name}.npz'))
            path = root/'sharpened'/f'{name}_sharpened.png'
            arrays[name] = read_png(path)[0]
            report['comparison'][name] = dict(variation=measurements(arrays[name]), ring_edges=ring_edges(arrays[name]))
    quality = np.maximum(quality[selected], 1e-12)
    report['half_quality_weights'] = dict(fraction=[float(q.sum()/quality.sum()) for q in (quality[::2], quality[1::2])],
        full_effective_frame_count=float(quality.sum()**2/np.sum(quality*quality)))
    baseline = arrays['baseline']
    centre = np.rint(center_of_mass(np.maximum(baseline-.02*baseline.max(), 0))).astype(int)
    report['paired_edges_vs_baseline'] = {}
    for name in NAMES[1:]:
        images = dict(production_local=arrays[name], existing_bilinear=baseline, AS_manual64=baseline)
        report['paired_edges_vs_baseline'][name] = {side: dict(paired_windows=v['paired_windows'],
            candidate_minus_baseline_width_px_min_median_max=v['local_minus_global_width_px_min_median_max'],
            fraction_candidate_wider=v['fraction_local_wider'])
            for side,v in edge_sensitivity(images, {k:centre for k in images}).items()}
    report['sharpened_prior_parity'] = {}
    for name in ('baseline','refined'):
        expected = read_png(Path('out/saturn-peak-refinement/sharpened')/(name+'_sharpened.png'))[0]
        np.testing.assert_array_equal(arrays[name],expected)
        report['sharpened_prior_parity'][name] = True
    report['stage_sha256'] = {path.name:digest(path) for path in sorted((root/'sharpened').glob('*_stages.npz'))}
    report['limits'] = [
        'A 512-frame pilot of the fixed 5738-frame selection, not a full-count comparison against AutoStakkert.',
        'The three factorial controls freeze baseline parents, stage guards, robust factors and final guard decisions.',
        'Controls are causal diagnostics, not deployable candidates; final guard decisions are deliberately forced.',
        'Confidence of newly accepted APs uses robust factors calculated from baseline measurements.',
        'Fine variation mixes detail and artifacts; ring widths are descriptive, not calibrated resolution.',
        'Half-difference/2 cancels shared reference and estimator biases, so cannot measure total noise.',
        'Halves share the reference and are temporally interleaved, not independent atmospheric realizations.',
        'Fourier correlations are broad-band diagnostics without calibrated significance or resolution thresholds.',
        'Gaussian residuals are used only for measurement; no output filtering or normalization was introduced.',
        'This isolates changes on baseline stage inputs, not a complete decomposition of the full refined algorithm.']
    text = json.dumps(report, indent=2)+'\n'
    (root/'analysis.json').write_text(text)
    args.record.write_text(text)
    gallery(root, centre)
    for name in NAMES:
        print(name, 'sharp variation', [v['highpass_percent'] for v in report['comparison'][name]['variation'].values()],
            'raw half difference', [v['highpass_percent'] for v in report['raw_halves'][name]['half_difference_over_two'].values()],
            'widths', [v['width_10_90_px'] for v in report['comparison'][name]['ring_edges'].values()])
    print('paired edges', report['paired_edges_vs_baseline'])


if __name__ == '__main__':
    main()
