"""Verify paired reference interventions and measure common output response."""
import json
from pathlib import Path

import numpy as np

from planetrecon.result import load_snapshot
from tools.analyse_joint_saturn import sharp_identity
from tools.coherent_saturn_experiment import inputs
from tools.joint_saturn_experiment import digest, CAPTURE
from tools.reference_imprint_probe import DOSES, SCALES, variants, probes, metrics, response_summary
from tools.validate_local_warp_centring import image_errors
from tools.alignment_noise_experiment import measurements


def geometric_component(response, baseline, mask):
    """Describe translation-like sensitivity; never alter any output image."""
    gy, gx = np.gradient(baseline)
    design = np.column_stack([gx[mask], gy[mask]])
    delta = np.linalg.lstsq(design, response[mask], rcond=None)[0]
    residual = response[mask]-design@delta
    energy = np.mean(response[mask]**2)
    return dict(translation_tangent_xy_px_per_reference_adu=delta.tolist(),
        explained_energy_fraction=float(1-np.mean(residual**2)/energy) if energy>1e-30 else None,
        residual_rms=float(np.sqrt(np.mean(residual**2))))


def verify_report(root):
    report = json.loads((root/'report.json').read_text())
    for path, value in report['hashes'].items():
        if path.startswith(('tools/', 'planetrecon/')) and digest(path) != value:
            raise ValueError('Experiment source changed: '+path)
    if report['probe_seed'] != 9031 or report['doses'] != list(DOSES) or report['scales'] != SCALES:
        raise ValueError('Probe specification changed')
    with np.load(root/'probe.npz') as data:
        patterns = {k: data[k].copy() for k in SCALES}
        mask, reference = data['mask'].copy(), data['reference'].copy()
    expected, expected_mask = probes(reference)
    np.testing.assert_array_equal(mask, expected_mask)
    for key in patterns:
        np.testing.assert_array_equal(patterns[key], expected[key])
    return report, patterns, mask


def sharp_arrays(root, keys):
    result = {}
    for key in keys:
        with np.load(root/'sharpened'/f'{key}_stages.npz') as data:
            result[key] = data['sharpened'].mean(2)
    return result


def agreement(a, b, mask):
    x, y = a[mask], b[mask]
    return dict(correlation=float(np.corrcoef(x, y)[0, 1]) if np.std(x)>1e-15 and np.std(y)>1e-15 else None,
                difference_rms=float(np.sqrt(np.mean((x-y)**2))))


def render_responses(roots):
    import os
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from PySide6.QtGui import QGuiApplication, QImage, QPainter, QColor, QFont
    app = QGuiApplication.instance() or QGuiApplication([])
    canvas = QImage(1530, 720, QImage.Format.Format_RGB32)
    canvas.fill(QColor('black'))
    painter = QPainter(canvas)
    painter.setPen(QColor('white'))
    painter.setFont(QFont('Sans', 14))
    painter.drawText(10, 28, 'Reference-only texture intervention: response is a DIFFERENCE, not a reconstructed planet')
    roots = {name: Path(root) for name, root in roots.items()}
    with np.load(roots['production']/'probe.npz') as data:
        patterns, mask = {k: data[k] for k in SCALES}, data['mask']
    crop = np.s_[70:330, 90:600]
    for row, (name, pattern) in enumerate(patterns.items()):
        responses = {method: (load_snapshot(root/f'{name}_1_plus.npz').image-
                              load_snapshot(root/f'{name}_1_minus.npz').image)/2 for method, root in roots.items()}
        bound = max(float(np.percentile(abs(value[mask]), 99)) for value in responses.values())
        arrays = [pattern, responses['production'], responses['coherent']]
        labels = [f'{name}: probe display range +/-3 ADU', f'Production response, +/-{bound:.4f} ADU/ADU',
                  f'Coherent response, +/-{bound:.4f} ADU/ADU']
        for col, (array, label) in enumerate(zip(arrays, labels)):
            limit = 3. if col == 0 else max(bound, 1e-12)
            pixels = np.ascontiguousarray(np.rint(255*np.clip(.5+.5*array[crop]/limit, 0., 1.)), dtype=np.uint8)
            image = QImage(pixels.data, 510, 260, pixels.strides[0], QImage.Format.Format_Grayscale8).copy()
            x, y = col*510, 50+row*320
            painter.drawText(x+6, y+23, label)
            painter.drawImage(x, y+35, image)
    painter.drawText(10, 710, '512 identical frame IDs/weights per branch; fixed global shifts and AP geometry; dose +/-1 reference ADU')
    painter.end()
    path = Path('out/reference-imprint-response.png')
    if not canvas.save(str(path)):
        raise OSError(path)


def main():
    recipe = json.loads(Path('out/saturn-coherent-field/sharpened/sharpening_report.json').read_text())
    output = dict(controls={}, saturn={}, limits=[
        'Artificial perturbations measure sensitivity, not the amount of native reference noise.',
        'Two fixed probe patterns do not span all reference errors or constitute a statistical population.',
        'Global shifts and AP eligibility/geometry stay fixed; only matching templates change.',
        'The coherent spline design and gradient kernels also stay fixed at the original reference.',
        'Input observations are identical in paired branches; output pixels are neither filtered nor normalized.',
        'Positive projected response shows texture transfer; other changes may be geometric sensitivity.',
        'Checkpoint prefixes span progressively more capture time as well as more frames.',
        'Interleaved halves are a reproducibility check, not proof of independent atmospheric or detector noise.',
        'A 512-frame diagnostic is not a full-count comparison with AutoStakkert.',
        'Sharpened finite differences are nonlinear responses, not calibrated transfer functions.'])
    _, indices, _, _, _, hashes = inputs()
    capture_hash = digest(CAPTURE)
    roots = {}
    for method in ('production', 'coherent'):
        root = Path(f'out/reference-imprint-control-{method}')
        report, patterns, mask = verify_report(root)
        report['sharpening'] = sharp_identity(root, recipe)
        for case, values in report['cases'].items():
            if report['frames'] != 32:
                raise ValueError('Incomplete control')
            with np.load(root/f'{case}.npz') as data:
                actual = response_summary({k: data[k] for k in variants()}, patterns, mask)
                if actual != values['response']:
                    raise ValueError('Control response differs')
                values['response_geometry'] = {}
                for name in SCALES:
                    for dose in DOSES:
                        delta = (data[f'{name}_{dose:g}_plus']-data[f'{name}_{dose:g}_minus'])/(2*dose)
                        values['response_geometry'][f'{name}_{dose:g}'] = geometric_component(delta, data['base'], mask)
            sharpened = sharp_arrays(root, [f'{case}_{key}' for key in [*variants(), 'oracle']])
            for key, v in values['variants'].items():
                v['sharpened_error'] = image_errors(sharpened[f'{case}_{key}'], sharpened[f'{case}_oracle'], mask)
            values['sharpened_response'] = response_summary(
                {key: sharpened[f'{case}_{key}'] for key in variants()}, patterns, mask)
        output['controls'][method] = report
        root = Path(f'out/reference-imprint-saturn-{method}')
        roots[method] = str(root)
        report, patterns, mask = verify_report(root)
        for key, value in hashes.items():
            if report['hashes'][key] != value:
                raise ValueError('Capture inputs changed')
        if capture_hash != report['hashes']['capture'] or report['frames'] != 512:
            raise ValueError('Capture identity/count differs')
        positions = np.linspace(0, len(indices)-1, 512, dtype=int)
        if report['positions'] != positions.tolist() or report['frame_indices'] != indices[positions].tolist():
            raise ValueError('Capture selection differs')
        if digest(root/'accumulators.npz') != report['accumulators_sha256']:
            raise ValueError('Changed accumulators')
        report['sharpening'] = sharp_identity(root, recipe)
        with np.load(root/'accumulators.npz') as data:
            if data['keys'].tolist() != list(variants()):
                raise ValueError('Variant order differs')
            total, support = data['signal'], data['support']
            means = np.divide(total.sum(1), support.sum(1), out=np.zeros_like(total[:, 0]), where=support.sum(1)>0)
            halves = np.divide(total, support, out=np.zeros_like(total), where=support>0)
        for k, key in enumerate(variants()):
            np.testing.assert_array_equal(means[k], load_snapshot(root/f'{key}.npz').image)
        images = dict(zip(variants(), means))
        actual = response_summary(images, patterns, mask)
        if actual != report['checkpoints']['512']:
            raise ValueError('Full response differs')
        report['half_response_agreement'] = {}
        report['dose_response_agreement'] = {}
        report['response_geometry'] = {}
        report['disc_response'] = {}
        for name, pattern in patterns.items():
            derivatives = []
            for dose in DOSES:
                plus, minus = [list(variants()).index(f'{name}_{dose:g}_{s}') for s in ('plus', 'minus')]
                d0, d1 = (halves[plus]-halves[minus])/(2*dose)
                report['half_response_agreement'][f'{name}_{dose:g}'] = agreement(d0, d1, mask)
                derivatives.append((means[plus]-means[minus])/(2*dose))
                report['response_geometry'][f'{name}_{dose:g}'] = geometric_component(derivatives[-1], images['base'], mask)
                report['disc_response'][f'{name}_{dose:g}'] = measurements(images['base'], difference=derivatives[-1])
            report['dose_response_agreement'][name] = agreement(*derivatives, mask)
        report['sharpened_response'] = response_summary(sharp_arrays(root, variants()), patterns, mask)
        output['saturn'][method] = report
        print(method, json.dumps(dict(raw=actual, halves=report['half_response_agreement'],
                                     doses=report['dose_response_agreement']), indent=2), flush=True)
    gate_path = Path('out/reference-imprint-gates.json')
    gates = json.loads(gate_path.read_text())
    expected = np.linspace(0, len(indices)-1, 512, dtype=int)[np.linspace(0, 511, 16, dtype=int)].tolist()
    if gates['positions'] != expected or set(gates['methods']) != {'production', 'coherent'}:
        raise ValueError('Incomplete gate diagnostic')
    for path, value in gates['hashes'].items():
        if path.startswith(('tools/', 'planetrecon/')) and digest(path) != value:
            raise ValueError('Changed gate source')
    for key, value in hashes.items():
        if gates['hashes'][key] != value:
            raise ValueError('Changed gate inputs')
    gates['summary'] = {}
    for method, rows in gates['methods'].items():
        if [row['position'] for row in rows] != expected or any(set(row['variants']) != set(variants()) for row in rows):
            raise ValueError('Missing gate rows/variants')
        summary = {}
        for key in variants():
            if key == 'base':
                continue
            values = [row['variants'][key] for row in rows]
            baselines = [row['variants']['base'] for row in rows]
            summary[key] = dict(
                changed_ap_median_max=np.percentile([v['changed_active_aps'] for v in values], [50, 100]).tolist(),
                field_difference_median_max_px=np.percentile([v['field_difference_rms_px'] for v in values], [50, 100]).tolist(),
                frames_with_stage_guard_change=sum(v['stage_acceptance'] != b['stage_acceptance'] for v, b in zip(values, baselines)))
            if method == 'coherent':
                summary[key]['frames_with_spline_guard_change'] = sum(
                    v['spline_guard']['field_guard_accepted'] != b['spline_guard']['field_guard_accepted'] for v, b in zip(values, baselines))
        gates['summary'][method] = summary
    output['gate_diagnostic'] = dict(path=str(gate_path), sha256=digest(gate_path), results=gates)
    Path('results/registration/reference-imprint.json').write_text(json.dumps(output, indent=2)+'\n')
    render_responses(roots)


if __name__ == '__main__':
    main()
