"""Verify regularization screening and score the exact sharpening workflow."""
import json
from pathlib import Path

import numpy as np

from tools.ap_stability_screen import choose, hashes
from tools.analyse_joint_saturn import sharp_identity
from tools.joint_saturn_experiment import digest, CAPTURE
from tools.coherent_saturn_experiment import inputs
from tools.regularized_ap_fit import POLICIES
from tools.validate_local_warp_centring import image_errors


def main():
    root = Path('out/ap-stability-screen')
    report = json.loads((root/'report.json').read_text())
    if report['hashes'] != hashes() or report['policies'] != {k: list(v) for k, v in POLICIES.items()}:
        raise ValueError('Screen implementation changed')
    if report['capture_sha256'] != digest(CAPTURE):
        raise ValueError('Capture changed')
    _, indices, _, _, _, input_hashes = inputs()
    if report['sensitivity']['hashes'] != input_hashes:
        raise ValueError('Prepared inputs changed')
    expected = np.linspace(0, len(indices)-1, 512, dtype=int)[np.linspace(0, 511, 16, dtype=int)].tolist()
    if report['sensitivity']['positions'] != expected or [r['position'] for r in report['sensitivity']['rows']] != expected:
        raise ValueError('Incomplete sensitivity screen')
    if report['selection'] != choose(report['controls'], report['sensitivity']['summary']):
        raise ValueError('Candidate decision changed')
    recipe = json.loads(Path('out/saturn-coherent-field/sharpened/sharpening_report.json').read_text())
    report['sharpening'] = sharp_identity(root, recipe)
    report['raw_artifact_sha256'] = {}
    for case, values in report['controls'].items():
        path = root/f'{case}.npz'
        report['raw_artifact_sha256'][case] = digest(path)
        with np.load(path) as data:
            mask, target = data['mask'], data['noiseless_oracle']
            for name in POLICIES:
                actual = image_errors(data['clean_'+name], target, mask)
                if actual != values[name]['clean_image']:
                    raise ValueError('Control artifacts changed')
        with np.load(root/'sharpened'/f'{case}_oracle_stages.npz') as data:
            target = data['sharpened'].mean(2)
        for name in POLICIES:
            with np.load(root/'sharpened'/f'{case}_{name}_stages.npz') as data:
                actual = image_errors(data['sharpened'].mean(2), target, mask)
                values[name]['sharpened_error_linear_0_1'] = actual
        print(case, {name: dict(clean=round(v['clean_image']['rmse_adu'], 6),
            field=round(v['field_rmse_px'], 6), sharp=round(v['sharpened_error_linear_0_1']['rmse_adu'], 6))
            for name, v in values.items()}, flush=True)
    report['limits'] = [
        'This changes only the AP-to-field fit; noisy or discontinuous AP measurements remain.',
        'Reference perturbations measure sensitivity and do not estimate native reference noise.',
        'Artificial Gaussian scenes and fixed smooth shears do not cover real atmospheric motion.',
        'More stable fields can be less accurate; the fixed selection requires both properties.',
        'A screen pass is not a demonstrated improvement in a full real-capture stack.',
        'The amplitude penalty can suppress real offsets, unlike pure bending regularization.']
    Path('results/registration/ap-stability-screen.json').write_text(json.dumps(report, indent=2)+'\n')
    print('Selected:', report['selection']['selected'], flush=True)


if __name__ == '__main__':
    main()
