"""Describe a fixed-prior noise/weighting pilot without selecting a prior or model."""
import argparse
from pathlib import Path
import sys
import json
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import h5py
import numpy as np
from tools.audit_full_gate1 import write_json, file_hash
from tools.experiment_stages import StageStore


def run(studies, input_path, directory):
    from planetrecon import constants as C
    from planetrecon.optics import bin_box
    from tools.input_compatibility import archived_input_errors
    if errors := archived_input_errors(input_path):
        raise ValueError(errors)
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory/'protocol.json', {
        'question': 'At fixed domain, ridge and data selection, how much do noise and variance weighting change a numerically certified optical-scene solution?',
        'runner_sha256': file_hash(__file__), 'input_sha256': file_hash(input_path),
        'required_cases': ['expected/scalar', 'expected/spatial', 'observed/scalar', 'observed/spatial'],
        'decision': 'Diagnostic only; retain all cases and numerical failures. Truth metrics never select weighting, prior or initialization. No scientific acceptance threshold or Gate-1/Q2/Q3 authorization.',
        'budget_rule': 'Per-case caps may differ after measured conditioning; all objective, initialization, frame and tolerance settings must match. A failed numerical certificate prevents qualification.',
        'q3_authorized': False})
    with h5py.File(input_path) as f:
        bin_factor = int(json.loads(f['config/json_utf8'][()])['bin_factor'])
        flux = float(f['config'].attrs['source_rate_scale'])*float(f['config'].attrs['texp_s'])/C.T0_S
        truth = {crop: f[f'object/{crop}_truth'][...].astype(float)*flux for crop in ('feature', 'bland')}
        origins = {crop: tuple(f[f'object/{crop}_crop_origin'][...]) for crop in truth}
    rows, images, signatures = [], {}, []
    for study in studies:
        protocol = json.loads((study/'protocol.json').read_text())
        report = json.loads((study/'report.json').read_text())
        if protocol['input_sha256'] != file_hash(input_path):
            raise ValueError('mixed inputs')
        key = protocol['noise']+'/'+protocol['weighting']
        if key in images:
            raise ValueError('duplicate case')
        signatures.append({k: v for k, v in protocol.items() if k not in ('noise', 'weighting', 'budgets', 'wall_budget_s')})
        store = StageStore(study/'stages', protocol)
        def missing():
            raise ValueError('missing completed numerical stage')
        image, info = store.run('budget-'+str(protocol['budgets'][-1]), missing)
        crop = protocol['crop']
        ox, oy = origins[crop]
        h, w = truth[crop].shape
        det = bin_box(image, bin_factor)[oy:oy+h, ox:ox+w]
        images[key] = det
        rows.append({'case': key, 'numerical_status': report['status'],
                     'numerical_pilot_passed': report['numerical_pilot_passed'],
                     'source_input_unchanged': report['source_input_unchanged'],
                     'input_protocol_sha256': file_hash(study/'protocol.json'),
                     'input_report_sha256': file_hash(study/'report.json'),
                     'detector_truth_relative_mse': float(np.sum((det-truth[crop])**2)/np.sum(truth[crop]**2)),
                     'ridge': protocol['ridge'], 'budgets': protocol['budgets'], 'runs': report['runs'], 'wall_s': report['wall_s'],
                     'image_relative_budget_change': report['image_relative_change']})
    required = {'expected/scalar', 'expected/spatial', 'observed/scalar', 'observed/spatial'}
    if set(images) != required or any(s != signatures[0] for s in signatures):
        raise ValueError('incomplete or incompatible noise/weighting matrix')
    changes = {noise: float(np.linalg.norm(images[noise+'/spatial']-images[noise+'/scalar'])/np.linalg.norm(images[noise+'/scalar'])) for noise in ('expected', 'observed')}
    noise_changes = {w: float(np.linalg.norm(images['observed/'+w]-images['expected/'+w])/np.linalg.norm(images['expected/'+w])) for w in ('scalar', 'spatial')}
    result = {'status': 'diagnostic', 'complete': True, 'q3_authorized': False,
              'all_numerical_pilots_passed': all(r['numerical_pilot_passed'] and r['source_input_unchanged'] for r in rows),
              'cases': rows, 'weighting_image_relative_changes': changes,
              'noise_image_relative_changes': noise_changes,
              'limits': 'One seed/regime/crop, three frames, one unselected ridge/domain. Oracle frozen variances. This is not full P2/P3 qualification or a weighting recommendation.'}
    write_json(directory/'report.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--studies', nargs=4, type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.studies, args.input, args.out), indent=2))
