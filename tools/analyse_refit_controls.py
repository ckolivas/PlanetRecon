"""Collect matched synthetic and full-count refit evidence with provenance checks."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from tools.coherent_saturn_experiment import inputs
from tools.refit_saturn_experiment import cached_validation


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    prior_path = Path('results/registration/validated-local-registration.json')
    prior = json.loads(prior_path.read_text())
    result = dict(production_changed=False,brightness_normalization=False,output_filtering=False,
                  prior_record=dict(path=str(prior_path),sha256=sha(prior_path)),cases=[],
                  code_sha256={str(p):sha(p) for p in [Path('tools')/f'{name}.py' for name in
                    ('refit_registration','refit_saturn_experiment','coherent_registration','validated_registration')]})
    for scene in ('saturn','texture'):
        root = Path(f'out/refit-{scene}-controls')
        report = json.loads((root/'analysis.json').read_text())
        if report['refit_policy'] != 'full_any':
            raise ValueError('Unexpected synthetic refit policy')
        if {k:report['sharpening'][k] for k in prior['sharpening']} != prior['sharpening']:
            raise ValueError('Control sharpening recipe changed')
        for name,c in report['scenes'][scene].items():
            old = next(v for v in prior['cases'] if v['group']=='primary' and
                       v['scene']==scene and v['condition']==name)
            if (c['seed'],c['frame_count']) != (old['seed'],old['frame_count']):
                raise ValueError('Synthetic frame count or random seed changed')
            oracle_hash = sha(root/f'{scene}_{name}_noiseless_oracle.png')
            if oracle_hash != old['oracle_sha256']:
                raise ValueError('Control oracle changed')
            local = dict(c['variants']['local'])
            local.pop('image_vs_mean_blurred_scene')
            result['cases'].append(dict(scene=scene,condition=name,frames=c['frame_count'],seed=c['seed'],
                source=dict(path=str(root/'analysis.json'),sha256=sha(root/'analysis.json')),
                oracle_sha256=oracle_hash,coherent=old['coherent'],validated=old['validated'],full_any=local))
    scaled_root = Path('out/refit-scaled-controls')
    scaled_report = json.loads((scaled_root/'analysis.json').read_text())
    if scaled_report['refit_policy'] != 'full_scaled':
        raise ValueError('Unexpected scaled refit policy')
    if {k:scaled_report['sharpening'][k] for k in prior['sharpening']} != prior['sharpening']:
        raise ValueError('Scaled-control sharpening recipe changed')
    result['scaled_cases'] = []
    for scene,cases in scaled_report['scenes'].items():
        for name,c in cases.items():
            old = next(v for v in prior['cases'] if v['group']=='primary' and v['scene']==scene and v['condition']==name)
            oracle_hash = sha(scaled_root/f'{scene}_{name}_noiseless_oracle.png')
            if oracle_hash != old['oracle_sha256'] or (c['seed'],c['frame_count']) != (old['seed'],old['frame_count']):
                raise ValueError('Scaled-control inputs differ')
            local = dict(c['variants']['local'])
            local.pop('image_vs_mean_blurred_scene')
            result['scaled_cases'].append(dict(scene=scene,condition=name,frames=c['frame_count'],seed=c['seed'],
                source=dict(path=str(scaled_root/'analysis.json'),sha256=sha(scaled_root/'analysis.json')),
                oracle_sha256=oracle_hash,coherent=old['coherent'],validated=old['validated'],full_scaled=local))
    _,indices,quality,_,_,hashes = inputs()
    hashes['validated_registration'] = sha('tools/validated_registration.py')
    cached = cached_validation(Path('out/saturn-validated-field'),hashes,len(indices))
    coherent = {}
    for path in Path('out/saturn-coherent-field').glob('shard_*.npz'):
        with np.load(path) as data:
            stats = json.loads(str(data['metadata']))['fit_statistics']
            coherent.update({int(i):s for i,s in zip(data['positions'],stats,strict=True)})
    result['gate_groups'] = {}
    for n in range(3):
        mask = np.array([s['accepted_halves']==n for s in cached])
        positions = np.flatnonzero(mask)
        result['gate_groups'][str(n)] = dict(frames=int(mask.sum()),
            weight_fraction=float(quality[mask].sum()/quality.sum()),median_quality=float(np.median(quality[mask])),
            coherent_already_global=sum(not coherent[int(i)]['field_guard_accepted'] for i in positions),
            median_full_fit_points=float(np.median([coherent[int(i)]['points'] for i in positions])))
    pilot = Path('out/saturn-refit-ablation')
    result['pilot'] = json.loads((pilot/'analysis.json').read_text())
    selected = result['pilot']['sample_positions']
    compared = 0
    max_score_difference = 0.
    for path in pilot.glob('shard_*.npz'):
        with np.load(path) as data:
            stats = json.loads(str(data['metadata']))['fit_statistics']
            for i,s in zip(data['positions'],stats,strict=True):
                old = cached[selected[i]]
                if old['accepted_halves'] != s['accepted_halves']:
                    raise ValueError('Pilot validation decisions changed')
                for a,b in zip(old['validation'],s['validation'],strict=True):
                    for key in ('accepted','global_blur_model','local_blur_model'):
                        if a[key] != b[key]:
                            raise ValueError('Pilot validation model selection changed')
                    for key in ('global_loss','local_loss','inverse_error_px'):
                        difference = abs(a[key]-b[key])
                        max_score_difference = max(max_score_difference,difference)
                        np.testing.assert_allclose(a[key],b[key],atol=1e-10,rtol=0)
                compared += 1
    if compared != len(selected):
        raise ValueError('Incomplete pilot checks')
    result['pilot_decisions_reproduced'] = compared
    result['pilot_max_score_difference'] = max_score_difference
    full_path = Path('out/saturn-refit-full/analysis.json')
    result['full'] = json.loads(full_path.read_text())
    if result['full']['n_used']!=len(indices) or result['full']['frame_indices']!=indices.tolist():
        raise ValueError('Full replay frame identities changed')
    with np.load('out/saturn-coherent-field/coherent.npz') as before, np.load(full_path.parent/'coherent.npz') as after:
        delta = after['image']-before['image']
        result['coherent_replay_max_pixel_difference_adu'] = float(abs(delta).max())
        result['coherent_replay_max_support_difference'] = float(abs(after['coverage']-before['coverage']).max())
        np.testing.assert_allclose(after['image'],before['image'],atol=1e-10,rtol=0)
        np.testing.assert_allclose(after['coverage'],before['coverage'],atol=1e-10,rtol=0)
    result['limits'] = [
        'A full-data refit after motion gating is not itself independently validated.',
        'Synthetic cases are a limited known-motion family; the Saturn capture has no known geometric truth.',
        'The 512-frame pilot and 5738-frame stack respond differently to the same nonlinear sharpening.',
        'No production defaults, photometric normalization, or output filtering have been changed.']
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(len(result['cases'])+len(result['scaled_cases']),'synthetic cases;',compared,'pilot decisions reproduced;',len(indices),'full replay frames')


if __name__ == '__main__':
    main()
