"""Collect reproducible blur-estimation controls and the complete Saturn replay."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def source(path):
    return dict(path=str(path), sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = dict(production_changed=False, output_filtering=False, brightness_normalization=False,
        model=source('tools/blur_matched_registration.py'), cases=[])
    recipe = None
    identities = set()
    for folder in ('blur-matched-saturn-controls', 'blur-matched-texture-controls', 'blur-matched-repeat-controls'):
        path = Path('out')/folder/'analysis.json'
        report = json.loads(path.read_text())
        if report['model_sha256'] != result['model']['sha256']:
            raise ValueError('Control model differs')
        current = {k:report['sharpening'][k] for k in ('wavelet','deconvolution','implementation_sha256')}
        if recipe is None:recipe = current
        elif recipe != current:raise ValueError('Sharpening differs')
        for scene, cases in report['scenes'].items():
            for condition, case in cases.items():
                identity = (scene, condition, case['seed'])
                if identity in identities:raise ValueError('Duplicate control')
                identities.add(identity)
                values = {k:v for k,v in case.items() if k != 'statistics'}
                values.update(scene=scene, condition=condition, source=source(path),
                    baseline_max_difference_adu=report['maximum_coherent_baseline_difference_adu'])
                result['cases'].append(values)
    expected = {(scene, condition, seed) for scene in ('saturn','texture')
                for condition in ('static_noise','static_blur_noise','motion_blur_noise','offset_motion_blur')
                for seed in (9017,9018)}
    if identities != expected:raise ValueError('Incomplete controls')
    result['synthetic_frames'] = sum(c['frames'] for c in result['cases'])
    result['sharpening'] = recipe
    path = Path('out/blur-bias-probe.json')
    result['probe_source'] = source(path)
    result['probe'] = json.loads(path.read_text())
    if result['probe']['model_sha256'] != result['model']['sha256']:
        raise ValueError('Probe model differs')
    range_path = Path('out/blur-range-probe.json')
    result['range_probe_source'] = source(range_path)
    result['range_probe'] = json.loads(range_path.read_text())
    if result['range_probe']['model_sha256'] != result['model']['sha256']:
        raise ValueError('Range probe model differs')
    path = Path('out/saturn-blur-matched-full/analysis.json')
    real = json.loads(path.read_text())
    if real['n_used'] != 5738 or real['hashes']['blur_matched_registration'] != result['model']['sha256']:
        raise ValueError('Full replay identity differs')
    if {k:real['sharpening'][k] for k in recipe} != recipe:
        raise ValueError('Real sharpening differs')
    for item in real['sharpening']['images'].values():
        for key in ('source','output'):
            if source(item[key])['sha256'] != item[key+'_sha256']:
                raise ValueError('Sharpening input or output changed')
    result['real_source'] = source(path)
    result['real'] = {k:v for k,v in real.items() if k not in ('frame_indices','sample_positions')}
    for key in ('reference','indices','quality','global_shifts','model_code'):
        if result['range_probe']['hashes'][key] != real['hashes'][key]:
            raise ValueError('Range probe inputs differ')
    cached = {}
    for shard in sorted(path.parent.glob('shard_*.npz')):
        with np.load(shard) as data:
            stats = json.loads(str(data['metadata']))['fit_statistics']
            for position, values in zip(data['positions'], stats, strict=True):
                if int(position) in cached:raise ValueError('Duplicate frame position')
                cached[int(position)] = values
    if set(cached) != set(range(5738)):raise ValueError('Incomplete replay traces')
    for row in result['range_probe']['frames']:
        if cached[row['position']]['global_sigma'] != row['sigma_2']:
            raise ValueError('CPU range probe differs from CUDA replay')
    with np.load(path.parent/'coherent.npz') as now, np.load('out/saturn-refit-full/coherent.npz') as old:
        result['baseline_replay_max_difference_adu'] = float(np.max(abs(now['image']-old['image'])))
        for key in ('image','coverage'):
            np.testing.assert_allclose(now[key], old[key], atol=1e-10, rtol=0)
    from tools.alignment_noise_experiment import read_png, measurements
    from tools.analyse_refined_registration import ring_edges
    result['original_comparison_context'] = {}
    for name, image_path in {
        'production_PR':'2024-09-27-1154_3-CK-R-SatPs.png',
        'AS_manual64':'AS_F5738/2024-09-27-1154_3-CK-R-Sat_r64_lapl4_ap54s.png'}.items():
        pixels = read_png(image_path)[0]
        result['original_comparison_context'][name] = dict(source=source(image_path),
            variation=measurements(pixels), ring_edges=ring_edges(pixels))
    result['context_limit'] = ('Original user-sharpened PR/AS files provide context, not a controlled estimator ablation; '
                               'AS uses its own reference, selection and transfer response.')
    result['limits'] = [
        'One real capture has no geometric truth; variation and widths are not noise or resolution measurements.',
        'Additional isotropic Gaussian blur from zero to two pixels cannot represent all seeing or sharpen a softer reference.',
        'Blur and motion remain confounded; estimating blur after a biased local fit can underestimate it.',
        'AP positions, eligibility, spline design, response kernels and support are fixed at the original reference.',
        'Only model templates change; original frame brightness and final output sampling are unchanged.']
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    print(result['synthetic_frames'], 'control frames,', len(result['probe']['frames']),
          'stationary probe frames, 5738 real frames; baseline reproduced')


if __name__ == '__main__':main()
