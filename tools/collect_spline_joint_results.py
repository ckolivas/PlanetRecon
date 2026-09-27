"""Verify and collect spline-model controls and independent detector probes."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from tools.validate_local_warp_centring import image_errors


def source(path):
    return dict(path=str(path), sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())


def verify_hashes(report):
    for path, digest in report['hashes'].items():
        if source(path)['sha256'] != digest:
            raise ValueError(f'Changed source: {path}')


def verify_sharpening(root, expected):
    report = json.loads((root/'sharpened/sharpening_report.json').read_text())
    for key in ('wavelet', 'deconvolution', 'implementation_sha256'):
        if report[key] != expected[key]:
            raise ValueError(f'Sharpening changed: {key}')
    for item in report['images'].values():
        for key in ('source', 'output'):
            if source(item[key])['sha256'] != item[key+'_sha256']:
                raise ValueError('Changed sharpening artifact')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    model, sampler = source('tools/joint_registration.py'), source('tools/spline_joint_registration.py')
    result = dict(model=model, sampler=sampler, production_changed=False,
                  normalization=False, output_filtering=False, real_capture_replay=False, controls=[])
    expected_recipe = json.loads(Path('out/saturn-refit-full/sharpened/sharpening_report.json').read_text())
    result['recipe'] = {k: expected_recipe[k] for k in ('wavelet', 'deconvolution', 'implementation_sha256')}
    cases = {'static_noise', 'static_blur_noise', 'motion_blur_noise', 'offset_motion_blur'}
    for scene in ('saturn', 'texture'):
        root = Path('out')/f'joint-spline-{scene}-controls'
        report = json.loads((root/'analysis.json').read_text())
        previous = json.loads((Path('out')/f'joint-fit-{scene}-controls/analysis.json').read_text())
        if report['model_sha256'] != model['sha256'] or report['sampler_sha256'] != sampler['sha256']:
            raise ValueError('Control model differs')
        if set(report['scenes']) != {scene} or set(report['scenes'][scene]) != cases:
            raise ValueError('Incomplete controls')
        config = dict(report['config'])
        if config.pop('sampler') != 'spline' or config != previous['config']:
            raise ValueError('Configuration differs')
        verify_sharpening(root, expected_recipe)
        for name, case in report['scenes'][scene].items():
            if case['seed'] != 9017 or case['frames'] != 48:
                raise ValueError('Unexpected control size or seed')
            old = previous['scenes'][scene][name]
            if case['oracle_sha256'] != old['oracle_sha256']:
                raise ValueError('Control truth differs')
            result['controls'].append(dict(scene=scene, case=name, source=source(root/'analysis.json'),
                frames=case['frames'], seed=case['seed'], variants=case['variants'],
                diagnostics=case['joint_diagnostics'],
                previous_bicubic_variants=old['variants'],
                baseline_max_difference_adu=report['maximum_coherent_baseline_difference_adu']))
    path = Path('out/joint-translation-spline-probe.json')
    probe = json.loads(path.read_text())
    if probe['model_sha256'] != model['sha256'] or probe['sampler_sha256'] != sampler['sha256']:
        raise ValueError('Translation model differs')
    if probe['generator'] != 'scipy_spline' or probe['sampler'] != 'spline':
        raise ValueError('Wrong translation probe')
    expected = {(s, shift, blur, noise) for s in ('saturn', 'texture')
                for shift in ((.25, .5), (.6, -.4), (-.7, .35)) for blur in (0., 1.) for noise in (0., 2.)}
    actual = {(f['scene'], tuple(f['shift_xy']), f['blur'], f['noise']) for f in probe['frames']}
    if actual != expected or len(probe['frames']) != len(expected):
        raise ValueError('Incomplete translation test')
    result['translation'] = dict(source=source(path), results=probe)
    path = Path('out/joint-analytic-sampling-probe.json')
    probe = json.loads(path.read_text())
    verify_hashes(probe)
    expected = {(s, shift, blur, noise) for s in ('resolved', 'fine')
                for shift in ((.25, .5), (.6, -.4), (-.7, .35)) for blur in (0., 1.) for noise in (0., 2.)}
    actual = {(f['scene'], tuple(f['shift_xy']), f['blur'], f['noise']) for f in probe['frames']}
    if actual != expected or len(probe['frames']) != len(expected):
        raise ValueError('Incomplete independent translation test')
    if any(set(f['variants']) != {'bicubic', 'spline'} for f in probe['frames']):
        raise ValueError('Missing sampler comparison')
    result['analytic_translation'] = dict(source=source(path), results=probe)
    root = Path('out/joint-analytic-motion')
    motion = json.loads((root/'report.json').read_text())
    verify_hashes(motion)
    motion['sharpening'] = verify_sharpening(root, expected_recipe)
    if motion['frames'] != 24 or set(motion['scenes']) != {'resolved', 'fine'}:
        raise ValueError('Incomplete independent motion test')
    for scene, case in motion['scenes'].items():
        if len(case['statistics']) != motion['frames'] or case['quadrature_max_difference_adu'] > 1e-3:
            raise ValueError('Incomplete or inaccurate quadrature')
        with np.load(root/f'{scene}.npz') as raw:
            mask = raw['mask']
            with np.load(root/'sharpened'/f'{scene}_noiseless_oracle_stages.npz') as sharp:
                target = sharp['sharpened'].mean(2)
            for method, values in case['variants'].items():
                with np.load(root/'sharpened'/f'{scene}_{method}_stages.npz') as sharp:
                    image = sharp['sharpened'].mean(2)
                error = image_errors(image, target, mask)
                values['sharpened'] = dict(rmse_linear_0_1=error['rmse_adu'],
                    gradient_vector_rmse_linear_per_px=error['gradient_vector_rmse_adu_per_px'])
                geometry = (raw['clean_'+method]-raw['noiseless_oracle'])[mask]
                noise = (raw[method]-raw['clean_'+method])[mask]
                gm, nm, cross = float(np.mean(geometry**2)), float(np.mean(noise**2)), float(2*np.mean(geometry*noise))
                np.testing.assert_allclose(np.mean((geometry+noise)**2), gm+nm+cross, atol=1e-12)
                values['raw_error_decomposition'] = dict(clean_geometry_mse_adu2=gm,
                    warped_noise_mse_adu2=nm, twice_cross_term_adu2=cross)
    (root/'analysis.json').write_text(json.dumps(motion, indent=2)+'\n')
    result['analytic_motion'] = dict(source=source(root/'analysis.json'), results=motion)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    print('Verified 384 established control frames, 48 translation probes, and 48 independent moving frames.')


if __name__ == '__main__':
    main()
