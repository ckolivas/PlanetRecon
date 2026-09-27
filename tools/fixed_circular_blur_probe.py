"""Control for fixing PSF parameters before motion, independently of elongation."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import minimize_scalar
import torch

from planetrecon.backends.torch_circular import sample
from planetrecon.backends.torch_accel import TorchBackend
from planetrecon.io.ser import SERSource
from planetrecon.pipeline.circular_align import CircularMultiscaleRegistration
from planetrecon.export import ExportConfig, export_result
from planetrecon.result import load_snapshot, save_snapshot
from tools.analytic_sampling_probe import components, detector_image
from tools.elliptical_blur_probe import EllipticalReference, integrated_scene, covariance, training_mask, set_fixed_template
from tools.spline_joint_registration import SplineJointRegistration
from tools.validate_local_warp_centring import shear_coordinates, save_png, image_errors
from tools.coherent_saturn_experiment import inputs
from tools.joint_saturn_experiment import CAPTURE, digest
from tools.alignment_noise_experiment import read_png


def fit_circular(estimator, frame, shift, mask):
    yy, xx = np.indices(frame.shape, dtype=float)
    coords = np.array([yy[mask]-shift[1], xx[mask]-shift[0]])
    target = frame[mask]
    def objective(sigma):
        prediction = map_coordinates(estimator.image(np.eye(2)*sigma**2), coords, order=3, mode='reflect')
        x, y = prediction-prediction.mean(), target-target.mean()
        gain = np.clip(np.mean(x*y)/max(np.mean(x*x), 1e-20), .5, 2.)
        return float(np.mean((gain*x-y)**2))
    result = minimize_scalar(objective, bounds=(0., 3.), method='bounded', options={'xatol': 1e-6})
    candidates = [(float(result.fun), float(result.x)), (objective(0.), 0.), (objective(3.), 3.)]
    loss, sigma = min(candidates)
    return estimator.image(np.eye(2)*sigma**2), dict(sigma=sigma, training_mse=loss, success=bool(result.success))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    control_root = Path('out/elliptical-blur-controls')
    controls = json.loads((control_root/'report.json').read_text())
    real_root = Path('out/saturn-elliptical-pilot')
    real = json.loads((real_root/'report.json').read_text())
    for old in (controls, real):
        for path, value in old['hashes'].items():
            if path.startswith('tools/') and digest(path) != value:
                raise ValueError('Changed source: '+path)
    report = dict(hashes={'tools/fixed_circular_blur_probe.py': digest(__file__)}, control_report_sha256=digest(control_root/'report.json'),
                  pilot_report_sha256=digest(real_root/'report.json'), controls={}, real=[])
    shape = tuple(controls['shape'])
    yy, xx = np.indices(shape)
    for kind, cases in controls['scenes'].items():
        features = components(kind, shape)
        reference = detector_image(features, shape)
        engine = SplineJointRegistration(reference, CircularMultiscaleRegistration(reference))
        estimator = EllipticalReference(reference)
        for case, previous in cases.items():
            prefix = f'{kind}_{case}'
            with np.load(control_root/f'{prefix}.npz') as data:
                mask, target = data['mask'], data['noiseless_oracle']
            image, twin = np.zeros(shape), np.zeros(shape)
            field_mse, stats = 0., []
            rng = np.random.default_rng(controls['seed'])
            for phase in 2*np.pi*np.arange(controls['frames'])/controls['frames']:
                psf = np.eye(2) if case.startswith('circular') else covariance(1.8, .5, phase)
                a, b = (1.5*np.sin(phase), 1.2*np.cos(phase)) if 'motion' in case else (0., 0.)
                translation = (.35*np.sin(phase+.4), .4*np.cos(phase))
                clean = integrated_scene(features, shape, psf, shift_xy=translation, a=a, b=b)
                truth = shear_coordinates(shape, a, b)-np.stack((xx, yy))+np.array(translation)[:, None, None]
                shift = truth[:, mask].mean(1)
                frame = clean+rng.normal(0, 2. if case.endswith('noise') else 0., shape)
                matched, blur = fit_circular(estimator, frame, shift, training_mask(engine, shift))
                set_fixed_template(engine, matched)
                field, fit = engine.fit(frame, shift, .3, lambda: None)
                for source, accum in [(frame, image), (clean, twin)]:
                    accum += sample(torch.tensor(source, device=engine.device), engine.yy+field[1],
                                    engine.xx+field[0]).cpu().numpy()/controls['frames']
                field_mse += float(np.mean(np.sum((field.cpu().numpy()[:, mask]-truth[:, mask])**2, axis=0)))/controls['frames']
                stats.append(dict(blur=blur, fit=fit))
            report['controls'][prefix] = dict(statistics=stats, field_rmse_px=float(np.sqrt(field_mse)),
                image=image_errors(image, target, mask), clean_image=image_errors(twin, target, mask))
            save_png(args.out/f'{prefix}.png', image)
            np.savez_compressed(args.out/f'{prefix}.npz', image=image, clean=twin)
            print(prefix, 'field RMSE', np.sqrt(field_mse), flush=True)
    root, indices, quality, shifts, reference, hashes = inputs()
    for key, value in hashes.items():
        if value != real['hashes'][key]:
            raise ValueError('Real inputs changed')
    if digest(CAPTURE) != real['hashes']['capture']:
        raise ValueError('Capture changed')
    engine = SplineJointRegistration(reference, CircularMultiscaleRegistration(reference))
    estimator, backend = EllipticalReference(reference), TorchBackend()
    total, coverage = np.zeros_like(reference), np.zeros_like(reference)
    with SERSource(CAPTURE) as source:
        for n, pos in enumerate(real['positions']):
            frame = source.read_raw(int(indices[pos]))
            matched, blur = fit_circular(estimator, frame, shifts[pos], training_mask(engine, shifts[pos]))
            set_fixed_template(engine, matched)
            field, fit = engine.fit(frame, shifts[pos], .3, lambda: None)
            signal, support, *_ = backend.backproject(frame, field, 'mono')
            weight = max(float(quality[pos]), 1e-12)
            total += weight*signal
            coverage += weight*support
            report['real'].append(dict(position=int(pos), blur=blur, fit=fit))
            if (n+1) % 32 == 0:
                print('Saturn', n+1, 'complete', flush=True)
    result = load_snapshot(real_root/'isotropic.npz')
    result.image = np.divide(total, coverage, out=np.zeros_like(total), where=coverage>0)
    result.coverage, result.validity = coverage, coverage>0
    result.provenance = dict(result.provenance, policy='fixed_circular', source_sha256=digest(__file__))
    save_snapshot(args.out/'saturn.npz', result)
    _, text = read_png(root/'local.png')
    mapping = json.loads(text)['mapping']
    export_result(result, args.out/'saturn.png', ExportConfig('png16', mapping['black'], mapping['white'], 1.))
    (args.out/'report.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
