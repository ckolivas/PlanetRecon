"""Paired reference-only perturbations with fixed AP geometry and observations."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_erosion, gaussian_filter
import torch

from planetrecon.backends.torch_accel import TorchBackend
from planetrecon.backends.torch_circular import TorchCircularRegistration, sample
from planetrecon.pipeline.circular_align import CircularMultiscaleRegistration
from planetrecon.pipeline.local_align import LocalRegistration
from planetrecon.io.ser import SERSource
from planetrecon.export import ExportConfig, export_result
from planetrecon.result import load_snapshot, save_snapshot
from tools.coherent_registration import CoherentRegistration
from tools.coherent_saturn_experiment import inputs
from tools.joint_saturn_experiment import CAPTURE, digest
from tools.analytic_sampling_probe import components, detector_image
from tools.analytic_motion_probe import integrated_motion
from tools.validate_local_warp_centring import shear_coordinates, save_png, image_errors
from tools.alignment_noise_experiment import read_png

DOSES = (.25, 1.)
SCALES = {'fine': 1., 'broad': 4.}


def probes(reference, seed=9031):
    mask = binary_erosion(reference > .08*reference.max(), iterations=3)
    mask[:8] = mask[-8:] = False
    mask[:, :8] = mask[:, -8:] = False
    taper = gaussian_filter((reference > .04*reference.max()).astype(float), 8.)
    rng = np.random.default_rng(seed)
    patterns = {}
    for name, sigma in SCALES.items():
        noise = rng.normal(size=reference.shape)
        pattern = gaussian_filter(noise, sigma)-gaussian_filter(noise, 3*sigma)
        pattern -= np.mean((pattern*taper)[mask])/np.mean(taper[mask])
        pattern *= taper
        pattern /= np.sqrt(np.mean(pattern[mask]**2))
        patterns[name] = pattern
    return patterns, mask


def variants():
    result = {'base': (None, 0.)}
    for name in SCALES:
        for dose in DOSES:
            for sign, value in [('plus', dose), ('minus', -dose)]:
                result[f'{name}_{dose:g}_{sign}'] = (name, value)
    return result


class ReferenceStates:
    """Change matching templates only; retain eligibility, geometry and kernels."""
    def __init__(self, engine, reference, patterns):
        self.engine = engine
        self.states = {'base': (engine.reference, [(v['templates'], v['strength']) for v in engine.layers])}
        for key, (name, dose) in variants().items():
            if name is None:
                continue
            proxy = torch.tensor(LocalRegistration.proxy(reference+dose*patterns[name]), device=engine.device)
            patches = []
            for layer in engine.layers:
                patch = proxy[layer['y'][:, None, None]+layer['py'], layer['x'][:, None, None]+layer['px']]
                patch = patch-(patch*layer['weight']).sum((-2, -1))[:, None, None]
                strength = (patch.square()*layer['weight']).sum((-2, -1)).sqrt()
                patches.append((patch, strength))
            self.states[key] = proxy, patches

    def select(self, key):
        self.engine.reference, patches = self.states[key]
        for layer, (template, strength) in zip(self.engine.layers, patches, strict=True):
            layer['templates'], layer['strength'] = template, strength


def metrics(response, pattern, mask):
    a, b = response[mask], pattern[mask]
    ac, bc = a-a.mean(), b-b.mean()
    correlation = float(np.corrcoef(a, b)[0, 1]) if np.std(a)>1e-15 else None
    return dict(rms=float(np.sqrt(np.mean(a*a))),
                pattern_gain=float(ac@bc/(bc@bc)), correlation=correlation)


def response_summary(images, patterns, mask):
    result = {}
    for name, pattern in patterns.items():
        for dose in DOSES:
            plus, minus = [images[f'{name}_{dose:g}_{s}'] for s in ('plus', 'minus')]
            response = (plus-minus)/(2*dose)
            even = (plus+minus)*.5-images['base']
            result[f'{name}_{dose:g}'] = dict(response_per_reference_adu=metrics(response, pattern, mask),
                even_response_rms_adu=float(np.sqrt(np.mean(even[mask]**2))))
    return result


def engine_for(reference, method, device):
    matcher = CircularMultiscaleRegistration(reference)
    if method == 'production':
        return TorchCircularRegistration(matcher, device=device)
    return CoherentRegistration(matcher, device=device, spacing=16., stiffness=.01, patch_average=True)


def source_hashes():
    return {p: digest(p) for p in ['tools/reference_imprint_probe.py', 'tools/coherent_registration.py',
        'planetrecon/backends/torch_circular.py', 'planetrecon/backends/triton_correlation.py',
        'planetrecon/pipeline/local_align.py',
        'tools/analytic_motion_probe.py', 'tools/analytic_sampling_probe.py']}


def run_controls(out, method, device):
    shape, count, seed = (128, 160), 32, 9032
    features = components('resolved', shape)
    reference = detector_image(features, shape, blur=1.)
    patterns, mask = probes(reference)
    engine = engine_for(reference, method, device)
    states = ReferenceStates(engine, reference, patterns)
    yy, xx = np.indices(shape)
    report = dict(method=method, frames=count, seed=seed, probe_seed=9031, scales=SCALES, doses=DOSES,
                  hashes=source_hashes(), cases={},
                  reference_sha256=hashlib.sha256(reference.tobytes()).hexdigest())
    np.savez_compressed(out/'probe.npz', reference=reference, mask=mask, **patterns)
    for case in ('stationary', 'moving'):
        images, twins, field_mse = {}, {}, {}
        oracle = np.zeros(shape)
        rng = np.random.default_rng(seed)
        for n, phase in enumerate(2*np.pi*np.arange(count)/count):
            a, b = (1.5*np.sin(phase), 1.2*np.cos(phase)) if case == 'moving' else (0., 0.)
            translation = (.35*np.sin(phase+.4), .4*np.cos(phase))
            clean = integrated_motion(features, shape, a, b, 1., order=8, shift_xy=translation)
            truth = shear_coordinates(shape, a, b)-np.stack((xx, yy))+np.array(translation)[:, None, None]
            shift = truth[:, mask].mean(1)
            frame = clean+rng.normal(0, 2., shape)
            frame_before = frame.copy()
            proxy = LocalRegistration.proxy(frame)
            oracle += sample(torch.tensor(clean, device=device), engine.yy+torch.tensor(truth[1], device=device),
                             engine.xx+torch.tensor(truth[0], device=device)).cpu().numpy()/count
            for key in variants():
                states.select(key)
                field = engine.displacement(proxy, shift, lambda: None)
                for source, accum in [(frame, images), (clean, twins)]:
                    value = sample(torch.tensor(source, device=device), engine.yy+field[1], engine.xx+field[0]).cpu().numpy()
                    accum.setdefault(key, np.zeros(shape))[:] += value/count
                delta = field.cpu().numpy()[:, mask]-truth[:, mask]
                field_mse[key] = field_mse.get(key, 0.)+np.mean(np.sum(delta*delta, axis=0))/count
            np.testing.assert_array_equal(frame, frame_before)
        values = {}
        for key in variants():
            values[key] = dict(image=image_errors(images[key], oracle, mask),
                clean_image=image_errors(twins[key], oracle, mask), field_rmse_px=float(np.sqrt(field_mse[key])))
            save_png(out/f'{case}_{key}.png', images[key])
        save_png(out/f'{case}_oracle.png', oracle)
        np.savez_compressed(out/f'{case}.npz', **images, noiseless_oracle=oracle,
                            **{'clean_'+key: value for key, value in twins.items()})
        report['cases'][case] = dict(variants=values, response=response_summary(images, patterns, mask),
                                    clean_response=response_summary(twins, patterns, mask))
        print(method, case, report['cases'][case]['response'], flush=True)
    (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')


def run_capture(out, method, device, count):
    root, indices, quality, shifts, reference, hashes = inputs()
    positions = np.linspace(0, len(indices)-1, count, dtype=int)
    patterns, mask = probes(reference)
    engine = engine_for(reference, method, device)
    states = ReferenceStates(engine, reference, patterns)
    backend = TorchBackend()
    original = load_snapshot(root/'local.npz')
    _, text = read_png(root/'local.png')
    mapping = json.loads(text)['mapping']
    export = ExportConfig('png16', mapping['black'], mapping['white'], 1.)
    hashes.update(source_hashes())
    hashes['capture'] = digest(CAPTURE)
    report = dict(method=method, frames=count, probe_seed=9031, scales=SCALES, doses=DOSES,
        positions=positions.tolist(), frame_indices=indices[positions].tolist(),
        hashes=hashes, output_filtering=False, normalization=False, production_changed=False, checkpoints={})
    keys = list(variants())
    total, support = np.zeros((len(keys), 2, *reference.shape)), np.zeros((len(keys), 2, *reference.shape))
    np.savez_compressed(out/'probe.npz', reference=reference, mask=mask, **patterns)
    with SERSource(CAPTURE) as source:
        for n, pos in enumerate(positions):
            frame = source.read_raw(int(indices[pos]))
            proxy = LocalRegistration.proxy(frame)
            weight = max(float(quality[pos]), 1e-12)
            for k, key in enumerate(keys):
                states.select(key)
                field = engine.displacement(proxy, shifts[pos], lambda: None)
                signal, coverage, *_ = backend.backproject(frame, field, 'mono')
                total[k, n % 2] += weight*signal
                support[k, n % 2] += weight*coverage
            if n+1 in (64, 128, 256, count):
                sums, weights = total.sum(1), support.sum(1)
                images = np.divide(sums, weights, out=np.zeros_like(sums), where=weights>0)
                report['checkpoints'][str(n+1)] = response_summary(dict(zip(keys, images)), patterns, mask)
                print(method, n+1, 'of', count, flush=True)
    if digest(CAPTURE) != hashes['capture']:
        raise ValueError('Capture changed')
    np.savez_compressed(out/'accumulators.npz', signal=total, support=support, keys=np.array(keys))
    halves = np.divide(total, support, out=np.zeros_like(total), where=support>0)
    report['halves'] = [response_summary(dict(zip(keys, halves[:, half])), patterns, mask) for half in (0, 1)]
    for k, key in enumerate(keys):
        result = replace(original, image=images[k], coverage=weights[k], validity=weights[k]>0, n_used=count,
            provenance=dict(diagnostic_only=True, method=method, reference_variant=key, hashes=hashes,
                            frame_indices=indices[positions].tolist()))
        save_snapshot(out/f'{key}.npz', result)
        export_result(result, out/f'{key}.png', export)
    report['accumulators_sha256'] = digest(out/'accumulators.npz')
    (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--method', choices=['production', 'coherent'], required=True)
    p.add_argument('--capture', action='store_true')
    p.add_argument('--frames', type=int, default=512)
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    if not 64 <= args.frames <= 5738:
        p.error('Frame count must be between 64 and 5738')
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    if args.capture:
        run_capture(args.out, args.method, args.device, args.frames)
    else:
        run_controls(args.out, args.method, args.device)


if __name__ == '__main__':
    main()
