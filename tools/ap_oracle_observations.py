"""Separate measured AP displacements, gating, field fitting and support taper.

Truth observations are averages under the fitter's fixed gradient kernels.
They are an oracle for that linear observation model, not a claim that a
translation matcher on a deforming patch must return the same average.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from planetrecon.backends.torch_circular import sample
from planetrecon.pipeline.local_align import LocalRegistration
from tools.analytic_sampling_probe import components, detector_image
from tools.ap_stability_screen import hashes
from tools.ap_stability_validation import integrated
from tools.joint_saturn_experiment import digest
from tools.reference_imprint_probe import probes
from tools.regularized_ap_fit import RegularizedAPFit
from tools.validate_local_warp_centring import shear_coordinates, save_png, image_errors, weighted_field_errors

NAMES = ['measured', 'truth_gated', 'truth_accepted_uniform', 'truth_all', 'truth_untapered', 'oracle']
CASES = {'stationary': (160., 240.), 'long_motion': (160., 240.),
         'short_motion': (80., 120.), 'short_offset': (80., 120.)}


def source_hashes():
    return dict(hashes(), **{p: digest(p) for p in (
        'tools/ap_oracle_observations.py', 'tools/ap_stability_validation.py',
        'tools/analytic_sampling_probe.py', 'tools/validate_local_warp_centring.py')})


def patch_averages(engine, field):
    """Integrate a dense field over exactly the design matrix's reference patches."""
    field = np.asarray(field)
    if field.shape != (2, *engine.shape) or not np.isfinite(field).all():
        raise ValueError('Expected a finite two-component dense field')
    rows = []
    for layer, weights in zip(reversed(engine.layers), engine.kernels):
        h = layer['h']
        for x, y, k in zip(layer['x'].cpu().numpy(), layer['y'].cpu().numpy(), weights.cpu().numpy()):
            rows.append((field[:, y-h:y+h+1, x-h:x+h+1]*k).sum((-2, -1)))
    return np.asarray(rows).reshape(-1, 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    shape, count, seed = (128, 160), 32, 9033
    features = components('resolved', shape)
    reference = detector_image(features, shape, blur=1.)
    _, mask = probes(reference)
    model = RegularizedAPFit(reference, device=args.device, policies=['baseline'])
    engine = model.engine
    inner = mask & (engine.support >= 1.-1e-12)
    yy, xx = np.indices(shape)
    report = dict(hashes=source_hashes(), frames=count, seed=seed, device=args.device,
        variants=NAMES, shape=shape, ap_count=len(engine.points),
        coefficient_count=engine.fitter.ny*engine.fitter.nx,
        object_pixels=int(mask.sum()), untapered_object_pixels=int(inner.sum()), cases={})
    np.savez_compressed(args.out/'geometry.npz', reference=reference, mask=mask, inner=inner,
                        support=engine.support, points=engine.points)
    for case, wavelengths in CASES.items():
        rng = np.random.default_rng(seed)
        images, twins = {k: np.zeros(shape) for k in NAMES}, {k: np.zeros(shape) for k in NAMES}
        fields, statistics = {k: [] for k in NAMES}, {k: [] for k in NAMES}
        truth_fields, measured_rows, ideal_rows, confidences = [], [], [], []
        quadrature_error = 0.
        for index, phase in enumerate(2*np.pi*np.arange(count)/count):
            a, b = (0., 0.) if case == 'stationary' else (1.5*np.sin(phase), 1.2*np.cos(phase))
            if case == 'short_offset':
                a, b = a+.8, b-.5
            translation = (.35*np.sin(phase+.4), .4*np.cos(phase))
            clean = integrated(features, shape, a, b, translation, wavelengths)
            # All poses were checked in the preceding validation; repeat endpoints here.
            if index in (0, count-1):
                precise = integrated(features, shape, a, b, translation, wavelengths, order=8)
                quadrature_error = max(quadrature_error, float(np.max(abs(clean-precise))))
            truth = shear_coordinates(shape, a, b, wavelengths=wavelengths)-np.stack((xx, yy))+np.array(translation)[:, None, None]
            shift = np.asarray(translation) if case == 'short_offset' else truth[:, mask].mean(1)
            frame = clean+rng.normal(0, 2., shape)
            measured, confidence = model.observations(LocalRegistration.proxy(frame), shift)
            ideal = patch_averages(engine, truth-shift[:, None, None])
            measured_rows.append(measured)
            ideal_rows.append(ideal)
            confidences.append(confidence)
            truth_fields.append(truth)
            variants = {
                'measured': (measured, confidence), 'truth_gated': (ideal, confidence),
                'truth_accepted_uniform': (ideal, (confidence>0).astype(float)),
                'truth_all': (ideal, np.ones_like(confidence))}
            frame_tensor, clean_tensor = (torch.as_tensor(x, device=args.device) for x in (frame, clean))
            for name in NAMES:
                if name == 'oracle':
                    field = torch.as_tensor(truth, device=args.device)
                    stats = {}
                elif name == 'truth_untapered':
                    residual, stats = model.fitters['baseline'].fit(ideal, np.ones_like(confidence))
                    # Diagnostic only: omit both taper and guard, and expose this explicitly.
                    field = torch.as_tensor(residual+shift[:, None, None], device=args.device)
                    stats['taper_and_guard_bypassed'] = True
                else:
                    field, stats = model.field('baseline', variants[name], shift)
                if not bool(torch.isfinite(field).all()):
                    raise ValueError('Nonfinite oracle experiment field')
                fields[name].append(field.cpu().numpy())
                statistics[name].append(stats)
                for tensor, accum in ((frame_tensor, images), (clean_tensor, twins)):
                    accum[name] += sample(tensor, engine.yy+field[1], engine.xx+field[0]).cpu().numpy()/count
        target = twins['oracle']
        values = {}
        truth_fields = np.asarray(truth_fields)
        for name in NAMES:
            values[name] = dict(image=image_errors(images[name], target, mask),
                clean_image=image_errors(twins[name], target, mask),
                versus_noisy_oracle=image_errors(images[name], images['oracle'], mask),
                field=weighted_field_errors(np.asarray(fields[name]), truth_fields, np.ones(count)/count, mask),
                inner_field=weighted_field_errors(np.asarray(fields[name]), truth_fields, np.ones(count)/count, inner),
                statistics=statistics[name])
            save_png(args.out/f'{case}_{name}.png', images[name])
        save_png(args.out/f'{case}_target.png', target)
        if case != 'stationary':
            with np.load(f'out/ap-stability-validation/{case}.npz') as prior:
                np.testing.assert_allclose(images['measured'], prior['baseline'], atol=1e-9, rtol=0)
                np.testing.assert_allclose(twins['measured'], prior['clean_baseline'], atol=1e-9, rtol=0)
        path = args.out/f'{case}.npz'
        np.savez_compressed(path, **images, **{'clean_'+k: v for k, v in twins.items()},
            mask=mask, inner=inner, target=target,
            measured_observations=measured_rows, ideal_observations=ideal_rows, confidence=confidences,
            truth_fields=truth_fields, **{'field_'+k: np.asarray(v) for k, v in fields.items()})
        delta = np.asarray(measured_rows)-np.asarray(ideal_rows)
        confidence = np.asarray(confidences)
        norm2 = np.sum(delta*delta, axis=2)
        report['cases'][case] = dict(wavelengths=wavelengths, raw_sha256=digest(path),
            quadrature_endpoint_error_adu=quadrature_error, variants=values,
            accepted_ap_counts=np.count_nonzero(confidence, axis=1).tolist(),
            observation_vector_rmse_px=dict(all=float(np.sqrt(norm2.mean())),
                accepted=float(np.sqrt(norm2[confidence>0].mean())),
                weighted=float(np.sqrt(np.sum(norm2*confidence)/confidence.sum()))))
        print(case, {k: round(v['field']['total_vector_rmse_px'], 6) for k, v in values.items()}, flush=True)
    if report['hashes'] != source_hashes():
        raise ValueError('Sources changed during run')
    report['geometry_sha256'] = digest(args.out/'geometry.npz')
    (args.out/'report.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
