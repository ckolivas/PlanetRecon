"""Repeat AP matching without noise and isolate matching/noise coupling.

Paired same/cross stacks contain the same source samples and estimated fields;
only which independent noise realization supplies the field is exchanged.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from planetrecon.backends.torch_circular import sample
from planetrecon.pipeline.local_align import LocalRegistration
from tools.ap_oracle_observations import source_hashes as prior_hashes, CASES
from tools.ap_stability_validation import integrated
from tools.analytic_sampling_probe import components
from tools.regularized_ap_fit import RegularizedAPFit
from tools.joint_saturn_experiment import digest
from tools.validate_local_warp_centring import save_png, image_errors, weighted_field_errors


def hashes():
    return dict(prior_hashes(), **{'tools/ap_noise_coupling.py': digest(__file__)})


def pair_stacks(aa, ab, ba, bb):
    """First index is field provenance, second is input-noise provenance."""
    return .5*(aa+bb), .5*(ab+ba)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    prior = Path('out/ap-oracle-observations')
    prior_report = json.loads((prior/'report.json').read_text())
    if prior_report['hashes'] != prior_hashes() or prior_report['geometry_sha256'] != digest(prior/'geometry.npz'):
        raise ValueError('Prior experiment changed')
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    with np.load(prior/'geometry.npz') as data:
        reference, mask, inner = data['reference'], data['mask'], data['inner']
    shape, count, seed_a, seed_b = reference.shape, 32, 9033, 19033
    model = RegularizedAPFit(reference, device=args.device, policies=['baseline'])
    engine = model.engine
    features = components('resolved', shape)
    report = dict(hashes=hashes(), prior_report_sha256=digest(prior/'report.json'),
        geometry_sha256=digest(prior/'geometry.npz'), count=count, seeds=[seed_a, seed_b], noise_sigma_adu=2.,
        device=args.device, cases={})
    for case, wavelengths in CASES.items():
        path = prior/f'{case}.npz'
        if digest(path) != prior_report['cases'][case]['raw_sha256']:
            raise ValueError('Prior arrays changed')
        with np.load(path) as data:
            truths, ideal = data['truth_fields'], data['ideal_observations']
            previous_fields, previous_observations, previous_confidence = data['field_measured'], data['measured_observations'], data['confidence']
            previous_stack, previous_clean, previous_oracle = data['measured'], data['clean_measured'], data['target']
        rng_a, rng_b = np.random.default_rng(seed_a), np.random.default_rng(seed_b)
        sums = {(field, source): np.zeros(shape) for field in ('clean', 'a', 'b', 'truth') for source in ('clean', 'a', 'b')}
        fields, observations, confidence = ({k: [] for k in ('clean', 'a', 'b')} for _ in range(3))
        stats = {k: [] for k in ('clean', 'b')}
        for index, phase in enumerate(2*np.pi*np.arange(count)/count):
            a, b = (0., 0.) if case == 'stationary' else (1.5*np.sin(phase), 1.2*np.cos(phase))
            if case == 'short_offset':
                a, b = a+.8, b-.5
            translation = (.35*np.sin(phase+.4), .4*np.cos(phase))
            clean = integrated(features, shape, a, b, translation, wavelengths)
            frame_a = clean+rng_a.normal(0, 2., shape)
            frame_b = clean+rng_b.normal(0, 2., shape)
            shift = np.asarray(translation) if case == 'short_offset' else truths[index][:, mask].mean(1)
            frame_fields = dict(a=torch.as_tensor(previous_fields[index], device=args.device),
                                truth=torch.as_tensor(truths[index], device=args.device))
            observations['a'].append(previous_observations[index])
            confidence['a'].append(previous_confidence[index])
            for name, pixels in [('clean', clean), ('b', frame_b)]:
                observation = model.observations(LocalRegistration.proxy(pixels), shift)
                field, statistic = model.field('baseline', observation, shift)
                frame_fields[name] = field
                observations[name].append(observation[0])
                confidence[name].append(observation[1])
                stats[name].append(statistic)
            sources = {k: torch.as_tensor(v, device=args.device) for k, v in [('clean', clean), ('a', frame_a), ('b', frame_b)]}
            for name, field in frame_fields.items():
                if name != 'truth':
                    fields[name].append(field.cpu().numpy())
                for source, tensor in sources.items():
                    sums[name, source] += sample(tensor, engine.yy+field[1], engine.xx+field[0]).cpu().numpy()/count
        np.testing.assert_allclose(sums['a', 'a'], previous_stack, rtol=0, atol=1e-9)
        np.testing.assert_allclose(sums['a', 'clean'], previous_clean, rtol=0, atol=1e-9)
        np.testing.assert_allclose(sums['truth', 'clean'], previous_oracle, rtol=0, atol=1e-9)
        paired_same, paired_cross = pair_stacks(sums['a', 'a'], sums['a', 'b'], sums['b', 'a'], sums['b', 'b'])
        images = dict(no_noise_match=sums['clean', 'a'], noisy_match=sums['a', 'a'],
            independent_match=sums['b', 'a'], oracle=sums['truth', 'a'],
            paired_same=paired_same, paired_cross=paired_cross,
            paired_no_noise_match=.5*(sums['clean', 'a']+sums['clean', 'b']),
            paired_oracle=.5*(sums['truth', 'a']+sums['truth', 'b']), target=previous_oracle)
        for name, pixels in images.items():
            save_png(args.out/f'{case}_{name}.png', pixels)
        common = (np.asarray(confidence['clean'])>0)&(np.asarray(confidence['a'])>0)&(np.asarray(confidence['b'])>0)
        values = {}
        for name in fields:
            obs, conf = np.asarray(observations[name]), np.asarray(confidence[name])
            error = np.sum((obs-ideal)**2, axis=2)
            values[name] = dict(field=weighted_field_errors(np.asarray(fields[name]), truths, np.ones(count)/count, mask),
                inner_field=weighted_field_errors(np.asarray(fields[name]), truths, np.ones(count)/count, inner),
                clean_stack=image_errors(sums[name, 'clean'], previous_oracle, mask),
                observation_vector_rmse_px=dict(all=float(np.sqrt(error.mean())),
                    accepted=float(np.sqrt(error[conf>0].mean())), common=float(np.sqrt(error[common].mean()))),
                accepted_ap_counts=np.count_nonzero(conf, axis=1).tolist())
        artifact = args.out/f'{case}.npz'
        np.savez_compressed(artifact, **images, mask=mask, inner=inner, truth=truths, ideal=ideal,
            **{'field_'+k: np.asarray(v) for k, v in fields.items()},
            **{'observations_'+k: np.asarray(v) for k, v in observations.items()},
            **{'confidence_'+k: np.asarray(v) for k, v in confidence.items()},
            **{'clean_'+k: sums[k, 'clean'] for k in fields},
            pair_clean_same=.5*(sums['a', 'clean']+sums['b', 'clean']),
            pair_clean_cross=.5*(sums['a', 'clean']+sums['b', 'clean']))
        report['cases'][case] = dict(prior_raw_sha256=digest(path), raw_sha256=digest(artifact), matching=values,
            common_accepted_ap_counts=common.sum(1).tolist(), statistics=stats,
            image_errors={k: image_errors(v, previous_oracle, mask) for k, v in images.items()},
            paired_same_minus_cross=image_errors(paired_same, paired_cross, mask),
            prior_noisy_stack_max_difference_adu=float(np.max(abs(sums['a', 'a']-previous_stack))))
        print(case, {k: dict(field=v['field']['total_vector_rmse_px'], ap=v['observation_vector_rmse_px']['common'],
                            clean=v['clean_stack']['rmse_adu']) for k, v in values.items()}, flush=True)
    if report['hashes'] != hashes():
        raise ValueError('Sources changed during replay')
    (args.out/'report.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
