"""Screen fixed field constraints against known motion and reference sensitivity."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from planetrecon.backends.torch_circular import sample
from planetrecon.io.ser import SERSource
from planetrecon.pipeline.local_align import LocalRegistration
from tools.analytic_sampling_probe import components, detector_image
from tools.analytic_motion_probe import integrated_motion
from tools.reference_imprint_probe import probes, ReferenceStates, source_hashes
from tools.regularized_ap_fit import RegularizedAPFit, POLICIES
from tools.validate_local_warp_centring import shear_coordinates, save_png, image_errors
from tools.coherent_saturn_experiment import inputs
from tools.joint_saturn_experiment import CAPTURE, digest

PROBES = ['base', 'fine_0.25_plus', 'fine_0.25_minus', 'broad_0.25_plus', 'broad_0.25_minus']


def hashes():
    values = source_hashes()
    for path in ['tools/regularized_ap_fit.py', 'tools/ap_stability_screen.py', 'tools/local_warp_trace.py']:
        values[path] = digest(path)
    return values


def controls(out, device):
    shape, count, seed = (128, 160), 32, 9032
    features = components('resolved', shape)
    reference = detector_image(features, shape, blur=1.)
    _, mask = probes(reference)
    model = RegularizedAPFit(reference, device=device)
    engine = model.engine
    yy, xx = np.indices(shape)
    report = {}
    for case in ('stationary', 'moving', 'offset_moving'):
        images, twins, errors, stats = {}, {}, {}, {key: [] for key in POLICIES}
        oracle = np.zeros(shape)
        rng = np.random.default_rng(seed)
        for n, phase in enumerate(2*np.pi*np.arange(count)/count):
            a, b = (1.5*np.sin(phase), 1.2*np.cos(phase)) if case != 'stationary' else (0., 0.)
            if case == 'offset_moving':
                a, b = a+.8, b-.5
            translation = (.35*np.sin(phase+.4), .4*np.cos(phase))
            clean = integrated_motion(features, shape, a, b, 1., order=8, shift_xy=translation)
            truth = shear_coordinates(shape, a, b)-np.stack((xx, yy))+np.array(translation)[:, None, None]
            shift = np.array(translation) if case == 'offset_moving' else truth[:, mask].mean(1)
            frame = clean+rng.normal(0, 2., shape)
            proxy = LocalRegistration.proxy(frame)
            observations = model.observations(proxy, shift)
            oracle += sample(torch.tensor(clean, device=device), engine.yy+torch.tensor(truth[1], device=device),
                             engine.xx+torch.tensor(truth[0], device=device)).cpu().numpy()/count
            for name in POLICIES:
                field, statistic = model.field(name, observations, shift)
                if name == 'baseline' and n == 0:
                    torch.testing.assert_close(field, engine.displacement(proxy, shift, lambda: None), atol=1e-10, rtol=0)
                for source, accum in [(frame, images), (clean, twins)]:
                    value = sample(torch.tensor(source, device=device), engine.yy+field[1], engine.xx+field[0]).cpu().numpy()
                    accum.setdefault(name, np.zeros(shape))[:] += value/count
                delta = field.cpu().numpy()[:, mask]-truth[:, mask]
                errors[name] = errors.get(name, 0.)+np.mean(np.sum(delta*delta, axis=0))/count
                stats[name].append(statistic)
        values = {}
        for name in POLICIES:
            values[name] = dict(image=image_errors(images[name], oracle, mask), clean_image=image_errors(twins[name], oracle, mask),
                field_rmse_px=float(np.sqrt(errors[name])), guard_rejections=sum(not s['field_guard_accepted'] for s in stats[name]),
                insufficient_points=sum(s['fallback'] for s in stats[name]))
            save_png(out/f'{case}_{name}.png', images[name])
        save_png(out/f'{case}_oracle.png', oracle)
        np.savez_compressed(out/f'{case}.npz', **images, mask=mask, noiseless_oracle=oracle,
                            **{'clean_'+key: value for key, value in twins.items()})
        if case != 'offset_moving':
            with np.load(f'out/reference-imprint-control-coherent/{case}.npz') as prior:
                np.testing.assert_allclose(images['baseline'], prior['base'], atol=1e-10, rtol=0)
                np.testing.assert_allclose(twins['baseline'], prior['clean_base'], atol=1e-10, rtol=0)
        report[case] = values
        print(case, {key: round(v['field_rmse_px'], 5) for key, v in values.items()}, flush=True)
    return report


def stability(device):
    _, indices, _, shifts, reference, input_hashes = inputs()
    model = RegularizedAPFit(reference, device=device)
    patterns, mask = probes(reference)
    states = ReferenceStates(model.engine, reference, patterns)
    selected = np.linspace(0, len(indices)-1, 512, dtype=int)
    positions = selected[np.linspace(0, 511, 16, dtype=int)]
    rows = []
    with SERSource(CAPTURE) as source:
        for pos in positions:
            proxy = LocalRegistration.proxy(source.read_raw(int(indices[pos])))
            baseline_fields, row = {}, dict(position=int(pos), variants={})
            for probe in PROBES:
                states.select(probe)
                observations = model.observations(proxy, shifts[pos])
                row['variants'][probe] = {}
                for name in POLICIES:
                    field, statistic = model.field(name, observations, shifts[pos])
                    if probe == 'base':
                        baseline_fields[name] = field
                    else:
                        statistic['difference_rms_px'] = float((field-baseline_fields[name])[:, mask].square().sum(0).mean().sqrt())
                    row['variants'][probe][name] = statistic
            rows.append(row)
    summary = {}
    for name in POLICIES:
        summary[name] = {}
        for kind in ('fine', 'broad'):
            values = [r['variants'][f'{kind}_0.25_{sign}'][name]['difference_rms_px'] for r in rows for sign in ('plus', 'minus')]
            summary[name][kind] = dict(median_px=float(np.median(values)), maximum_px=float(np.max(values)))
    print('stability', summary, flush=True)
    return dict(positions=positions.tolist(), hashes=input_hashes, rows=rows, summary=summary)


def choose(control_results, sensitivity):
    """Predeclared gates: accuracy on every case, then >=50% stability gain."""
    result = {}
    for name in POLICIES:
        failures = []
        for case, values in control_results.items():
            actual, baseline = values[name], values['baseline']
            if actual['clean_image']['rmse_adu'] > 1.05*baseline['clean_image']['rmse_adu']+.002:
                failures.append(case+': clean-image accuracy')
            if actual['field_rmse_px'] > 1.05*baseline['field_rmse_px']+.005:
                failures.append(case+': motion accuracy')
            if actual['guard_rejections']+actual['insufficient_points'] > baseline['guard_rejections']+baseline['insufficient_points']:
                failures.append(case+': extra fallback')
        ratios = {kind: sensitivity[name][kind]['median_px']/sensitivity['baseline'][kind]['median_px'] for kind in ('fine', 'broad')}
        if max(ratios.values()) > .5:
            failures.append('less than 50 percent stability improvement')
        result[name] = dict(failures=failures, sensitivity_ratios=ratios)
    eligible = [name for name in POLICIES if name != 'baseline' and not result[name]['failures']]
    winner = min(eligible, key=lambda name: max(result[name]['sensitivity_ratios'].values())) if eligible else None
    return dict(criteria='Each clean RMSE <= baseline*1.05+0.002 ADU; field RMSE <= baseline*1.05+0.005 px; no extra control fallback; fine and broad sensitivity medians <= half baseline.',
                candidates=result, selected=winner)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    report = dict(hashes=hashes(), policies=POLICIES, frames_per_control=32, seed=9032,
        capture_sha256=digest(CAPTURE), production_changed=False, output_filtering=False, normalization=False)
    report['controls'] = controls(args.out, args.device)
    report['sensitivity'] = stability(args.device)
    if digest(CAPTURE) != report['capture_sha256']:
        raise ValueError('Capture changed')
    report['selection'] = choose(report['controls'], report['sensitivity']['summary'])
    (args.out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report['selection'], indent=2), flush=True)


if __name__ == '__main__':
    main()
