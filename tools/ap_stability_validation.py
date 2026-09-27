"""Fresh-noise validation of the selected field prior on shorter-scale motion."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from planetrecon.backends.torch_circular import sample
from planetrecon.pipeline.local_align import LocalRegistration
from tools.analytic_sampling_probe import components, detector_image
from tools.reference_imprint_probe import probes
from tools.regularized_ap_fit import RegularizedAPFit
from tools.ap_stability_screen import hashes
from tools.joint_saturn_experiment import digest
from tools.validate_local_warp_centring import shear_coordinates, save_png, image_errors


def integrated(features, shape, a, b, shift, wavelengths, order=4):
    nodes, weights = np.polynomial.legendre.leggauss(order)
    yy, xx = np.indices(shape, dtype=float)
    cy, cx = (np.array(shape)-1)/2
    image = np.zeros(shape)
    for dy, wy in zip(nodes/2, weights/2):
        for dx, wx in zip(nodes/2, weights/2):
            qx, qy = xx+dx-shift[0], yy+dy-shift[1]
            y = qy-b*np.sin(2*np.pi*(qx-cx)/wavelengths[1]+.4)
            x = qx-a*np.sin(2*np.pi*(y-cy)/wavelengths[0]+.5)
            value = np.full(shape, 3.)
            for amplitude, fx, fy, sx, sy in features:
                vx, vy = sx*sx+1., sy*sy+1.
                value += amplitude*sx*sy/np.sqrt(vx*vy)*np.exp(-.5*((x-fx)**2/vx+(y-fy)**2/vy))
            image += wx*wy*value
    return image


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    screen_path = Path('out/ap-stability-screen/report.json')
    screen = json.loads(screen_path.read_text())
    selected = screen['selection']['selected']
    if selected is None or screen['hashes'] != hashes():
        raise ValueError('No frozen selected candidate')
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    shape, count, seed = (128, 160), 32, 9033
    features = components('resolved', shape)
    reference = detector_image(features, shape, blur=1.)
    _, mask = probes(reference)
    names = ['baseline', selected]
    model = RegularizedAPFit(reference, device='cpu', policies=names)
    yy, xx = np.indices(shape)
    report = dict(selected=selected, hashes=hashes(), source_sha256=digest(__file__),
        screen_report_sha256=digest(screen_path), frames=count, seed=seed, device='cpu', cases={})
    for case, wavelengths in [('long_motion', (160., 240.)), ('short_motion', (80., 120.)), ('short_offset', (80., 120.))]:
        images, twins, errors, statistics = {}, {}, {}, {name: [] for name in names}
        oracle, quadrature_error = np.zeros(shape), 0.
        rng = np.random.default_rng(seed)
        for phase in 2*np.pi*np.arange(count)/count:
            a, b = 1.5*np.sin(phase), 1.2*np.cos(phase)
            if case == 'short_offset':
                a, b = a+.8, b-.5
            translation = (.35*np.sin(phase+.4), .4*np.cos(phase))
            clean = integrated(features, shape, a, b, translation, wavelengths)
            accurate = integrated(features, shape, a, b, translation, wavelengths, order=8)
            quadrature_error = max(quadrature_error, float(np.max(abs(clean-accurate))))
            truth = shear_coordinates(shape, a, b, wavelengths=wavelengths)-np.stack((xx, yy))+np.array(translation)[:, None, None]
            shift = translation if case == 'short_offset' else truth[:, mask].mean(1)
            frame = clean+rng.normal(0, 2., shape)
            observations = model.observations(LocalRegistration.proxy(frame), shift)
            oracle += sample(torch.tensor(clean), model.engine.yy+truth[1], model.engine.xx+truth[0]).numpy()/count
            for name in names:
                field, stats = model.field(name, observations, shift)
                statistics[name].append(stats)
                for source, accum in [(frame, images), (clean, twins)]:
                    value = sample(torch.tensor(source), model.engine.yy+field[1], model.engine.xx+field[0]).numpy()
                    accum.setdefault(name, np.zeros(shape))[:] += value/count
                delta = field.numpy()[:, mask]-truth[:, mask]
                errors[name] = errors.get(name, 0.)+np.mean(np.sum(delta*delta, axis=0))/count
        values = {}
        for name in names:
            values[name] = dict(image=image_errors(images[name], oracle, mask), clean_image=image_errors(twins[name], oracle, mask),
                field_rmse_px=float(np.sqrt(errors[name])), statistics=statistics[name])
            save_png(args.out/f'{case}_{name}.png', images[name])
        save_png(args.out/f'{case}_oracle.png', oracle)
        np.savez_compressed(args.out/f'{case}.npz', **images, mask=mask, noiseless_oracle=oracle,
                            **{'clean_'+key: value for key, value in twins.items()})
        report['cases'][case] = dict(wavelengths=wavelengths, quadrature_max_error_adu=quadrature_error, variants=values)
        print(case, {k: [v['field_rmse_px'], v['clean_image']['rmse_adu']] for k, v in values.items()}, flush=True)
    (args.out/'report.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
