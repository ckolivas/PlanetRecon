"""Stack independently integrated continuous scenes with known local motion."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_erosion
import torch

from planetrecon.backends.torch_circular import sample
from planetrecon.pipeline.circular_align import CircularMultiscaleRegistration
from tools.analytic_sampling_probe import components, detector_image
from tools.joint_registration import JointRegistration
from tools.spline_joint_registration import SplineJointRegistration
from tools.validate_local_warp_centring import shear_coordinates, image_errors, save_png


def integrated_motion(features, shape, a, b, blur, *, order=4, shift_xy=(0., 0.)):
    """Gauss-Legendre pixel integration of analytically inverse-warped light.

    Continuous Gaussian convolution precedes the known invertible shear.
    Neither the reference pixels nor a spline/bicubic interpolator is used.
    """
    nodes, weights = np.polynomial.legendre.leggauss(order)
    nodes, weights = nodes/2, weights/2
    y, x = np.indices(shape, dtype=float)
    cy, cx = (np.asarray(shape)-1)/2
    result = np.zeros(shape)
    for dy, wy in zip(nodes, weights):
        for dx, wx in zip(nodes, weights):
            qx, qy = x+dx-shift_xy[0], y+dy-shift_xy[1]
            yy = qy-b*np.sin(2*np.pi*(qx-cx)/240.+.4)
            xx = qx-a*np.sin(2*np.pi*(yy-cy)/160.+.5)
            value = np.full(shape, 3.)
            for amplitude, fx, fy, sx, sy in features:
                vx, vy = sx*sx+blur*blur, sy*sy+blur*blur
                value += amplitude*sx*sy/np.sqrt(vx*vy)*np.exp(
                    -.5*((xx-fx)**2/vx+(yy-fy)**2/vy))
            result += wx*wy*value
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--frames', type=int, default=24)
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    if args.frames < 8:
        p.error('at least eight frames required')
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    shape = (128, 160)
    paths = ['tools/joint_registration.py', 'tools/spline_joint_registration.py',
             'tools/analytic_sampling_probe.py', 'tools/analytic_motion_probe.py']
    report = dict(frames=args.frames, seed=9021, shape=shape, quadrature_order=4,
        normalization=False, output_filtering=False, production_changed=False,
        hashes={p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}, scenes={})
    for kind in ('resolved', 'fine'):
        features = components(kind, shape)
        reference = detector_image(features, shape)
        mask = binary_erosion(reference > .08*reference.max(), iterations=3)
        mask[:8] = mask[-8:] = False
        mask[:, :8] = mask[:, -8:] = False
        engines = {name: cls(reference, CircularMultiscaleRegistration(reference), device=args.device)
                   for name, cls in [('bicubic', JointRegistration), ('spline', SplineJointRegistration)]}
        engine = engines['spline']
        images, twins, errors, statistics = {}, {}, {}, []
        rng = np.random.default_rng(report['seed'])
        yy, xx = np.indices(shape)
        quadrature_errors = []
        for i, phase in enumerate(2*np.pi*np.arange(args.frames)/args.frames):
            a, b, blur = 1.5*np.sin(phase), 1.2*np.cos(phase), .75*(1+np.sin(3*phase+.3))
            clean = integrated_motion(features, shape, a, b, blur)
            # Check independent integration accuracy at every tested pose.
            fine = integrated_motion(features, shape, a, b, blur, order=8)
            quadrature_errors.append(float(np.max(abs(clean-fine))))
            truth = shear_coordinates(shape, a, b)-np.stack((xx, yy))
            shift = truth[:, mask].mean(1)
            origin = torch.tensor(shift, device=args.device)[:, None, None]
            frame = clean+rng.normal(0, 2, shape)
            fields = dict(global_only=origin.expand(2, *shape), oracle=torch.tensor(truth, device=args.device))
            stats = {}
            for name, fitter in engines.items():
                field, stats[name] = fitter.fit(frame, shift, .3, lambda: None)
                fields[name] = field
                fields[name+'_validated'] = field if stats[name].get('heldout_improves') else fields['global_only']
            for name, field in fields.items():
                for source, target in [(frame, images), (clean, twins)]:
                    warped = sample(torch.tensor(source, device=args.device),
                                    engine.yy+field[1], engine.xx+field[0]).cpu().numpy()
                    target.setdefault(name, np.zeros(shape))[:] += warped/args.frames
                delta = field.cpu().numpy()[:, mask]-truth[:, mask]
                errors[name] = errors.get(name, 0.)+np.mean(np.sum(delta*delta, axis=0))/args.frames
            statistics.append(stats)
            if (i+1) % 8 == 0:
                print(kind, i+1, 'frames complete', flush=True)
        oracle = twins['oracle']
        values = {}
        for name in images:
            values[name] = dict(image=image_errors(images[name], oracle, mask),
                clean_image=image_errors(twins[name], oracle, mask), field_vector_rmse_px=float(np.sqrt(errors[name])))
            save_png(args.out/f'{kind}_{name}.png', images[name])
        save_png(args.out/f'{kind}_noiseless_oracle.png', oracle)
        np.savez_compressed(args.out/f'{kind}.npz', **images, mask=mask, noiseless_oracle=oracle,
                            **{'clean_'+key: value for key, value in twins.items()})
        report['scenes'][kind] = dict(statistics=statistics, variants=values,
            quadrature_max_difference_adu=max(quadrature_errors))
        (args.out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
        print(kind, {k: round(v['clean_image']['rmse_adu'], 6) for k, v in values.items()}, flush=True)


if __name__ == '__main__':
    main()
