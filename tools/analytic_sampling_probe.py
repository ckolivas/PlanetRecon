"""Independent continuous-scene test of the joint forward-sampling models.

Each detector pixel is the exact area integral of a sum of continuous Gaussian
features after a known Gaussian PSF and translation. No image interpolation or
fitter renderer generates the observations. Widths describe the scene features,
not a measured telescope PSF. The reference has no added noise.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_erosion
from scipy.special import erf
import torch

from planetrecon.backends.torch_circular import sample
from planetrecon.pipeline.circular_align import CircularMultiscaleRegistration
from tools.joint_registration import JointRegistration
from tools.spline_joint_registration import SplineJointRegistration
from tools.validate_local_warp_centring import image_errors


def components(kind, shape=(128, 160)):
    """Fixed continuous scenes with well sampled or subpixel-width features."""
    if kind not in ('resolved', 'fine'):
        raise ValueError(kind)
    cy, cx = (np.asarray(shape)-1)/2
    result = [(115., cx, cy, 38., 25.)]
    rng = np.random.default_rng(103)
    low, high = (1.5, 3.) if kind == 'resolved' else (.65, 1.2)
    for _ in range(60):
        result.append((rng.uniform(10, 30), cx+rng.uniform(-42, 42),
                       cy+rng.uniform(-28, 28), rng.uniform(low, high), rng.uniform(low, high)))
    return np.asarray(result)


def detector_image(features, shape, shift_xy=(0., 0.), blur=0.):
    """Exact unit-square pixel integrals, with flux-preserving Gaussian blur."""
    y, x = np.indices(shape, dtype=float)
    x -= shift_xy[0]
    y -= shift_xy[1]
    image = np.full(shape, 3.)
    for amplitude, cx, cy, sx, sy in features:
        wx, wy = np.sqrt(sx*sx+blur*blur), np.sqrt(sy*sy+blur*blur)
        # Integral of exp(-x^2 / 2 w^2) across a pixel. The sx/sy factors
        # also account for the amplitude change under normalized convolution.
        ix = sx*np.sqrt(np.pi/2)*(erf((x+.5-cx)/(np.sqrt(2)*wx))-
                                  erf((x-.5-cx)/(np.sqrt(2)*wx)))
        iy = sy*np.sqrt(np.pi/2)*(erf((y+.5-cy)/(np.sqrt(2)*wy))-
                                  erf((y-.5-cy)/(np.sqrt(2)*wy)))
        image += amplitude*ix*iy
    return image


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--noise', type=float, nargs='+', default=[0., 2.])
    p.add_argument('--samplers', nargs='+', choices=['bicubic', 'spline'], default=['bicubic', 'spline'])
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--iterations', type=int, default=100)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    torch.set_num_threads(4)
    sources = ['tools/joint_registration.py', 'tools/spline_joint_registration.py', __file__]
    report = dict(generator='exact continuous Gaussian pixel integrals', shape=[128, 160],
        seed=9020, iterations=args.iterations, samplers=args.samplers,
        hashes={str(path): hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in sources},
        frames=[])
    for kind in ('resolved', 'fine'):
        features = components(kind)
        reference = detector_image(features, tuple(report['shape']))
        mask = binary_erosion(reference > .08*reference.max(), iterations=3)
        mask[:8] = mask[-8:] = False
        mask[:, :8] = mask[:, -8:] = False
        engines = {name: cls(reference, CircularMultiscaleRegistration(reference),
                             device=args.device, maxiter=args.iterations)
                   for name, cls in [('bicubic', JointRegistration), ('spline', SplineJointRegistration)]
                   if name in args.samplers}
        for shift in ((.25, .5), (.6, -.4), (-.7, .35)):
            for blur in (0., 1.):
                clean = detector_image(features, reference.shape, shift, blur)
                for noise in args.noise:
                    observed = clean+np.random.default_rng(report['seed']).normal(0, noise, clean.shape)
                    row = dict(scene=kind, shift_xy=shift, blur=blur, noise=noise,
                               reference_sha256=hashlib.sha256(reference.tobytes()).hexdigest(),
                               observation_sha256=hashlib.sha256(observed.tobytes()).hexdigest(), variants={})
                    for name, engine in engines.items():
                        field, stats = engine.fit(observed, shift, .3, lambda: None)
                        origin = torch.tensor(shift, device=args.device, dtype=torch.float64)[:, None, None]
                        pixels = torch.tensor(clean, device=args.device)
                        oracle = sample(pixels, engine.yy+origin[1], engine.xx+origin[0]).cpu().numpy()
                        warped = sample(pixels, engine.yy+field[1], engine.xx+field[0]).cpu().numpy()
                        row['variants'][name] = dict(stats=stats, clean_image=image_errors(warped, oracle, mask),
                            false_motion_rms_px=float((field-origin)[:, mask].square().sum(0).mean().sqrt()))
                    report['frames'].append(row)
                print(kind, shift, blur, 'complete', flush=True)
                args.out.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
