"""Test whether unmodelled PSF elongation is mistaken for local motion.

Observations are independent integrals of continuous Gaussian features. Only
reference templates are blur-matched; original observed pixels are stacked.
The known-PSF branch is an oracle diagnostic, not a practical estimator.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.fft import rfft2, irfft2, fftfreq, rfftfreq
from scipy.ndimage import binary_erosion, map_coordinates
from scipy.optimize import least_squares
import torch

from planetrecon.backends.torch_circular import sample
from planetrecon.pipeline.circular_align import CircularMultiscaleRegistration
from tools.analytic_sampling_probe import components, detector_image
from tools.joint_saturn_experiment import digest
from tools.spline_joint_registration import SplineJointRegistration, spline_coefficients
from tools.validate_local_warp_centring import shear_coordinates, image_errors, save_png


def covariance(major, minor, angle):
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c, -s], [s, c]])
    return rotation @ np.diag([major**2, minor**2]) @ rotation.T


def integrated_scene(features, shape, psf, *, shift_xy=(0., 0.), a=0., b=0., order=4):
    """Pixel integrals of continuous elliptical blur followed by known shears."""
    psf = np.asarray(psf, dtype=float)
    if psf.shape != (2, 2) or not np.allclose(psf, psf.T) or np.linalg.eigvalsh(psf).min() < -1e-12:
        raise ValueError('PSF covariance must be positive semidefinite and symmetric')
    nodes, weights = np.polynomial.legendre.leggauss(order)
    y, x = np.indices(shape, dtype=float)
    cy, cx = (np.asarray(shape)-1)/2
    image = np.zeros(shape)
    for dy, wy in zip(nodes/2, weights/2):
        for dx, wx in zip(nodes/2, weights/2):
            qx, qy = x+dx-shift_xy[0], y+dy-shift_xy[1]
            yy = qy-b*np.sin(2*np.pi*(qx-cx)/240.+.4)
            xx = qx-a*np.sin(2*np.pi*(yy-cy)/160.+.5)
            value = np.full(shape, 3.)
            for amplitude, fx, fy, sx, sy in features:
                matrix = psf+np.diag([sx*sx, sy*sy])
                inverse = np.linalg.inv(matrix)
                rx, ry = xx-fx, yy-fy
                exponent = inverse[0, 0]*rx*rx+2*inverse[0, 1]*rx*ry+inverse[1, 1]*ry*ry
                value += amplitude*sx*sy/np.sqrt(np.linalg.det(matrix))*np.exp(-.5*exponent)
            image += wx*wy*value
    return image


class EllipticalReference:
    """Profile a global elliptical Gaussian using training detector pixels only."""
    def __init__(self, reference, padding=32):
        self.shape, self.padding = reference.shape, padding
        padded = np.pad(reference, padding, mode='reflect')
        self.spectrum = rfft2(padded)
        self.padded_shape = padded.shape
        self.fy = fftfreq(padded.shape[0])[:, None]
        self.fx = rfftfreq(padded.shape[1])[None, :]

    def image(self, psf):
        exponent = psf[0, 0]*self.fx**2+2*psf[0, 1]*self.fx*self.fy+psf[1, 1]*self.fy**2
        blurred = irfft2(self.spectrum*np.exp(-2*np.pi**2*exponent), s=self.padded_shape)
        p = self.padding
        return blurred[p:p+self.shape[0], p:p+self.shape[1]]

    def fit(self, frame, shift, training_mask):
        yy, xx = np.indices(self.shape, dtype=float)
        coords = np.array([yy[training_mask]-shift[1], xx[training_mask]-shift[0]])
        target = frame[training_mask]
        if len(target) < 64:
            raise ValueError('Insufficient training samples')

        def matrix(parameters):
            a, b, c = parameters
            lower = np.array([[a, 0.], [c, b]])
            return lower @ lower.T

        def residual(parameters):
            prediction = map_coordinates(self.image(matrix(parameters)), coords, order=3, mode='reflect')
            x, y = prediction-prediction.mean(), target-target.mean()
            gain = np.clip(np.mean(x*y)/max(np.mean(x*x), 1e-20), .5, 2.)
            return gain*x-y

        result = least_squares(residual, [1., 1., 0.], bounds=([0., 0., -3.], [3., 3., 3.]),
                               max_nfev=60, ftol=1e-8, xtol=1e-8, gtol=1e-7)
        psf = matrix(result.x)
        return self.image(psf), dict(covariance=psf.tolist(), success=bool(result.success),
            evaluations=int(result.nfev), training_mse=float(np.mean(result.fun**2)),
            parameters=result.x.tolist(), message=result.message)


def set_fixed_template(engine, reference):
    # Preserve AP geometry, field support, stiffness and all fitting samples.
    coefficients = spline_coefficients([reference])[0]
    engine.bank = torch.tensor(np.repeat(coefficients[None], len(engine.variances), axis=0),
                               device=engine.device, dtype=torch.float64)


def training_mask(engine, shift):
    mask = (sample(engine.mask, engine.yy-shift[1], engine.xx-shift[0]) > .5).cpu().numpy()
    train = np.zeros_like(mask)
    train[::2, ::2] = mask[::2, ::2]
    return train


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--frames', type=int, default=12)
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    if args.frames < 8:
        p.error('At least eight frames required')
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    shape = (128, 160)
    sources = ['tools/elliptical_blur_probe.py', 'tools/joint_registration.py',
               'tools/spline_joint_registration.py', 'tools/analytic_sampling_probe.py']
    report = dict(frames=args.frames, seed=9027, shape=shape, production_changed=False,
        output_filtering=False, normalization=False, hashes={p: digest(p) for p in sources}, scenes={})
    for kind in ('resolved', 'fine'):
        features = components(kind, shape)
        reference = detector_image(features, shape)
        engine = SplineJointRegistration(reference, CircularMultiscaleRegistration(reference), device=args.device)
        original_bank = engine.bank
        estimator = EllipticalReference(reference)
        mask = binary_erosion(reference > .08*reference.max(), iterations=3)
        mask[:8] = mask[-8:] = False
        mask[:, :8] = mask[:, -8:] = False
        yy, xx = np.indices(shape)
        report['scenes'][kind] = {}
        for case in ('circular_static_clean', 'elliptical_static_clean',
                     'elliptical_static_noise', 'elliptical_motion_noise'):
            images, twins, errors, statistics = {}, {}, {}, []
            rng = np.random.default_rng(report['seed'])
            quadrature_error = 0.
            for i, phase in enumerate(2*np.pi*np.arange(args.frames)/args.frames):
                psf = np.eye(2) if case.startswith('circular') else covariance(1.8, .5, phase)
                a, b = (1.5*np.sin(phase), 1.2*np.cos(phase)) if 'motion' in case else (0., 0.)
                translation = (.35*np.sin(phase+.4), .4*np.cos(phase))
                clean = integrated_scene(features, shape, psf, shift_xy=translation, a=a, b=b)
                fine = integrated_scene(features, shape, psf, shift_xy=translation, a=a, b=b, order=8)
                quadrature_error = max(quadrature_error, float(np.max(abs(clean-fine))))
                truth = shear_coordinates(shape, a, b)-np.stack((xx, yy))+np.array(translation)[:, None, None]
                shift = truth[:, mask].mean(1)
                noise = 2. if case.endswith('noise') else 0.
                frame = clean+rng.normal(0, noise, shape)
                original = frame.copy()
                estimated_reference, ellipse_stats = estimator.fit(frame, shift, training_mask(engine, shift))
                known_reference = integrated_scene(features, shape, psf, order=8)
                fields = dict(global_only=torch.tensor(shift, device=args.device)[:, None, None].expand(2, *shape),
                              oracle=torch.tensor(truth, device=args.device))
                stats = dict(ellipse=ellipse_stats, true_covariance=psf.tolist())
                for name, template in [('isotropic', None), ('ellipse_estimated', estimated_reference),
                                       ('ellipse_known', known_reference)]:
                    if template is None:
                        engine.bank = original_bank
                    else:
                        set_fixed_template(engine, template)
                    fields[name], stats[name] = engine.fit(frame, shift, .3, lambda: None)
                np.testing.assert_array_equal(frame, original)
                for name, field in fields.items():
                    for source, target in [(frame, images), (clean, twins)]:
                        warped = sample(torch.tensor(source, device=args.device),
                                        engine.yy+field[1], engine.xx+field[0]).cpu().numpy()
                        target.setdefault(name, np.zeros(shape))[:] += warped/args.frames
                    delta = field.cpu().numpy()[:, mask]-truth[:, mask]
                    value = float(np.mean(np.sum(delta*delta, axis=0)))
                    errors[name] = errors.get(name, 0.)+value/args.frames
                    stats.setdefault('field_rmse_px', {})[name] = np.sqrt(value)
                statistics.append(stats)
            target = twins['oracle']
            variants = {}
            prefix = f'{kind}_{case}'
            for name in images:
                variants[name] = dict(image=image_errors(images[name], target, mask),
                    clean_image=image_errors(twins[name], target, mask), field_rmse_px=float(np.sqrt(errors[name])))
                save_png(args.out/f'{prefix}_{name}.png', images[name])
            save_png(args.out/f'{prefix}_noiseless_oracle.png', target)
            np.savez_compressed(args.out/f'{prefix}.npz', **images, mask=mask, noiseless_oracle=target,
                                **{'clean_'+key: value for key, value in twins.items()})
            report['scenes'][kind][case] = dict(variants=variants, statistics=statistics,
                                               quadrature_max_error_adu=quadrature_error)
            (args.out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
            print(kind, case, {k: round(v['field_rmse_px'], 5) for k, v in variants.items()}, flush=True)


if __name__ == '__main__':
    main()
