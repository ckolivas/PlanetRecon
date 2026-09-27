"""Test mean-warp subtraction against known invertible motion and independent noise.

This is an experiment, not a change to production registration. Coordinates are
composed shears with an analytic inverse; frame generation uses cubic sampling,
whereas every reconstruction uses production's bilinear sampler.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.ndimage import binary_erosion, gaussian_filter, map_coordinates
import torch

from planetrecon.backends.torch_circular import TorchCircularRegistration, sample
from planetrecon.pipeline.circular_align import CircularMultiscaleRegistration
from planetrecon.pipeline.local_align import LocalRegistration


def shear_coordinates(shape, a, b, inverse=False, wavelengths=(160., 240.)):
    """Return x,y coordinates, applying the inverse in reverse composition order."""
    y, x = np.indices(shape, dtype=float)
    cy, cx = (np.asarray(shape)-1)/2
    if inverse:
        yy = y-b*np.sin(2*np.pi*(x-cx)/wavelengths[1]+.4)
        xx = x-a*np.sin(2*np.pi*(yy-cy)/wavelengths[0]+.5)
    else:
        xx = x+a*np.sin(2*np.pi*(y-cy)/wavelengths[0]+.5)
        yy = y+b*np.sin(2*np.pi*(xx-cx)/wavelengths[1]+.4)
    return np.stack((xx, yy))


def make_frame(reference, a, b, blur, wavelengths=(160., 240.)):
    blurred = gaussian_filter(reference, blur) if blur else reference
    inverse = shear_coordinates(reference.shape, a, b, inverse=True, wavelengths=wavelengths)
    frame = map_coordinates(blurred, inverse[::-1], order=3, mode='reflect')
    y, x = np.indices(reference.shape)
    field = shear_coordinates(reference.shape, a, b, wavelengths=wavelengths)-np.stack((x, y))
    return frame, field, blurred


def weighted_field_errors(estimated, truth, weights, mask):
    """Separate fixed reference-coordinate bias from errors varying across frames."""
    error = estimated[..., mask]-truth[..., mask]
    mean_error = np.einsum('n,nap->ap', weights, error)
    mean_estimated = np.einsum('n,nap->ap', weights, estimated[..., mask])
    mean_truth = np.einsum('n,nap->ap', weights, truth[..., mask])
    def rms(a):
        return float(np.sqrt(np.mean(np.sum(a*a, axis=0))))
    return {
        'total_vector_rmse_px': float(np.sqrt(np.einsum('n,nap->', weights, error**2)/mask.sum())),
        'mean_error_vector_rms_px': rms(mean_error),
        'temporal_error_vector_rms_px': float(np.sqrt(np.einsum(
            'n,nap->', weights, (error-mean_error)**2)/mask.sum())),
        'estimated_mean_vector_rms_px': rms(mean_estimated),
        'true_mean_vector_rms_px': rms(mean_truth),
    }


def cases(n):
    phase = 2*np.pi*np.arange(n)/n
    a, b = 1.5*np.sin(phase), 1.2*np.cos(phase)
    blur = .75*(1+np.sin(3*phase+.3))
    ones, zeros = np.ones(n), np.zeros(n)
    return {
        'static_noise': (zeros, zeros, zeros, 2., ones),
        'static_blur_noise': (zeros, zeros, blur, 2., ones),
        'motion_clean': (a, b, zeros, 0., ones),
        'motion_blur_noise': (a, b, blur, 2., ones),
        'motion_correlated_blur': (a, b, .75*(1+np.sin(phase)), 2., ones),
        'offset_motion_blur': (a+.8, b-.5, blur, 2., ones),
        'weighted_motion_blur': (a, b, blur, 2., np.exp(.8*np.sin(phase))),
    }


def texture_scene(shape):
    y, x = np.indices(shape)
    cy, cx = (np.asarray(shape)-1)/2
    radius = np.sqrt(((x-cx)/190)**2+((y-cy)/140)**2)
    envelope = np.clip((1-radius)*12, 0., 1.)
    texture = gaussian_filter(np.random.default_rng(107).normal(size=shape), 2.)
    texture *= 12/np.std(texture)
    return 3+envelope*(115+texture+15*np.sin(y/11)+8*np.cos((x+y)/8))


def save_png(path, pixels):
    from PySide6.QtGui import QImage
    # One fixed linear 0..255 mapping for every scene, case and variant.
    codes = np.ascontiguousarray(np.rint(np.clip(pixels/255, 0., 1.)*65535), dtype=np.uint16)
    image = QImage(codes.data, codes.shape[1], codes.shape[0], codes.strides[0],
                   QImage.Format.Format_Grayscale16)
    if not image.save(str(path)):
        raise OSError(path)


def image_errors(image, target, mask):
    difference = image-target
    gradients = np.stack(np.gradient(difference))
    return {'rmse_adu': float(np.sqrt(np.mean(difference[mask]**2))),
            'gradient_vector_rmse_adu_per_px': float(np.sqrt(np.mean(np.sum(gradients[:, mask]**2, axis=0))))}


def run_case(engine, reference, parameters, n, seed, mask, out, name, match_known_blur=False,
             wavelengths=(160., 240.)):
    aa, bb, blur, noise, weights = parameters
    weights = weights/weights.sum()
    rng = np.random.default_rng(seed)
    frames, truths, estimates, clean_frames, shifts = [], [], [], [], []
    desired = np.zeros_like(reference)
    blur_engines = {}
    for a, b, sigma, weight in zip(aa, bb, blur, weights):
        clean, truth, blurred = make_frame(reference, a, b, sigma, wavelengths)
        frame = clean+rng.normal(0, noise, reference.shape)
        # Supply the best constant-vector fit to the true displacement over the
        # scoring mask. Subtract only residual local motion, as production does.
        # This prevents a failure caused merely by deleting global translation.
        shift = truth[:, mask].mean(axis=1)
        matching_engine = engine
        if match_known_blur:
            # An oracle diagnostic: blur is supplied by the simulation, not estimated.
            # Round only the cache key; matching template blur error is below 1e-8 px.
            key = round(float(sigma), 8)
            if key not in blur_engines:
                blur_engines[key] = TorchCircularRegistration(
                    CircularMultiscaleRegistration(blurred), device=engine.device)
            matching_engine = blur_engines[key]
        if hasattr(matching_engine,'displacement_raw'):
            estimated = matching_engine.displacement_raw(frame,shift,lambda:None)
        else:
            estimated = matching_engine.displacement(LocalRegistration.proxy(frame), shift, lambda: None)
        frames.append(frame)
        clean_frames.append(clean)
        truths.append(truth)
        estimates.append(estimated.cpu().numpy())
        shifts.append(shift)
        desired += weight*blurred
    truth, estimated = np.stack(truths), np.stack(estimates)
    origin = np.asarray(shifts)[:, :, None, None]
    mean = np.einsum('n,nahw->ahw', weights, estimated-origin)
    true_mean = np.einsum('n,nahw->ahw', weights, truth-origin)
    fields = {'global': np.broadcast_to(origin, estimated.shape), 'local': estimated,
              'centred': estimated-mean, 'oracle': truth, 'oracle_centred': truth-true_mean}
    images = {key: np.zeros_like(reference) for key in fields}
    clean_images = {key: np.zeros_like(reference) for key in fields}
    noiseless_oracle = np.zeros_like(reference)
    def pull(frame, field):
        pixels = torch.as_tensor(frame, device=engine.device)
        displacement = torch.tensor(field, device=engine.device)
        return sample(pixels, engine.yy+displacement[1], engine.xx+displacement[0]).cpu().numpy()
    for i, (frame, clean, weight) in enumerate(zip(frames, clean_frames, weights)):
        for key, value in fields.items():
            images[key] += weight*pull(frame, value[i])
            # Use the fields measured from noisy frames on their clean twins.
            # This separates geometric error from interpolation suppressing noise.
            clean_images[key] += weight*pull(clean, value[i])
        noiseless_oracle += weight*pull(clean, truth[i])
    report = {'seed': seed, 'frame_count': n, 'a_px': aa.tolist(), 'b_px': bb.tolist(),
              'motion_wavelengths_px': list(wavelengths),
              'blur_sigma_px': blur.tolist(), 'independent_noise_sigma_adu': noise,
              'oracle_global_shifts_xy': np.asarray(shifts).tolist(),
              'true_mean_residual_vector_rms_px': float(np.sqrt(np.mean(np.sum(true_mean[:, mask]**2, axis=0)))),
              'weights': weights.tolist(), 'mask_pixels': int(mask.sum()), 'variants': {}}
    for key in fields:
        report['variants'][key] = {
            'field': weighted_field_errors(fields[key], truth, weights, mask),
            'image_vs_noiseless_oracle': image_errors(images[key], noiseless_oracle, mask),
            'clean_image_vs_noiseless_oracle': image_errors(clean_images[key], noiseless_oracle, mask),
            'image_vs_mean_blurred_scene': image_errors(images[key], desired, mask)}
        save_png(out/f'{name}_{key}.png', images[key])
    save_png(out/f'{name}_noiseless_oracle.png', noiseless_oracle)
    np.savez_compressed(out/f'{name}.npz', **images, noiseless_oracle=noiseless_oracle,
                        **{f'clean_{k}': v for k,v in clean_images.items()},
                        mean_blurred_scene=desired, mean_estimated_field=mean, mean_true_field=true_mean, mask=mask)
    # Pointwise temporal displacement differences are exactly unchanged by subtracting a fixed field.
    np.testing.assert_allclose(estimated[1]-estimated[0], fields['centred'][1]-fields['centred'][0], atol=1e-14)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--reference', type=Path, default=Path('out/saturn-controlled-alignment/reference.npy'))
    parser.add_argument('--frames', type=int, default=48)
    parser.add_argument('--seed', type=int, default=9017)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--scenes', nargs='+', choices=['saturn', 'texture'], default=['saturn', 'texture'])
    parser.add_argument('--cases', nargs='+', choices=list(cases(48)))
    parser.add_argument('--match-known-blur', action='store_true',
                        help='Oracle diagnostic: give the matching template the known simulated frame blur')
    parser.add_argument('--coherent-spacing', type=float, help='Experimental shared spline motion grid spacing')
    parser.add_argument('--coherent-stiffness', type=float, default=.1)
    parser.add_argument('--patch-average', action='store_true')
    parser.add_argument('--validate-pixels',action='store_true',help='Experimental independent-pixel motion validation')
    parser.add_argument('--refit-policy', choices=['full_any','full_scaled'],
                        help='Diagnostic full-data refit after independent-pixel motion gating')
    parser.add_argument('--motion-wavelengths', type=float, nargs=2, default=(160., 240.), metavar=('Y', 'X'))
    args = parser.parse_args()
    if args.frames < 8:
        parser.error('--frames must be at least 8')
    if not np.isfinite(args.motion_wavelengths).all() or min(args.motion_wavelengths) <= 0:
        parser.error('--motion-wavelengths must be positive finite values')
    if args.coherent_spacing is not None and (not np.isfinite(args.coherent_spacing)
            or args.coherent_spacing < 4 or args.match_known_blur):
        parser.error('--coherent-spacing must be finite and >= 4; cannot combine with --match-known-blur')
    if not np.isfinite(args.coherent_stiffness) or args.coherent_stiffness <= 0:
        parser.error('--coherent-stiffness must be positive and finite')
    if args.validate_pixels and (args.coherent_spacing is None or args.match_known_blur):
        parser.error('--validate-pixels requires --coherent-spacing and cannot use known blur')
    if args.validate_pixels:
        args.patch_average = True
    if args.refit_policy and not args.validate_pixels:
        parser.error('--refit-policy requires --validate-pixels')
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    reference = np.load(args.reference)
    report = {'reference_sha256': hashlib.sha256(reference.tobytes()).hexdigest(),
              'device': args.device, 'production_changed': False,
              'global_shift': 'Oracle best constant-vector fit to true displacement over scene mask, independently per frame',
              'matching_template_uses_known_blur': args.match_known_blur,
              'coherent_spacing_px': args.coherent_spacing,
              'coherent_stiffness': args.coherent_stiffness, 'patch_average': args.patch_average,
              'independent_pixel_validation':args.validate_pixels,
              'refit_policy': args.refit_policy,
              'export_black_white_adu': [0., 255.], 'scenes': {}, 'limits': [
                  'Synthetic composed shears are not a full atmospheric or planetary rotation model.',
                  'Reference is fixed; Saturn source contains its existing reference texture and noise.',
                  'Noise is independent Gaussian with an illustrative amplitude, not a fitted sensor model.',
                  'Generation uses cubic interpolation; reconstruction uses the production bilinear sampler.',
                  'Blur is applied before geometric motion; reference does not share the extra blur.',
                  'All frames are retained; only the explicitly weighted case changes scalar contributions.',
                  'Image RMSE uses a noiseless oracle with identical reconstruction interpolation.',
                  'No photometric normalization or output smoothing is applied.']}
    started = time.perf_counter()
    for scene_name, scene in [('saturn', reference), ('texture', texture_scene(reference.shape))]:
        if scene_name not in args.scenes:
            continue
        mask = binary_erosion(scene > .08*scene.max(), iterations=3)
        mask[:8] = mask[-8:] = False
        mask[:, :8] = mask[:, -8:] = False
        matcher = CircularMultiscaleRegistration(scene)
        if args.validate_pixels:
            from tools.validated_registration import ValidatedRegistration
            if args.refit_policy:
                from tools.refit_registration import RefitRegistration
                engine=RefitRegistration(scene,matcher,device=args.device,spacing=args.coherent_spacing,
                                         stiffness=args.coherent_stiffness,policy=args.refit_policy)
            else:
                engine=ValidatedRegistration(scene,matcher,device=args.device,spacing=args.coherent_spacing,
                                             stiffness=args.coherent_stiffness)
        elif args.coherent_spacing is None:
            engine = TorchCircularRegistration(matcher, device=args.device)
        else:
            from tools.coherent_registration import CoherentRegistration
            engine = CoherentRegistration(matcher, device=args.device, spacing=args.coherent_spacing,
                                          stiffness=args.coherent_stiffness,patch_average=args.patch_average)
        report['scenes'][scene_name] = {}
        for name, parameters in cases(args.frames).items():
            if args.cases and name not in args.cases:
                continue
            result = run_case(engine, scene, parameters, args.frames, args.seed, mask,
                              args.out, f'{scene_name}_{name}', args.match_known_blur, args.motion_wavelengths)
            if hasattr(engine, 'stats'):
                result['coherent_fit'] = engine.stats[-args.frames:]
            report['scenes'][scene_name][name] = result
            print(scene_name, name, 'RMSE local/centred/global/oracle:',
                  [round(result['variants'][v]['image_vs_noiseless_oracle']['rmse_adu'], 5)
                   for v in ('local', 'centred', 'global', 'oracle')], flush=True)
            report['elapsed_seconds'] = time.perf_counter()-started
            (args.out/'report.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
