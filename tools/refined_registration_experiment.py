"""Isolate global peak fitting and interpolation, holding Saturn inputs fixed.

Four cells: existing/refined shifts crossed with bilinear/cubic resampling.
The fixed reference, frame IDs, weights and AS comparison smoothing are shared.
Production stacking is unchanged. CUDA handles continuous correlation fitting;
bounded CPU batches handle interpolation and capture-order accumulation.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.ndimage import gaussian_filter, shift as ndshift
import torch

from planetrecon.export import ExportConfig, export_result
from planetrecon.io.ser import SERSource
from planetrecon.pipeline.align import correlation_filter, correlation_peak
from planetrecon.result import load_snapshot, save_snapshot
from planetrecon.runtime import apply_thread_limits
from tools.alignment_noise_experiment import read_png, measurements

NAMES = ('existing_bilinear', 'refined_bilinear', 'existing_cubic', 'refined_cubic')


def continuous_peak(cross, initial):
    """Newton maximum of the same filtered correlation's continuous Fourier sum.

    Inputs: batch of weighted complex cross spectra, initial (x,y) displacements.
    A joint Hessian retains mixed-axis curvature omitted by independent 1D fits.
    """
    _, h, w = cross.shape
    device = cross.device
    wx = 2*torch.pi*torch.fft.fftfreq(w, device=device, dtype=torch.float64)[None, :]
    wy = 2*torch.pi*torch.fft.fftfreq(h, device=device, dtype=torch.float64)[:, None]
    initial = torch.as_tensor(initial, dtype=torch.float64, device=device)
    current = initial.clone()
    valid = torch.ones(len(cross), device=device, dtype=torch.bool)
    last_step = torch.full_like(current, float('inf'))
    for _ in range(8):
        phase = current[:, 0, None, None]*wx+current[:, 1, None, None]*wy
        value = cross*torch.exp(1j*phase)
        real, imag = value.real, value.imag
        gx, gy = -(imag*wx).sum((-1,-2)), -(imag*wy).sum((-1,-2))
        hxx = -(real*wx*wx).sum((-1,-2))
        hyy = -(real*wy*wy).sum((-1,-2))
        hxy = -(real*wx*wy).sum((-1,-2))
        determinant = hxx*hyy-hxy*hxy
        good = (hxx < 0)&(hyy < 0)&(determinant > 0)
        denominator = torch.where(good, determinant, torch.ones_like(determinant))
        step = torch.stack([(hyy*gx-hxy*gy)/denominator,
                            (hxx*gy-hxy*gx)/denominator], dim=1)
        good &= torch.isfinite(step).all(dim=1)&(step.abs().max(dim=1).values < .75)
        valid &= good
        current -= torch.where(valid[:, None], step, 0.)
        last_step = step
        if bool((~valid | (step.abs().max(dim=1).values < 1e-9)).all()):
            break
    valid &= ((current-initial).abs().max(dim=1).values < .75)
    valid &= (last_step.abs().max(dim=1).values < 1e-7)
    return torch.where(valid[:, None], current, initial).cpu().numpy(), valid.cpu().numpy()


def project(raw, shifts):
    """Same recorded ADU for both kernels; cubic uses complete interior support."""
    raw = np.asarray(raw, dtype=float)
    h, w = raw.shape
    yy, xx = np.indices(raw.shape)
    output = []
    for order in (1, 3):
        for dx, dy in shifts:
            if order == 1:
                weight = ndshift(np.ones_like(raw), (-dy, -dx), order=1,
                                 prefilter=False, mode='grid-constant')
                signal = ndshift(raw, (-dy, -dx), order=1, prefilter=False, mode='grid-constant')
            else:
                weight = ((yy+dy >= 3)&(yy+dy <= h-4)&(xx+dx >= 3)&(xx+dx <= w-4)).astype(float)
                signal = ndshift(raw, (-dy, -dx), order=3, mode='nearest')*weight
            output.append((signal, weight))
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture', type=Path, default=Path('2024-09-27-1154_3-CK-R-Sat.ser'))
    p.add_argument('--baseline', type=Path, default=Path('out/saturn-controlled-alignment'))
    p.add_argument('--out', type=Path, default=Path('out/saturn-refined-registration'))
    p.add_argument('--limit', type=int, help='Pilot only; omitted for all 5738 selected frames')
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    apply_thread_limits(16)
    torch.set_num_threads(8)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for this bounded experiment')
    original = load_snapshot(args.baseline/'global_subpixel.npz')
    reference = np.load(args.baseline/'reference.npy')
    with np.load(args.baseline/'diagnostics.npz') as d:
        indices, quality, prior = (d[k].copy() for k in ('indices', 'quality', 'global_shifts'))
    if args.limit:
        indices, quality, prior = indices[:args.limit], quality[:args.limit], prior[:args.limit]
    _, text = read_png(args.baseline/'local.png')
    mapping = json.loads(text)['mapping']
    ref = torch.as_tensor(reference, device='cuda', dtype=torch.float64)
    spectrum = torch.conj(torch.fft.fft2(ref-ref.mean()))
    filt = torch.as_tensor(correlation_filter(reference.shape), device='cuda')
    sums = np.zeros((4, 2, *reference.shape))
    coverage = np.zeros_like(sums)
    refined, usable, baseline_errors = [], [], []
    started = time.perf_counter()
    with SERSource(args.capture) as source, ThreadPoolExecutor(max_workers=16) as pool:
        for start in range(0, len(indices), 16):
            ids = indices[start:start+16]
            # Only the owner accesses the SER's shared file cursor.
            frames = np.stack([source.read_raw(int(i)) for i in ids])
            images = torch.as_tensor(frames.astype(float), device='cuda')
            images -= images.mean((-1,-2), keepdim=True)
            cross = torch.fft.fft2(images)*spectrum*filt
            existing = np.array([correlation_peak(c) for c in torch.fft.ifft2(cross).real.cpu().numpy()])
            baseline_errors.append(float(np.max(np.abs(existing-prior[start:start+len(ids)]))))
            better, valid = continuous_peak(cross, existing)
            refined.extend(better); usable.extend(valid)
            futures = [pool.submit(project, frame, (old, new))
                       for frame, old, new in zip(frames, existing, better)]
            for offset, future in enumerate(futures):
                n = start+offset
                score = max(float(quality[n]), 1e-12)
                for j, (signal, weight) in enumerate(future.result()):
                    sums[j, n % 2] += score*signal
                    coverage[j, n % 2] += score*weight
            if start % 512 == 0:
                print(f'{start+len(ids)}/{len(indices)} frames, {time.perf_counter()-started:.1f}s', flush=True)
    refined = np.array(refined)
    report = {'n_used': len(indices), 'pilot': bool(args.limit),
        'device': torch.cuda.get_device_name(0),
        'reference_sha256': hashlib.sha256(reference.tobytes()).hexdigest(),
        'indices_sha256': hashlib.sha256(indices.tobytes()).hexdigest(),
        'baseline_shift_max_error_px': max(baseline_errors),
        'converged_fraction': float(np.mean(usable)),
        'refinement_magnitude_px_p50_p90_p99_max': np.percentile(np.linalg.norm(refined-prior,axis=1), [50,90,99,100]).tolist(),
        'variants': {}}
    np.testing.assert_allclose(max(baseline_errors), 0., atol=1e-8, rtol=0)
    np.savez_compressed(args.out/'diagnostics.npz', indices=indices, quality=quality,
                        existing=prior, refined=refined, converged=usable)
    for j, name in enumerate(NAMES):
        weight = coverage[j].sum(axis=0)
        image = np.divide(sums[j].sum(axis=0), weight, out=np.zeros_like(weight), where=weight>0)
        if j == 0 and not args.limit:
            error = float(np.max(np.abs(image-original.image)))
            report['baseline_image_max_error_adu'] = error
            assert error < 1e-8
        provenance = deepcopy(original.provenance)
        provenance['registration_experiment'] = {'name': name, 'diagnostic_only': True,
            'global_only': True, 'fixed_reference': True, 'cubic_support_border_px': 3}
        result = replace(original, image=image, coverage=weight, validity=weight>0,
                         provenance=provenance, n_used=len(indices))
        save_snapshot(args.out/f'{name}.npz', result)
        half = np.divide(sums[j], coverage[j], out=np.zeros_like(sums[j]), where=coverage[j]>0)
        np.savez_compressed(args.out/f'{name}_halves.npz', images=half, coverage=coverage[j])
        report['variants'][name] = {'raw': measurements(image)}
        result = replace(result, image=gaussian_filter(image, 1., radius=3))
        result.provenance['comparison_smoothing'] = {'sigma_px': 1., 'radius_px': 3}
        export_result(result, args.out/f'{name}_as_transfer.png',
                      ExportConfig('png16', mapping['black'], mapping['white'], 1.))
    report['seconds'] = time.perf_counter()-started
    (args.out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
