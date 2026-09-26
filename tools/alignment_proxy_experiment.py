"""Vary only global alignment-proxy smoothing, stacking untouched raw frames.

Uses the reference, selected IDs and weights from alignment_noise_experiment.
The sigma-1.5 global-only output must reproduce that experiment's baseline.
"""
import argparse
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from alignment_noise_experiment import measurements, read_png
from planetrecon.export import ExportConfig, export_result
from planetrecon.io.ser import SERSource
from planetrecon.pipeline.align import correlation_peak, correlation_filter
from planetrecon.pipeline.preprocess import best_frame_mask
from planetrecon.pipeline.preprocess_cache import load_cache
from planetrecon.reconstruction import ReconstructionConfig
from planetrecon.result import load_snapshot, save_snapshot
from planetrecon.runtime import apply_thread_limits


def proxy_filters(shape, sigmas):
    fy, fx = np.fft.fftfreq(shape[0])[:, None], np.fft.fftfreq(shape[1])[None, :]
    return np.stack([np.exp(-4*np.pi**2*sigma**2*(fx*fx+fy*fy)) for sigma in sigmas])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    apply_thread_limits(8)
    import torch
    from planetrecon.backends.torch_accel import TorchBackend
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for this controlled experiment')
    torch.set_num_threads(8)
    original = load_snapshot(args.experiment/'global_subpixel.npz')
    config = ReconstructionConfig.from_dict(original.provenance['config'])
    with np.load(args.experiment/'diagnostics.npz') as data:
        indices, quality, prior_shifts = (data[k].copy() for k in ('indices', 'quality', 'global_shifts'))
    reference = np.load(args.experiment/'reference.npy')
    _, text = read_png(args.experiment/'local.png')
    mapping = json.loads(text)['mapping']
    sigmas = [1.5, 3., 6., 12.]
    filters = proxy_filters(reference.shape, sigmas)
    np.testing.assert_array_equal(filters[0], correlation_filter(reference.shape))
    filters = torch.as_tensor(filters, device='cuda:0')
    reference_t = torch.as_tensor(reference, device='cuda:0')
    spectrum = torch.conj(torch.fft.fft2(reference_t-reference_t.mean()))
    sums = np.zeros((len(sigmas), 2, *reference.shape))
    coverage = np.zeros_like(sums)
    shifts = np.zeros((len(indices), len(sigmas), 2))
    backend = TorchBackend()
    started = time.perf_counter()
    with SERSource(args.capture) as source:
        assert source.color_mode() == 'mono'
        selection, status = load_cache(source, config)
        if selection is None:
            raise ValueError(status)
        np.testing.assert_array_equal(indices, np.flatnonzero(best_frame_mask(
            selection, config.stack_percent, config.frame_selection_mode)))
        np.testing.assert_array_equal(quality, selection.measurements[indices, 0])
        for n, index in enumerate(indices):
            raw = source.read_raw(int(index))
            image = torch.as_tensor(raw.astype(np.float64), device='cuda:0')
            cross = torch.fft.fft2(image-image.mean())*spectrum
            correlations = torch.fft.ifft2(cross[None]*filters).real.cpu().numpy()
            score = max(float(quality[n]), 1e-12) if config.quality_weighting else 1.
            for j, corr in enumerate(correlations):
                displacement = correlation_peak(corr)
                shifts[n, j] = displacement
                # The original raw array, never an alignment proxy, is projected.
                projected = backend.backproject(raw, displacement, 'mono')
                add, weight = projected[:2]
                sums[j, n % 2] += score*add
                coverage[j, n % 2] += score*weight
            if (n+1) % 512 == 0:
                print(json.dumps(dict(used=n+1, total=len(indices),
                    seconds=round(time.perf_counter()-started, 1))), flush=True)
    np.testing.assert_allclose(shifts[:, 0], prior_shifts, rtol=0, atol=1e-8)
    report = dict(n_used=len(indices), sigmas=sigmas, device=torch.cuda.get_device_name(0),
        reference_sha256=hashlib.sha256(reference.tobytes()).hexdigest(),
        selected_indices_sha256=hashlib.sha256(indices.tobytes()).hexdigest(),
        control='Global translation only for every sigma; fixed production reference and raw backprojector',
        maximum_baseline_shift_difference=float(abs(shifts[:, 0]-prior_shifts).max()), variants={})
    np.savez_compressed(args.out/'diagnostics.npz', indices=indices, quality=quality, shifts=shifts)
    for j, sigma in enumerate(sigmas):
        name = f'global_sigma{sigma:g}'
        weight = coverage[j].sum(axis=0)
        image = np.divide(sums[j].sum(axis=0), weight, out=np.zeros_like(weight), where=weight>0)
        halves = np.divide(sums[j], coverage[j], out=np.zeros_like(sums[j]), where=coverage[j]>0)
        if j == 0:
            np.testing.assert_allclose(image, original.image, rtol=0, atol=1e-8)
            report['maximum_baseline_image_difference_adu'] = float(abs(image-original.image).max())
        provenance = deepcopy(original.provenance)
        provenance['experiment'] = dict(kind='alignment_proxy_sigma', sigma=sigma,
            local_alignment_used=False, reference_sha256=report['reference_sha256'],
            note='Only alignment measurements are smoothed; original config retained as provenance')
        product = replace(original, image=image, coverage=weight, validity=weight>0, provenance=provenance)
        save_snapshot(args.out/f'{name}.npz', product)
        np.savez_compressed(args.out/f'{name}_halves.npz', images=halves, coverage=coverage[j])
        export_result(product, args.out/f'{name}.png', ExportConfig('png16', mapping['black'], mapping['white']))
        delta = shifts[:, j]-shifts[:, 0]
        report['variants'][name] = dict(raw=measurements(image),
            half_difference=measurements(image, difference=(halves[0]-halves[1])/2),
            shift_change_rms_px=float(np.sqrt(np.mean(delta**2))),
            shift_change_magnitude_percentiles=np.percentile(np.linalg.norm(delta,axis=1),[50,90,99,100]).tolist())
    report['seconds'] = time.perf_counter()-started
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__':
    main()
