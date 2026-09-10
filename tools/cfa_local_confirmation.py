"""Fixed 512-frame local/upper-half confirmation; see the declared working plan.

Requires the private Jupiter capture and existing preprocessing cache. Refuses to
repeat a started study or overwrite user outputs. No application settings change.
"""
from pathlib import Path
import hashlib
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scipy.ndimage import gaussian_filter, shift as move
from PySide6.QtGui import QImage
import tifffile

from planetrecon.backends.torch_accel import TorchBackend
from planetrecon.io.source import open_source
from planetrecon.pipeline.baseline import _colour_registration_plane, _saturated
from planetrecon.pipeline.local_align import LocalRegistration, build_template
from planetrecon.pipeline.preprocess import best_frame_mask
from planetrecon.pipeline.preprocess_cache import load_cache
from planetrecon.provenance import source_hash
from planetrecon.reconstruction import ReconstructionConfig
from tools.cfa_local_transport_probe import LocalQuadraticAccumulator, completed
from tools.compare_stack_reference import read_reference


def compare_regions(ordinary, candidate, target, validity):
    metrics = {}
    mapped = {name: np.zeros_like(target) for name in ('ordinary', 'quadratic')}
    combined = {name: [[], [], [], []] for name in mapped}
    masks = []
    for n, region in enumerate(('left', 'centre', 'right')):
        section = (slice(None), slice(n*96, (n+1)*96))
        ref = target[section]
        mask = validity[section].all(axis=-1).copy()
        mask[:8] = False; mask[-8:] = False; mask[:, :8] = False; mask[:, -8:] = False
        if mask.sum() < 100:
            raise ValueError(f'insufficient support in declared {region} region')
        masks.append(mask)
        metrics[region] = {'common_pixels': int(mask.sum())}
        for name, image in [('ordinary', ordinary), ('quadratic', candidate)]:
            image = image[section]
            aligned = image.copy()
            fits = []
            for c in range(3):
                a, b = np.linalg.lstsq(np.column_stack([image[..., c][mask], np.ones(mask.sum())]),
                                      ref[..., c][mask], rcond=None)[0]
                aligned[..., c] = image[..., c]*a+b
                fits.append([float(a), float(b)])
            hp = aligned-gaussian_filter(aligned, (3, 3, 0))
            refhp = ref-gaussian_filter(ref, (3, 3, 0))
            rms = float(np.linalg.norm((aligned-ref)[mask])/np.linalg.norm(ref[mask]))
            correlation = float(np.corrcoef(hp[mask].ravel(), refhp[mask].ravel())[0, 1])
            if not np.isfinite([rms, correlation]).all():
                raise ValueError(f'unconstrained comparison in {region}')
            metrics[region][name] = {'relative_rmse': rms, 'highpass_correlation_sigma3': correlation,
                                     'gain_offset': fits}
            mapped[name][section] = aligned
            for pool, values in zip(combined[name], [(aligned-ref)[mask], ref[mask], hp[mask], refhp[mask]]):
                pool.append(values.ravel())
        a, b = metrics[region]['ordinary'], metrics[region]['quadratic']
        metrics[region]['passes_both_metrics'] = (b['relative_rmse'] < a['relative_rmse']
                                                 and b['highpass_correlation_sigma3'] > a['highpass_correlation_sigma3'])
    metrics['aggregate'] = {}
    for name, arrays in combined.items():
        error, ref, hp, refhp = [np.concatenate(v) for v in arrays]
        metrics['aggregate'][name] = {'relative_rmse': float(np.linalg.norm(error)/np.linalg.norm(ref)),
                                      'highpass_correlation_sigma3': float(np.corrcoef(hp, refhp)[0, 1])}
    return metrics, mapped, np.concatenate(masks, axis=1)


def main():
    out = ROOT/'out/cfa-local-confirmation'
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    version = source_hash()
    files = ['tools/cfa_local_transport_probe.py', 'tools/cfa_local_confirmation.py', 'docs/local-quality-development.md']
    hashes = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files}
    for name in files:
        (out/Path(name).name).write_bytes((ROOT/name).read_bytes())
    backend = TorchBackend()
    region = (slice(164, 260), slice(184, 472))
    with open_source(ROOT/'2022-09-10-0649_2-GB-L-Jup_ZWO ASI224MC.ser') as source:
        config = ReconstructionConfig(local_alignment=True, stack_percent=50, frame_selection_mode='quality_range')
        cache, cache_report = load_cache(source, config)
        if cache is None:
            raise ValueError(f'matching preprocessing required: {cache_report}')
        selected = np.flatnonzero(best_frame_mask(cache, 50, 'quality_range'))
        qualities = cache.measurements[:, 0]
        anchor = int(selected[np.argmax(qualities[selected])])
        if anchor != 1947 or len(selected) != 1645:
            raise ValueError('capture selection or anchor differs from the declared protocol')
        indices = selected[np.linspace(0, len(selected)-1, 512).round().astype(int)]
        if anchor not in indices:
            indices[np.argmin(abs(indices-anchor))] = anchor
            indices.sort()
        train = selected[np.argsort(-qualities[selected], kind='stable')[:64]]
        color, shape = source.color_mode(), source.frame_shape()
        plane = lambda index: _colour_registration_plane(source.read_raw(int(index)), color)
        template = build_template(plane(anchor), train, plane, backend.phase_correlation, config.max_shift_px)
        matcher = LocalRegistration(template, use_cuda=True)
        model = LocalQuadraticAccumulator(shape, color, region=region)
        manifest = {'scope': 'Fixed 512-frame three-region development comparison; reference is not truth',
                    'source_hash': version, 'code_hashes': hashes, 'cache_identity': cache.identity,
                    'selected_capture_frames': len(selected), 'sample_frames': len(indices),
                    'indices': indices.tolist(), 'qualities': qualities[indices].tolist(),
                    'anchor': anchor, 'template_indices': train.tolist(),
                    'template_sha256': hashlib.sha256(template.tobytes()).hexdigest(),
                    'selection': 'upper 50% capture quality range then cached screening',
                    'weight': 'original linear cached Emil quality', 'output_region_yx': [164, 260, 184, 472]}
        (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        map_digest, raw_digest = hashlib.sha256(), hashlib.sha256()
        for n, index in enumerate(indices):
            raw = source.read_raw(int(index))
            if not np.isfinite(raw).all() or _saturated(raw, source.metadata().bit_depth):
                raise ValueError(f'unexpected rejected raw frame {index}')
            raw_digest.update(raw.tobytes())
            proxy = _colour_registration_plane(raw, color)
            global_shift = backend.phase_correlation(template, proxy)
            if not np.isfinite(global_shift).all() or max(abs(v) for v in global_shift) > config.max_shift_px:
                raise ValueError(f'unexpected rejected alignment {index}')
            local = matcher.displacement(proxy, global_shift)
            map_digest.update(np.asarray(local).tobytes())
            model.add(raw, local, float(qualities[index]))
            if (n+1) % 16 == 0:
                print(f'frames {n+1}/512 elapsed {time.monotonic()-started:.1f}s', flush=True)
        after, variance, degree, before = model.finish()
        full = completed(np.divide(model.sums, model.weights, out=np.zeros_like(model.sums), where=model.weights > 0),
                         model.weights, color)
        reference = read_reference(ROOT/'2022-09-10-0649_2-GB-L-Jup_ZWO ASI224MC_lapl6_ap59.png')
        dx, dy = backend.phase_correlation(reference[..., 1], full.image[..., 1])
        target = move(reference, (dy, dx, 0), order=1, mode='constant', prefilter=False)[region]
        metrics, previews, mask = compare_regions(before.image, after, target, before.validity)
        for name, image in [('ordinary', before.image), ('quadratic', after)]:
            tifffile.imwrite(out/(name+'.tif'), image.astype('float32'), photometric='rgb')
        previews['reference'] = target
        white = min(65535., max(float(v.max()) for v in previews.values())*1.43)
        for name, image in previews.items():
            pixels = np.ascontiguousarray(np.uint8(np.clip(image/white, 0, 1)*255))
            preview = QImage(pixels.data, pixels.shape[1], pixels.shape[0], pixels.strides[0], QImage.Format.Format_RGB888)
            if not preview.save(str(out/(name+'.png'))):
                raise OSError(f'could not save {name} preview')
        np.savez_compressed(out/'comparison.npz', ordinary=before.image, quadratic=after, variance=variance,
                            degree=degree, reference=target, mask=mask, indices=indices, qualities=qualities[indices],
                            template=template, direct_weights=model.weights[region], validity=before.validity)
        if source_hash() != version or any(hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != value for name, value in hashes.items()):
            raise RuntimeError('source changed during comparison')
        payload = {**manifest, 'raw_sha256': raw_digest.hexdigest(), 'maps_sha256': map_digest.hexdigest(),
                   'registration_xy': [dx, dy], 'metrics': metrics,
                   'passes_all_regions': all(metrics[r]['passes_both_metrics'] for r in ('left', 'centre', 'right')),
                   'degree_counts': {str(k): int((degree == k).sum()) for k in (-1, 0, 1, 2)},
                   'elapsed_s': time.monotonic()-started}
        (out/'confirmation.json').write_text(json.dumps(payload, indent=2)+'\n')
        print(json.dumps({'passes_all_regions': payload['passes_all_regions'], 'metrics': metrics}), flush=True)


if __name__ == '__main__':
    main()
