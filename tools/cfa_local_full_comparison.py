"""Full selected Jupiter comparison through the application; no sharpening.

Run once, or --resume after interruption. Private capture/cache are required.
Both outputs use the same actual application frames, maps and original weights.
"""
from copy import deepcopy
from dataclasses import replace
import argparse
import hashlib
import json
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scipy.ndimage import binary_erosion, gaussian_filter, shift as move
from PySide6.QtGui import QImage

from planetrecon.export import ExportConfig, export_result
from planetrecon.io.source import open_source
from planetrecon.pipeline.align import phase_correlation_shift
from planetrecon.pipeline.baseline import stack_source
from planetrecon.pipeline.preprocess import best_frame_mask
from planetrecon.pipeline.preprocess_cache import load_cache
from planetrecon.provenance import source_hash
from planetrecon.reconstruction import ReconstructionConfig
from planetrecon.result import save_snapshot, load_snapshot
from tools.cfa_local_confirmation import compare_regions
from tools.compare_stack_reference import read_reference

OUT = ROOT/'out/cfa-local-full'
SOURCE = ROOT/'2022-09-10-0649_2-GB-L-Jup_ZWO ASI224MC.ser'
REFERENCE = ROOT/'2022-09-10-0649_2-GB-L-Jup_ZWO ASI224MC_lapl6_ap59.png'


def write_json(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def fit_pair(ordinary, candidate, target, validity):
    """One fixed planet-interior mask; no fitted spatial transform per arm."""
    mask = binary_erosion(target[..., 1] > target[..., 1].max()*.15, iterations=10)
    mask &= validity.all(axis=2) & np.isfinite(target).all(axis=2)
    if mask.sum() < 100:
        raise ValueError('insufficient common planet interior')
    metrics, mapped = {}, {}
    refhp = target-gaussian_filter(target, (3, 3, 0))
    for name, image in [('ordinary', ordinary), ('interpolated', candidate)]:
        fitted = image.copy(); fits = []
        for c in range(3):
            a, b = np.linalg.lstsq(np.column_stack([image[..., c][mask], np.ones(mask.sum())]),
                                  target[..., c][mask], rcond=None)[0]
            fitted[..., c] = image[..., c]*a+b
            fits.append([float(a), float(b)])
        hp = fitted-gaussian_filter(fitted, (3, 3, 0))
        metrics[name] = {'relative_rmse': float(np.linalg.norm((fitted-target)[mask])/np.linalg.norm(target[mask])),
                         'highpass_correlation_sigma3': float(np.corrcoef(hp[mask].ravel(), refhp[mask].ravel())[0, 1]),
                         'gain_offset': fits}
        mapped[name] = fitted
    a, b = metrics['ordinary'], metrics['interpolated']
    metrics['common_pixels'] = int(mask.sum())
    metrics['passes_both_metrics'] = (b['relative_rmse'] < a['relative_rmse'] and
                                      b['highpass_correlation_sigma3'] > a['highpass_correlation_sigma3'])
    return metrics, mapped, mask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.resume:
        if not (OUT/'manifest.json').exists() or (OUT/'comparison.json').exists():
            raise ValueError('resume requires an unfinished declared comparison')
    else:
        OUT.mkdir(parents=True, exist_ok=False)
    stopped = False
    def cancel(*_):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGINT, cancel)
    signal.signal(signal.SIGTERM, cancel)
    started = time.monotonic(); last_log = 0
    config = ReconstructionConfig(device='gpu', threads=2, batch_frames=32,
        local_alignment=True, local_cfa_interpolation=True, stack_percent=50,
        frame_selection_mode='quality_range')
    version = source_hash()
    manifest = {'scope': 'Full selected-capture comparison; conventional reference is not independent truth',
        'config': config.to_dict(), 'package_sha256': version,
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'reference_sha256': hashlib.sha256(REFERENCE.read_bytes()).hexdigest(),
        'gate': 'lower relative RMS and higher sigma-three highpass correlation in the fixed planet interior and all three previous regions',
        'planet_mask': 'aligned reference green above 15 percent peak, eroded ten pixels, intersected with common RGB validity',
        'regional_strip_yx': [164, 260, 184, 472],
        'registration': 'one translation from full ordinary green to conventional reference; reference resampled once for both arms',
        'photometry': 'per-colour gain and offset on the same mask for each arm; no sharpening or fitted detail'}
    ordinary = None
    with open_source(SOURCE) as source:
        cache, report = load_cache(source, config, should_cancel=lambda: stopped)
        if cache is None:
            raise ValueError(f'matching cache required: {report}')
        mask = best_frame_mask(cache, 50, 'quality_range')
        ids = np.flatnonzero(mask)
        anchor = int(ids[np.argmax(cache.measurements[ids, 0])])
        if len(ids) != 1645 or anchor != 1947:
            raise ValueError('selection differs from the declared 1645 frames and anchor 1947')
        manifest.update(indices=ids.tolist(), qualities=cache.measurements[ids, 0].tolist(),
                        anchor=anchor, cache_identity=cache.identity, detector_shape=list(source.frame_shape()))
        if args.resume:
            if json.loads((OUT/'manifest.json').read_text()) != manifest:
                raise ValueError('comparison inputs, code or declaration changed')
            if (OUT/'ordinary.npz').exists():
                ordinary = load_snapshot(OUT/'ordinary.npz')
        else:
            write_json(OUT/'manifest.json', manifest)
        def event(result, info):
            nonlocal ordinary, last_log
            processed = int(info.get('n_processed', 0))
            if result.n_rejected != int((~mask[:processed]).sum()):
                raise ValueError('unexpected selected frame rejection; comparison stopped')
            elapsed = time.monotonic()-started
            if elapsed-last_log >= 30 or result.stage == 'local CFA final fit':
                record = {'stage': result.stage, 'used': result.n_used, 'selected': len(ids),
                          'processed': processed, 'captured': len(mask), 'elapsed_this_process_s': elapsed}
                write_json(OUT/'progress.json', record)
                print(json.dumps(record), flush=True); last_log = elapsed
            if result.stage == 'baseline' and processed == len(mask) and result.n_used == len(ids):
                ordinary = deepcopy(replace(result, stage='final', incomplete=False))
                ordinary.provenance.pop('local_cfa_interpolation', None)
                ordinary.provenance['paired_comparison'] = 'ordinary output from the same selected frames and local maps as the interpolated arm'
                save_snapshot(OUT/'ordinary.npz', ordinary)
        result = stack_source(source, config, on_event=event, should_cancel=lambda: stopped,
                              resume_from=OUT/'state.npz' if args.resume else None,
                              state_checkpoint=OUT/'state.npz')
    if stopped or result.incomplete:
        print('Interrupted; rerun with --resume to continue the last checkpoint.', flush=True)
        return 2
    if ordinary is None or result.n_used != len(ids):
        raise ValueError('paired complete ordinary output missing')
    save_snapshot(OUT/'interpolated.npz', result)
    reference = read_reference(REFERENCE)
    if reference.shape != result.image.shape:
        raise ValueError('reference grid differs from the capture')
    dx, dy = phase_correlation_shift(reference[..., 1], ordinary.image[..., 1])
    target = move(reference, (dy, dx, 0), order=1, mode='constant', prefilter=False)
    validity = ordinary.validity & result.validity
    full, previews, mask = fit_pair(ordinary.image, result.image, target, validity)
    region = (slice(164, 260), slice(184, 472))
    regional, _, _ = compare_regions(ordinary.image[region], result.image[region], target[region], validity[region])
    passed = full['passes_both_metrics'] and all(regional[r]['passes_both_metrics'] for r in ('left', 'centre', 'right'))
    for name, value in [('ordinary', ordinary), ('interpolated', result)]:
        if not (OUT/(name+'.tif')).exists():
            export_result(value, OUT/(name+'.tif'), ExportConfig(encoding='tiff32'))
    previews['reference'] = target
    white = min(65535., max(float(v.max()) for v in previews.values())*1.43)
    for name, value in previews.items():
        rgb = np.ascontiguousarray(np.uint8(np.clip(value/white, 0, 1)*255))
        image = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format.Format_RGB888)
        if not image.save(str(OUT/(name+'.png'))):
            raise OSError('preview save failed')
    np.savez_compressed(OUT/'comparison-arrays.npz', reference=target, mask=mask,
                        ordinary=previews['ordinary'], interpolated=previews['interpolated'])
    if source_hash() != version:
        raise ValueError('application source changed during comparison')
    payload = {**manifest, 'n_used': result.n_used, 'registration_xy': [dx, dy],
               'planet_interior': full, 'regions': regional, 'passes_all_checks': passed,
               'elapsed_this_process_s': time.monotonic()-started,
               'moment_execution_history': result.provenance['local_cfa_interpolation']['moment_execution_history'],
               'fit_channel_counts': result.provenance['local_cfa_interpolation']['fit_channel_counts'],
               'warnings': result.warnings}
    write_json(OUT/'comparison.json', payload)
    print(json.dumps({'passes_all_checks': passed, 'planet_interior': full, 'regions': regional}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
