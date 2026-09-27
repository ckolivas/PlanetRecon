"""Replay the fixed Saturn selection with the corrected joint forward model.

Bounded shards retain interleaved half-stacks and per-frame fit diagnostics.
Only motion estimation differs: raw values, quality weights, reference, global
shifts and final backprojection match the preceding capture experiments.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from planetrecon.backends.torch_accel import TorchBackend
from planetrecon.export import ExportConfig, export_result
from planetrecon.io.ser import SERSource
from planetrecon.pipeline.circular_align import CircularMultiscaleRegistration
from planetrecon.pipeline.local_align import LocalRegistration
from planetrecon.result import ReconstructionResult, load_snapshot, save_snapshot
from tools.alignment_noise_experiment import read_png
from tools.coherent_saturn_experiment import inputs, combine_shards
from tools.spline_joint_registration import SplineJointRegistration

POLICIES = ['global', 'coherent', 'joint_strong', 'joint_validated']
CAPTURE = Path('2024-09-27-1154_3-CK-R-Sat.ser')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def capture_identity():
    stat = CAPTURE.stat()
    return dict(path=str(CAPTURE.resolve()), size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def combine(root, count, shape, hashes, config):
    paths = sorted(root.glob('shard_*.npz'))
    for path in paths:
        with np.load(path) as shard:
            metadata = json.loads(str(shard['metadata']))
            actual = [s['position'] for s in metadata['fit_statistics']]
            if actual != shard['positions'].tolist():
                raise ValueError('Frame diagnostics do not match shard positions')
    return combine_shards(paths, count, shape, hashes, config)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--sample-count', type=int, default=5738)
    p.add_argument('--prepare', action='store_true')
    p.add_argument('--combine', action='store_true')
    p.add_argument('--worker', type=int, default=0)
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--shard-size', type=int, default=128)
    args = p.parse_args()
    if args.prepare and args.combine:
        p.error('prepare and combine are separate operations')
    if args.workers < 1 or not 0 <= args.worker < args.workers or args.shard_size < 1:
        p.error('invalid worker or shard size')
    root, indices, quality, shifts, reference, hashes = inputs()
    if not 2 <= args.sample_count <= len(indices):
        p.error('sample count must be between two and the fixed selection size')
    selected = np.linspace(0, len(indices)-1, args.sample_count, dtype=int)
    hashes['sample_positions'] = hashlib.sha256(selected.tobytes()).hexdigest()
    for name in ('joint_registration', 'spline_joint_registration', 'joint_saturn_experiment'):
        hashes[name] = digest(Path('tools')/(name+'.py'))
    config = dict(policies=POLICIES, sample_count=args.sample_count, spacing=32.,
                  stiffness=.3, iterations=100, sampler='prefiltered_cubic_spline',
                  training_stride=2, half_split='selected_position_modulo_two')
    manifest_path = args.out/'manifest.json'
    if args.prepare:
        args.out.mkdir(parents=True, exist_ok=False)
        identity = capture_identity()
        capture_hash = digest(CAPTURE)
        if identity != capture_identity():
            raise ValueError('Capture changed while hashing')
        manifest = dict(capture=identity, capture_sha256=capture_hash, hashes=hashes,
                        config=config, frame_indices=indices[selected].tolist())
        manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
        print('Prepared', args.sample_count, 'frames; capture SHA256', capture_hash, flush=True)
        return
    manifest = json.loads(manifest_path.read_text())
    if (manifest['capture'] != capture_identity() or manifest['hashes'] != hashes
            or manifest['config'] != config or manifest['frame_indices'] != indices[selected].tolist()):
        raise ValueError('Prepared inputs or code changed')
    hashes['capture'] = manifest['capture_sha256']
    shape = (len(POLICIES), 2, *reference.shape)
    if args.combine:
        total, support, stats = combine(args.out, len(selected), shape, hashes, config)
        # Check capture bytes again before labelling a completed experiment.
        if digest(CAPTURE) != hashes['capture']:
            raise ValueError('Capture content changed during replay')
        original = load_snapshot(root/'local.npz')
        _, text = read_png(root/'local.png')
        mapping = json.loads(text)['mapping']
        export = ExportConfig('png16', mapping['black'], mapping['white'], 1.)
        # Refuse overwrites before publishing any member of the export set.
        for name in POLICIES:
            for suffix in ('.npz', '.png', '_half0.png', '_half1.png'):
                if (args.out/(name+suffix)).exists():
                    raise FileExistsError(args.out/(name+suffix))
        for i, name in enumerate(POLICIES):
            coverage = support[i].sum(0)
            pixels = np.divide(total[i].sum(0), coverage, out=np.zeros_like(coverage), where=coverage>0)
            result = ReconstructionResult(image=pixels, coverage=coverage, validity=coverage>0,
                units=original.units, channel_order='mono', backend='cuda', precision='float64',
                stage='diagnostic', incomplete=False, reference_epoch=original.reference_epoch, n_used=len(selected),
                provenance=dict(diagnostic_only=True, policy=name, config=config, hashes=hashes,
                                frame_indices=indices[selected].tolist()))
            save_snapshot(args.out/f'{name}.npz', result)
            export_result(result, args.out/f'{name}.png', export)
            for half in (0, 1):
                result.coverage = support[i, half]
                result.validity = result.coverage>0
                result.image = np.divide(total[i, half], result.coverage, out=np.zeros_like(coverage),
                                         where=result.validity)
                result.n_used = int(np.count_nonzero(np.arange(len(selected)) % 2 == half))
                result.provenance = dict(result.provenance, half=half)
                export_result(result, args.out/f'{name}_half{half}.png', export)
        np.savez_compressed(args.out/'half_stacks.npz', signal=total, support=support)
        stats.sort(key=lambda s: s['position'])
        (args.out/'fit_statistics.json').write_text(json.dumps(stats, indent=2)+'\n')
        report = dict(n_used=len(selected), frame_indices=indices[selected].tolist(), hashes=hashes,
            config=config, capture=manifest['capture'], production_changed=False,
            brightness_normalization=False, output_filtering=False,
            accepted_motion=sum(s['fit']['heldout_improves'] and not s['fit']['fallback'] for s in stats),
            optimizer_successes=sum(s['fit'].get('success', False) for s in stats),
            optimizer_messages=dict(Counter(s['fit'].get('message', '') for s in stats)),
            geometric_fallbacks=sum(s['fit']['fallback'] for s in stats),
            fit_statistics_sha256=digest(args.out/'fit_statistics.json'),
            half_stacks_sha256=digest(args.out/'half_stacks.npz'))
        (args.out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
        print('Combined', len(selected), 'frames; accepted motion', report['accepted_motion'], flush=True)
        return
    torch.set_num_threads(4)
    engine = SplineJointRegistration(reference, CircularMultiscaleRegistration(reference))
    backend = TorchBackend()
    with SERSource(CAPTURE) as source:
        if source.color_mode() != 'mono' or source.frame_shape() != reference.shape:
            raise ValueError('Unexpected capture geometry or colour')
        for start in range(args.worker*args.shard_size, len(selected), args.workers*args.shard_size):
            path = args.out/f'shard_{start:05d}.npz'
            if path.exists():
                raise FileExistsError(path)
            positions = np.arange(start, min(start+args.shard_size, len(selected)))
            total, support = np.zeros(shape), np.zeros(shape)
            statistics = []
            started = time.perf_counter()
            for j, position in enumerate(positions):
                at = selected[position]
                frame = source.read_raw(int(indices[at]))
                joint, stats = engine.fit(frame, shifts[at], .3, lambda: None)
                coherent = engine.engine.displacement(LocalRegistration.proxy(frame), shifts[at], lambda: None)
                origin = torch.as_tensor(shifts[at], device=engine.device)[:, None, None]
                global_field = origin.expand_as(joint)
                fields = dict(global_only=global_field, coherent=coherent, joint_strong=joint,
                              joint_validated=joint if stats.get('heldout_improves', False) else global_field)
                fields['global'] = fields.pop('global_only')
                residual = (joint-origin)[:, engine.mask.bool()]
                statistics.append(dict(position=int(position), frame_index=int(indices[at]), fit=stats,
                    residual_vector_rms_px=float(residual.square().sum(0).mean().sqrt()),
                    coherent=engine.engine.stats[-1]))
                half = position % 2
                for i, name in enumerate(POLICIES):
                    signal, weight, *_ = backend.backproject(frame, fields[name], 'mono')
                    scalar = max(float(quality[at]), 1e-12)
                    total[i, half] += scalar*signal
                    support[i, half] += scalar*weight
                if (j+1) % 32 == 0:
                    print('worker', args.worker, 'shard', start, j+1, 'frames',
                          round(time.perf_counter()-started, 1), 'seconds', flush=True)
            metadata = dict(hashes=hashes, config=config, fit_statistics=statistics,
                            elapsed_seconds=time.perf_counter()-started)
            temporary = path.with_suffix('.tmp')
            with temporary.open('xb') as stream:
                np.savez_compressed(stream, positions=positions, signal=total, support=support,
                                    metadata=json.dumps(metadata))
            temporary.replace(path)
            print(path, flush=True)


if __name__ == '__main__':
    main()
