"""Matched Saturn replay with four disjoint native references and their ensemble."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from planetrecon.backends.torch_accel import TorchBackend
from planetrecon.io.ser import SERSource
from planetrecon.pipeline.local_align import LocalRegistration
from planetrecon.result import ReconstructionResult, load_snapshot, save_snapshot
from planetrecon.export import ExportConfig, export_result
from tools.alignment_noise_experiment import read_png
from tools.ap_fractional_phase import hashes as matching_hashes
from tools.coherent_saturn_experiment import inputs
from tools.joint_saturn_experiment import CAPTURE, digest, capture_identity, combine
from tools.regularized_ap_fit import RegularizedAPFit

from tools.native_reference_ensemble import NAMES, cohorts, build_references, NativeReferenceStates, mean_field


def identity():
    root, indices, quality, shifts, reference, versions = inputs()
    selected = np.linspace(0, len(indices)-1, 512, dtype=int)
    versions.update(matching_hashes())
    versions['replay_source'] = digest(__file__)
    versions['reference_source'] = digest('tools/native_reference_ensemble.py')
    versions['sample_positions'] = hashlib.sha256(selected.tobytes()).hexdigest()
    config = dict(names=NAMES, sample_count=len(selected), shard_size=16,
                  half_split='pilot_position_modulo_two', matching_sampler='bilinear',
                  final_sampler='unchanged_raw_backprojection', reference_seed=270927,
                  reference_cohorts=4, reference_cohort_size=64, fixed_geometry=True)
    return root, indices, quality, shifts, reference, selected, versions, config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    operation = p.add_mutually_exclusive_group()
    operation.add_argument('--prepare', action='store_true')
    operation.add_argument('--combine', action='store_true')
    p.add_argument('--worker', type=int, default=0)
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    if args.workers < 1 or not 0 <= args.worker < args.workers:
        p.error('Invalid worker')
    root, indices, quality, shifts, reference, selected, versions, config = identity()
    manifest_path = args.out/'manifest.json'
    if args.prepare:
        args.out.mkdir(parents=True, exist_ok=False)
        capture = capture_identity()
        capture_hash = digest(CAPTURE)
        if capture != capture_identity():
            raise ValueError('Capture changed')
        manifest = dict(hashes=versions, config=config, capture=capture,
            capture_sha256=capture_hash, selection_positions=selected.tolist(),
            frame_indices=indices[selected].tolist())
        members = cohorts(indices, quality, selected)
        with SERSource(CAPTURE) as source:
            references, sums, supports = build_references(source, indices, shifts, members, reference)
        np.savez_compressed(args.out/'references.npz', **references, sums=sums, supports=supports, members=members)
        manifest['references_sha256'] = digest(args.out/'references.npz')
        manifest['reference_frame_indices'] = indices[members].tolist()
        manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
        print('Prepared', len(selected), 'matched frames', flush=True)
        return
    manifest = json.loads(manifest_path.read_text())
    if (manifest['hashes'] != versions or manifest['config'] != config or
            manifest['capture'] != capture_identity() or
            manifest['selection_positions'] != selected.tolist() or
            manifest['frame_indices'] != indices[selected].tolist()):
        raise ValueError('Prepared experiment changed')
    versions['capture'] = manifest['capture_sha256']
    versions['references'] = digest(args.out/'references.npz')
    if versions['references'] != manifest['references_sha256']:
        raise ValueError('Reference arrays changed')
    with np.load(args.out/'references.npz') as data:
        references = {name:data[name] for name in NAMES[:-1]}
        members = data['members']
    np.testing.assert_array_equal(members, cohorts(indices, quality, selected))
    if manifest['reference_frame_indices'] != indices[members].tolist():
        raise ValueError('Reference membership changed')
    shape = (len(NAMES), 2, *reference.shape)
    if args.combine:
        total, support, stats = combine(args.out, len(selected), shape, versions, config)
        if digest(CAPTURE) != versions['capture']:
            raise ValueError('Capture changed during replay')
        stats.sort(key=lambda row: row['position'])
        if ([r['frame_index'] for r in stats] != indices[selected].tolist() or
                [r['selection_position'] for r in stats] != selected.tolist()):
            raise ValueError('Frame association changed')
        baseline = np.divide(total[0].sum(0),support[0].sum(0),out=np.zeros_like(reference),where=support[0].sum(0)>0)
        previous = load_snapshot('out/saturn-peak-refinement/baseline.npz').image
        np.testing.assert_allclose(baseline, previous, atol=1e-10, rtol=0)
        parity = float(np.max(abs(baseline-previous)))
        original = load_snapshot(root/'local.npz')
        _, text = read_png(root/'local.png')
        mapping = json.loads(text)['mapping']
        export = ExportConfig('png16', mapping['black'], mapping['white'], 1.)
        for name in NAMES:
            for suffix in ('.npz', '.png', '_half0.png', '_half1.png'):
                if (args.out/(name+suffix)).exists():
                    raise FileExistsError(name+suffix)
        for i, name in enumerate(NAMES):
            coverage = support[i].sum(0)
            image = np.divide(total[i].sum(0), coverage, out=np.zeros_like(coverage), where=coverage>0)
            result = ReconstructionResult(image=image, coverage=coverage, validity=coverage>0,
                units=original.units, channel_order='mono', backend='cuda', precision='float64',
                stage='diagnostic', incomplete=False, reference_epoch=original.reference_epoch,
                n_used=len(selected), provenance=dict(diagnostic_only=True, method=name,
                    hashes=versions, config=config, frame_indices=indices[selected].tolist()))
            save_snapshot(args.out/f'{name}.npz', result)
            export_result(result, args.out/f'{name}.png', export)
            for half in (0, 1):
                result.coverage = support[i, half]
                result.validity = result.coverage>0
                result.image = np.divide(total[i, half], result.coverage, out=np.zeros_like(coverage), where=result.validity)
                result.n_used = int(np.count_nonzero(np.arange(len(selected)) % 2 == half))
                result.provenance = dict(result.provenance, half=half)
                export_result(result, args.out/f'{name}_half{half}.png', export)
        np.savez_compressed(args.out/'halves.npz', signal=total, support=support)
        report = dict(n_used=len(selected), hashes=versions, config=config, statistics=stats,
            halves_sha256=digest(args.out/'halves.npz'), baseline_stack_parity_max_adu=parity,
            output_filtering=False, normalization=False, production_changed=False)
        (args.out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
        print('Combined', len(selected), 'matched frames', flush=True)
        return
    torch.set_num_threads(4)
    model = RegularizedAPFit(reference, policies=['baseline'])
    backend = TorchBackend()
    states = NativeReferenceStates(model.engine, references)
    with SERSource(CAPTURE) as source:
        if source.color_mode() != 'mono' or source.frame_shape() != reference.shape:
            raise ValueError('Unexpected capture geometry or colour')
        for start in range(args.worker*16, len(selected), args.workers*16):
            path = args.out/f'shard_{start:05d}.npz'
            if path.exists():
                if args.resume:
                    # Full metadata, shape and finite-value validation is repeated by combine.
                    with np.load(path) as data:
                        meta = json.loads(str(data['metadata']))
                        if meta['hashes'] != versions or meta['config'] != config:
                            raise ValueError('Cannot resume a different experiment')
                    continue
                raise FileExistsError(path)
            positions = np.arange(start, min(start+16, len(selected)))
            total, support, statistics = np.zeros(shape), np.zeros(shape), []
            started = time.perf_counter()
            for pos in positions:
                at = selected[pos]
                frame = source.read_raw(int(indices[at]))
                proxy = LocalRegistration.proxy(frame)
                row = dict(position=int(pos), selection_position=int(at), frame_index=int(indices[at]), fits={})
                native_fields = []
                for i,name in enumerate(NAMES):
                    if name == 'field_mean':
                        field,fit = mean_field(native_fields, shifts[at])
                        mask = torch.as_tensor(reference > reference.max()*.08, device=field.device)
                        row['reference_field_spread_px'] = float((torch.stack(native_fields)-field).square().sum(1)[:,mask].mean().sqrt())
                    else:
                        states.select(name)
                        obs = model.observations(proxy, shifts[at])
                        field,fit = model.field('baseline', obs, shifts[at])
                        fit['accepted_aps'] = int(np.count_nonzero(obs[1]))
                        if name.startswith('ref_'):
                            native_fields.append(field)
                    row['fits'][name] = fit
                    if name == 'baseline' and pos == start:
                        original = model.engine.displacement(proxy,shifts[at],lambda: None)
                        torch.testing.assert_close(field,original,atol=1e-10,rtol=0)
                        row['baseline_parity_max_px'] = float((field-original).abs().max())
                    signal,coverage,*_ = backend.backproject(frame,field,'mono')
                    weight = max(float(quality[at]),1e-12)
                    total[i,pos%2] += weight*signal
                    support[i,pos%2] += weight*coverage
                statistics.append(row)
            temporary = path.with_suffix('.tmp')
            with temporary.open('xb') as stream:
                np.savez_compressed(stream, positions=positions, signal=total, support=support,
                    metadata=json.dumps(dict(hashes=versions, config=config, fit_statistics=statistics)))
            temporary.replace(path)
            print('worker', args.worker, 'completed', start, len(positions),
                  round(time.perf_counter()-started, 1), 'seconds', flush=True)


if __name__ == '__main__':
    main()
