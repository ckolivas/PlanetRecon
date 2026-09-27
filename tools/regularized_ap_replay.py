"""Full-count paired replay of baseline and screened field constraints."""
import argparse
import json
from pathlib import Path
from dataclasses import replace

import numpy as np
import torch

from planetrecon.backends.torch_accel import TorchBackend
from planetrecon.io.ser import SERSource
from planetrecon.pipeline.local_align import LocalRegistration
from planetrecon.result import load_snapshot, save_snapshot
from planetrecon.export import ExportConfig, export_result
from tools.alignment_noise_experiment import read_png
from tools.ap_stability_screen import hashes
from tools.coherent_saturn_experiment import inputs
from tools.joint_saturn_experiment import CAPTURE, digest, capture_identity, combine
from tools.regularized_ap_fit import RegularizedAPFit, POLICIES


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--prepare', action='store_true')
    p.add_argument('--combine', action='store_true')
    p.add_argument('--worker', type=int, default=0)
    p.add_argument('--workers', type=int, default=1)
    args = p.parse_args()
    if args.prepare and args.combine or args.workers < 1 or not 0 <= args.worker < args.workers:
        p.error('Invalid operation/worker')
    screen_path = Path('out/ap-stability-screen/report.json')
    screen = json.loads(screen_path.read_text())
    selected = screen['selection']['selected']
    if selected is None or screen['hashes'] != hashes():
        raise ValueError('No unchanged passing candidate')
    root, indices, quality, shifts, reference, input_hashes = inputs()
    versions = dict(input_hashes, **hashes())
    versions['tools/regularized_ap_replay.py'] = digest(__file__)
    versions['screen_report'] = digest(screen_path)
    names = ['baseline', selected]
    config = dict(names=names, policies={name: list(POLICIES[name]) for name in names},
                  shard_size=128, half_split='selection_position_modulo_two')
    manifest_path = args.out/'manifest.json'
    if args.prepare:
        args.out.mkdir(parents=True, exist_ok=False)
        identity = capture_identity()
        capture_hash = digest(CAPTURE)
        if identity != capture_identity() or capture_hash != screen['capture_sha256']:
            raise ValueError('Capture changed')
        manifest = dict(hashes=versions, config=config, capture=identity,
            capture_sha256=capture_hash, frame_indices=indices.tolist())
        manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
        print('Prepared', len(indices), 'frames for', selected, flush=True)
        return
    manifest = json.loads(manifest_path.read_text())
    if (manifest['hashes'] != versions or manifest['config'] != config or
            manifest['capture'] != capture_identity() or manifest['frame_indices'] != indices.tolist()):
        raise ValueError('Prepared experiment changed')
    versions['capture'] = manifest['capture_sha256']
    shape = (len(names), 2, *reference.shape)
    if args.combine:
        total, support, stats = combine(args.out, len(indices), shape, versions, config)
        if digest(CAPTURE) != versions['capture']:
            raise ValueError('Capture changed during replay')
        stats.sort(key=lambda row: row['position'])
        prior = load_snapshot('out/saturn-coherent-field/coherent.npz')
        baseline = np.divide(total[0].sum(0), support[0].sum(0), out=np.zeros_like(reference), where=support[0].sum(0)>0)
        np.testing.assert_allclose(baseline, prior.image, atol=1e-10, rtol=0)
        _, text = read_png(root/'local.png')
        mapping = json.loads(text)['mapping']
        export = ExportConfig('png16', mapping['black'], mapping['white'], 1.)
        for name in names:
            for suffix in ('.npz', '.png', '_half0.png', '_half1.png'):
                if (args.out/(name+suffix)).exists():
                    raise FileExistsError(name+suffix)
        for i, name in enumerate(names):
            coverage = support[i].sum(0)
            image = np.divide(total[i].sum(0), coverage, out=np.zeros_like(reference), where=coverage>0)
            result = replace(prior, image=image, coverage=coverage, validity=coverage>0,
                provenance=dict(diagnostic_only=True, method=name, hashes=versions, config=config, frame_indices=indices.tolist()))
            save_snapshot(args.out/f'{name}.npz', result)
            export_result(result, args.out/f'{name}.png', export)
            for half in (0, 1):
                result.coverage = support[i, half]
                result.validity = result.coverage>0
                result.image = np.divide(total[i, half], result.coverage, out=np.zeros_like(reference), where=result.validity)
                result.n_used = int(np.count_nonzero(np.arange(len(indices)) % 2 == half))
                result.provenance = dict(result.provenance, half=half)
                export_result(result, args.out/f'{name}_half{half}.png', export)
        np.savez_compressed(args.out/'halves.npz', signal=total, support=support)
        report = dict(n_used=len(indices), hashes=versions, config=config, statistics=stats,
            halves_sha256=digest(args.out/'halves.npz'), baseline_max_difference_adu=float(np.max(abs(baseline-prior.image))),
            output_filtering=False, normalization=False, production_changed=False)
        (args.out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
        print('Combined', len(indices), 'frames; baseline parity', report['baseline_max_difference_adu'], flush=True)
        return
    torch.set_num_threads(4)
    model = RegularizedAPFit(reference, policies=names)
    backend = TorchBackend()
    with SERSource(CAPTURE) as source:
        for start in range(args.worker*128, len(indices), args.workers*128):
            path = args.out/f'shard_{start:05d}.npz'
            if path.exists():
                raise FileExistsError(path)
            positions = np.arange(start, min(start+128, len(indices)))
            total, support, statistics = np.zeros(shape), np.zeros(shape), []
            for pos in positions:
                frame = source.read_raw(int(indices[pos]))
                observations = model.observations(LocalRegistration.proxy(frame), shifts[pos])
                row = dict(position=int(pos), frame_index=int(indices[pos]), fits={})
                for i, name in enumerate(names):
                    field, row['fits'][name] = model.field(name, observations, shifts[pos])
                    signal, coverage, *_ = backend.backproject(frame, field, 'mono')
                    weight = max(float(quality[pos]), 1e-12)
                    total[i, pos % 2] += weight*signal
                    support[i, pos % 2] += weight*coverage
                statistics.append(row)
            temporary = path.with_suffix('.tmp')
            with temporary.open('xb') as stream:
                np.savez_compressed(stream, positions=positions, signal=total, support=support,
                    metadata=json.dumps(dict(hashes=versions, config=config, fit_statistics=statistics)))
            temporary.replace(path)
            print('worker', args.worker, 'completed', start, len(positions), flush=True)


if __name__ == '__main__':
    main()
