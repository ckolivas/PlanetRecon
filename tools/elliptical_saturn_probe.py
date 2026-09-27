"""Matched real-capture pilot of elliptical reference blur before joint fitting."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from planetrecon.backends.torch_accel import TorchBackend
from planetrecon.export import ExportConfig, export_result
from planetrecon.io.ser import SERSource
from planetrecon.pipeline.circular_align import CircularMultiscaleRegistration
from planetrecon.pipeline.local_align import LocalRegistration
from planetrecon.result import ReconstructionResult, load_snapshot, save_snapshot
from tools.alignment_noise_experiment import read_png
from tools.coherent_saturn_experiment import inputs
from tools.elliptical_blur_probe import EllipticalReference, set_fixed_template, training_mask
from tools.joint_saturn_experiment import CAPTURE, digest
from tools.spline_joint_registration import SplineJointRegistration


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--frames', type=int, default=128)
    args = p.parse_args()
    root, indices, quality, shifts, reference, hashes = inputs()
    if not 8 <= args.frames <= len(indices):
        p.error('Invalid frame count')
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    positions = np.linspace(0, len(indices)-1, args.frames, dtype=int)
    for path in ('tools/elliptical_saturn_probe.py', 'tools/elliptical_blur_probe.py',
                 'tools/joint_registration.py', 'tools/spline_joint_registration.py'):
        hashes[path] = digest(path)
    hashes['capture'] = digest(CAPTURE)
    engine = SplineJointRegistration(reference, CircularMultiscaleRegistration(reference))
    bank = engine.bank
    estimator = EllipticalReference(reference)
    backend = TorchBackend()
    report = dict(frames=args.frames, positions=positions.tolist(), frame_indices=indices[positions].tolist(),
        hashes=hashes, normalization=False, output_filtering=False, production_changed=False, statistics=[])
    sums, coverage = {}, {}
    with SERSource(CAPTURE) as source:
        for n, pos in enumerate(positions):
            frame = source.read_raw(int(indices[pos]))
            matched, ellipse_stats = estimator.fit(frame, shifts[pos], training_mask(engine, shifts[pos]))
            origin = torch.tensor(shifts[pos], device=engine.device)[:, None, None]
            fields = dict(global_only=origin.expand(2, *reference.shape),
                coherent=engine.engine.displacement(LocalRegistration.proxy(frame), shifts[pos], lambda: None))
            stats = dict(position=int(pos), ellipse=ellipse_stats)
            engine.bank = bank
            fields['isotropic'], stats['isotropic'] = engine.fit(frame, shifts[pos], .3, lambda: None)
            set_fixed_template(engine, matched)
            fields['ellipse_estimated'], stats['ellipse_estimated'] = engine.fit(frame, shifts[pos], .3, lambda: None)
            for name, field in fields.items():
                signal, support, *_ = backend.backproject(frame, field, 'mono')
                weight = max(float(quality[pos]), 1e-12)
                sums.setdefault(name, np.zeros_like(reference))[:] += weight*signal
                coverage.setdefault(name, np.zeros_like(reference))[:] += weight*support
                stats.setdefault('residual_vector_rms_px', {})[name] = float(
                    (field-origin)[:, engine.mask.bool()].square().sum(0).mean().sqrt())
            report['statistics'].append(stats)
            if (n+1) % 16 == 0:
                print(n+1, 'of', args.frames, 'frames complete', flush=True)
    if digest(CAPTURE) != hashes['capture']:
        raise ValueError('Capture changed')
    original = load_snapshot(root/'local.npz')
    _, text = read_png(root/'local.png')
    mapping = json.loads(text)['mapping']
    export = ExportConfig('png16', mapping['black'], mapping['white'], 1.)
    for name in sums:
        pixels = np.divide(sums[name], coverage[name], out=np.zeros_like(reference), where=coverage[name]>0)
        result = ReconstructionResult(image=pixels, coverage=coverage[name], validity=coverage[name]>0,
            units=original.units, channel_order='mono', backend='cuda', precision='float64',
            stage='diagnostic', incomplete=False, reference_epoch=original.reference_epoch, n_used=args.frames,
            provenance=dict(diagnostic_only=True, policy=name, hashes=hashes, frame_indices=indices[positions].tolist()))
        save_snapshot(args.out/f'{name}.npz', result)
        export_result(result, args.out/f'{name}.png', export)
    (args.out/'report.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
