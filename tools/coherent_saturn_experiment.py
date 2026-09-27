"""Replay fixed Saturn frames with the experimental coherent displacement fit.

Write bounded shards of weighted sums, then combine exactly once. Existing
reference, scalar quality weights and global shifts are held fixed.
"""
import argparse
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.ndimage import gaussian_filter
import torch

from planetrecon.backends.torch_accel import TorchBackend
from planetrecon.io.ser import SERSource
from planetrecon.pipeline.circular_align import CircularMultiscaleRegistration
from planetrecon.pipeline.local_align import LocalRegistration
from planetrecon.result import load_snapshot, save_snapshot
from planetrecon.export import export_result, ExportConfig
from tools.alignment_noise_experiment import read_png
from tools.coherent_registration import CoherentRegistration


def inputs():
    root = Path('out/saturn-controlled-alignment')
    with np.load(root/'diagnostics.npz') as data:
        indices, quality, shifts = [data[k].copy() for k in ('indices','quality','global_shifts')]
    reference = np.load(root/'reference.npy')
    hashes = {k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in
              [('reference',reference),('indices',indices),('quality',quality),('global_shifts',shifts)]}
    hashes['model_code'] = hashlib.sha256(Path('tools/coherent_registration.py').read_bytes()).hexdigest()
    return root, indices, quality, shifts, reference, hashes


def combine_shards(paths, count, shape, hashes, config):
    total, support = np.zeros(shape), np.zeros(shape)
    seen = np.zeros(count, dtype=int)
    stats = []
    for path in paths:
        with np.load(path) as data:
            metadata = json.loads(str(data['metadata']))
            if metadata['hashes'] != hashes or metadata['config'] != config:
                raise ValueError('shard input/model identity mismatch')
            positions = data['positions']
            if not np.issubdtype(positions.dtype,np.integer) or not np.all((positions>=0)&(positions<count)):
                raise ValueError('invalid shard positions')
            if len(metadata['fit_statistics']) != len(positions):
                raise ValueError('shard statistics/frame count mismatch')
            np.add.at(seen,positions,1)
            for name in ('signal','support'):
                if data[name].shape != shape or not np.isfinite(data[name]).all():
                    raise ValueError('invalid shard accumulator')
            if np.any(data['support']<0):
                raise ValueError('negative sample support')
            total += data['signal']
            support += data['support']
            stats.extend(metadata['fit_statistics'])
    if not np.all(seen==1):
        raise ValueError(f'Need each selected frame once: missing={np.count_nonzero(seen==0)}, repeated={np.count_nonzero(seen>1)}')
    return total,support,stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--count', type=int, default=512)
    parser.add_argument('--spacing', type=float, default=16.)
    parser.add_argument('--stiffness', type=float, default=.01)
    parser.add_argument('--patch-average', action='store_true')
    parser.add_argument('--validate-pixels', action='store_true',
                        help='Validate area-aware coherent motion against disjoint detector samples')
    parser.add_argument('--combine', action='store_true')
    args = parser.parse_args()
    root, indices, quality, shifts, reference, hashes = inputs()
    config = {'spacing':args.spacing,'stiffness':args.stiffness,'patch_average':args.patch_average}
    if args.validate_pixels:
        config.update(patch_average=True, independent_pixel_validation=True)
        hashes['validation_code'] = hashlib.sha256(Path('tools/validated_registration.py').read_bytes()).hexdigest()
    if args.combine:
        paths = sorted(args.out.glob('shard_*.npz'))
        total,support,stats = combine_shards(paths,len(indices),reference.shape,hashes,config)
        original = load_snapshot(root/'local.npz')
        image = np.divide(total, support, out=np.zeros_like(total), where=support>0)
        provenance = deepcopy(original.provenance)
        provenance['coherent_fit'] = {'diagnostic_only': True, 'config': config, 'hashes':hashes}
        result = replace(original,image=image,coverage=support,validity=support>0,provenance=provenance)
        save_snapshot(args.out/'coherent.npz',result)
        _,text = read_png(root/'local.png')
        mapping = json.loads(text)['mapping']
        export = ExportConfig('png16',mapping['black'],mapping['white'],1.)
        export_result(result,args.out/'coherent.png',export)
        # Only a separately named comparison copy receives the measured AS response.
        comparison = replace(result,image=gaussian_filter(image,1.,radius=3),provenance=deepcopy(provenance))
        comparison.provenance['comparison_smoothing'] = {'sigma_px':1.,'radius_px':3,'diagnostic_only':True}
        export_result(comparison,args.out/'coherent_as_transfer.png',export)
        report = {'n_used':len(indices),'hashes':hashes,'config':config,
                  'production_changed':False,'brightness_normalization':False,
                  'shards':[str(p) for p in paths], 'rejected_fields':sum(not s['field_guard_accepted'] for s in stats),
                  'insufficient_points':sum(s['fallback'] for s in stats),
                  'median_points':float(np.median([s['points'] for s in stats]))}
        if args.validate_pixels:
            report['accepted_half_counts'] = {str(n):sum(s['accepted_halves']==n for s in stats) for n in range(3)}
        (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(report,flush=True)
        return
    if args.start<0 or args.count<1 or args.start>=len(indices):
        parser.error('invalid shard range')
    args.out.mkdir(parents=True,exist_ok=True)
    path = args.out/f'shard_{args.start:05d}.npz'
    if path.exists():
        raise FileExistsError(path)
    torch.set_num_threads(4)
    matcher = CircularMultiscaleRegistration(reference)
    if args.validate_pixels:
        from tools.validated_registration import ValidatedRegistration
        engine = ValidatedRegistration(reference,matcher,spacing=args.spacing,stiffness=args.stiffness)
    else:
        engine = CoherentRegistration(matcher,spacing=args.spacing,
                                      stiffness=args.stiffness,patch_average=args.patch_average)
    backend = TorchBackend()
    total = np.zeros_like(reference)
    support = np.zeros_like(reference)
    positions = np.arange(args.start,min(args.start+args.count,len(indices)))
    started = time.perf_counter()
    with SERSource('2024-09-27-1154_3-CK-R-Sat.ser') as source:
        for j,position in enumerate(positions):
            frame = source.read_raw(int(indices[position]))
            if args.validate_pixels:
                field = engine.displacement_raw(frame,shifts[position],lambda:None)
            else:
                field = engine.displacement(LocalRegistration.proxy(frame),shifts[position],lambda:None)
            signal,weight,*_ = backend.backproject(frame,field,'mono')
            scalar = max(float(quality[position]),1e-12)
            total += scalar*signal
            support += scalar*weight
            if (j+1)%128==0:
                print(args.start,j+1,len(positions),round(time.perf_counter()-started,1),'seconds',flush=True)
    metadata = {'hashes':hashes,'config':config,'fit_statistics':engine.stats,
                'elapsed_seconds':time.perf_counter()-started}
    np.savez_compressed(path,positions=positions,signal=total,support=support,metadata=json.dumps(metadata))
    print(path, metadata['elapsed_seconds'],flush=True)


if __name__=='__main__':
    main()
