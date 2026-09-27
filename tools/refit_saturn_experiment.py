"""Matched Saturn ablation of detector splitting, gating and full-data refitting."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from planetrecon.backends.torch_accel import TorchBackend
from planetrecon.export import export_result, ExportConfig
from planetrecon.io.ser import SERSource
from planetrecon.pipeline.circular_align import CircularMultiscaleRegistration
from planetrecon.pipeline.local_align import LocalRegistration
from planetrecon.result import ReconstructionResult, load_snapshot, save_snapshot
from tools.alignment_noise_experiment import read_png
from tools.coherent_saturn_experiment import inputs, combine_shards
from tools.refit_registration import RefitRegistration, guarded
from tools.coherent_registration import CoherentRegistration

POLICIES = ['coherent','half_average','validated','full_any','full_scaled']


def cached_validation(root, hashes, count):
    cached = [None]*count
    expected_config = dict(spacing=16.,stiffness=.01,patch_average=True,independent_pixel_validation=True)
    for path in sorted(root.glob('shard_*.npz')):
        with np.load(path) as data:
            metadata = json.loads(str(data['metadata']))
            if metadata['config'] != expected_config:
                raise ValueError('Cached validation configuration differs')
            for key in ('reference','indices','quality','global_shifts','model_code'):
                if metadata['hashes'][key] != hashes[key]:
                    raise ValueError('Cached validation inputs differ')
            if metadata['hashes']['validation_code'] != hashes['validated_registration']:
                raise ValueError('Cached validation model differs')
            if not np.issubdtype(data['positions'].dtype,np.integer):
                raise ValueError('Invalid cached validation positions')
            for position,stats in zip(data['positions'],metadata['fit_statistics'],strict=True):
                if position<0 or position>=len(cached) or cached[position] is not None:
                    raise ValueError('Invalid or duplicate cached validation position')
                if stats['accepted_halves'] not in (0,1,2):
                    raise ValueError('Invalid cached validation decision')
                cached[position] = stats
    if any(s is None for s in cached):
        raise ValueError('Incomplete cached validation')
    return cached


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--sample-count', type=int, default=512)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--count', type=int, default=256)
    parser.add_argument('--combine', action='store_true')
    parser.add_argument('--cached-full-refit', action='store_true',
                        help='Reuse verified validation decisions and compute only full-data fits')
    parser.add_argument('--regional',action='store_true',help='Diagnostic spatial motion validation')
    parser.add_argument('--blur-matched',action='store_true',help='Diagnostic inferred reference blur')
    args = parser.parse_args()
    root, indices, quality, shifts, reference, hashes = inputs()
    if not 2 <= args.sample_count <= len(indices):
        parser.error('sample count must be between 2 and the selected frame count')
    selected = np.linspace(0,len(indices)-1,args.sample_count,dtype=int)
    hashes['sample_positions'] = hashlib.sha256(selected.tobytes()).hexdigest()
    for name in ('validated_registration','refit_registration'):
        hashes[name] = hashlib.sha256(Path(f'tools/{name}.py').read_bytes()).hexdigest()
    if sum((args.regional, args.cached_full_refit, args.blur_matched)) > 1:
        parser.error('regional, cached full refit and blur matching are separate experiments')
    policies = ['coherent','full_any','full_scaled'] if args.cached_full_refit else POLICIES
    if args.regional:
        policies = ['coherent','full_any','regional_any','regional_both']
        hashes['regional_registration'] = hashlib.sha256(Path('tools/regional_registration.py').read_bytes()).hexdigest()
    if args.blur_matched:
        policies = ['coherent','matched_global','matched_local']
        hashes['blur_matched_registration'] = hashlib.sha256(Path('tools/blur_matched_registration.py').read_bytes()).hexdigest()
    cached = None
    if args.cached_full_refit:
        cached = cached_validation(Path('out/saturn-validated-field'),hashes,len(indices))
        hashes['cached_decisions'] = hashlib.sha256(json.dumps(cached,sort_keys=True).encode()).hexdigest()
    config = dict(policies=policies, sample_count=args.sample_count, spacing=16., stiffness=.01)
    shape = (len(policies),*reference.shape)
    if args.combine:
        total,support,stats = combine_shards(sorted(args.out.glob('shard_*.npz')),
                                             len(selected),shape,hashes,config)
        pixels = np.divide(total,support,out=np.zeros_like(total),where=support>0)
        original = load_snapshot(root/'local.npz')
        _,text = read_png(root/'local.png')
        mapping = json.loads(text)['mapping']
        export = ExportConfig('png16',mapping['black'],mapping['white'],1.)
        for i,name in enumerate(policies):
            result = ReconstructionResult(image=pixels[i],coverage=support[i],validity=support[i]>0,
                units=original.units,channel_order='mono',backend='cuda',precision='float64',
                stage='diagnostic',incomplete=False,reference_epoch=original.reference_epoch,n_used=len(selected),
                provenance={'diagnostic_only':True,'policy':name,'config':config,'hashes':hashes,
                            'frame_indices':indices[selected].tolist()})
            save_snapshot(args.out/f'{name}.npz',result)
            export_result(result,args.out/f'{name}.png',export)
        report = dict(n_used=len(selected),sample_positions=selected.tolist(),
            frame_indices=indices[selected].tolist(),hashes=hashes,config=config,production_changed=False,
            brightness_normalization=False,output_filtering=False)
        if args.blur_matched:
            report['selected_blur_counts'] = {pose:{str(sigma):int(sum(s[pose+'_sigma']==sigma for s in stats))
                for sigma in np.arange(0.,2.01,.25)} for pose in ('global','local')}
        else:
            report['accepted_half_counts'] = {str(n):sum(s['accepted_halves']==n for s in stats) for n in range(3)}
        if args.regional:
            report['regional_guard_rejections'] = {key:sum(not s[key+'_guard_accepted'] for s in stats)
                                                  for key in ('regional_any','regional_both')}
        (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print('Combined',len(selected),'frames across',len(policies),'policies')
        return
    if args.start<0 or args.start>=len(selected) or args.count<1:
        parser.error('invalid shard range')
    args.out.mkdir(parents=True,exist_ok=True)
    path = args.out/f'shard_{args.start:05d}.npz'
    if path.exists():
        raise FileExistsError(path)
    positions = np.arange(args.start,min(args.start+args.count,len(selected)))
    torch.set_num_threads(4)
    matcher = CircularMultiscaleRegistration(reference)
    if args.blur_matched:
        from tools.blur_matched_registration import BlurMatchedRegistration
        engine = BlurMatchedRegistration(reference,matcher)
    elif args.regional:
        from tools.regional_registration import RegionalRegistration
        engine = RegionalRegistration(reference,matcher)
    else:
        engine = (CoherentRegistration(matcher,spacing=16.,stiffness=.01,patch_average=True)
                  if cached is not None else RefitRegistration(reference,matcher))
    backend = TorchBackend()
    total,support = np.zeros(shape),np.zeros(shape)
    started = time.perf_counter()
    statistics = []
    with SERSource('2024-09-27-1154_3-CK-R-Sat.ser') as source:
        for j,position in enumerate(positions):
            p = selected[position]
            frame = source.read_raw(int(indices[p]))
            if cached is None:
                fields = engine.variants(frame,shifts[p],lambda:None)
                statistics.append(engine.stats[-1])
            else:
                coherent = engine.displacement(LocalRegistration.proxy(frame),shifts[p],lambda:None)
                origin = torch.as_tensor(shifts[p],device=engine.device)[:,None,None]
                stats = dict(cached[p],full_fit=engine.stats[-1])
                passed = stats['accepted_halves'] if stats['field_guard_accepted'] else 0
                fields = dict(coherent=coherent,
                    full_any=coherent if passed else origin.expand_as(coherent).clone(),
                    full_scaled=guarded(origin+(coherent-origin)*(passed/2.),origin))
                statistics.append(stats)
            for i,name in enumerate(policies):
                signal,weight,*_ = backend.backproject(frame,fields[name],'mono')
                scalar = max(float(quality[p]),1e-12)
                total[i] += scalar*signal
                support[i] += scalar*weight
            if (j+1)%64==0:
                print(args.start,j+1,'frames',round(time.perf_counter()-started,1),'seconds',flush=True)
    metadata = dict(hashes=hashes,config=config,fit_statistics=statistics,
                    elapsed_seconds=time.perf_counter()-started)
    np.savez_compressed(path,positions=positions,signal=total,support=support,metadata=json.dumps(metadata))
    print(path,flush=True)


if __name__ == '__main__':
    main()
