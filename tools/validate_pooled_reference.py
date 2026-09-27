"""Frozen pooled reference on two unused 512-frame sets and two local matchers."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from planetrecon.backends.torch_accel import TorchBackend
from planetrecon.backends.torch_circular import TorchCircularRegistration
from planetrecon.io.ser import SERSource
from planetrecon.pipeline.local_align import LocalRegistration
from planetrecon.result import ReconstructionResult, load_snapshot, save_snapshot
from planetrecon.export import ExportConfig, export_result
from tools.alignment_noise_experiment import read_png
from tools.ap_fractional_phase import hashes as matching_hashes
from tools.coherent_saturn_experiment import inputs
from tools.fractional_ap_trace import observations
from tools.local_warp_trace import trace
from tools.joint_saturn_experiment import CAPTURE, digest, capture_identity, combine
from tools.regularized_ap_fit import RegularizedAPFit
from tools.native_reference_ensemble import NativeReferenceStates

PRIOR = Path('out/saturn-native-references')
NAMES = ['production_base','production_pool','coherent_base','coherent_pool']


def unused_positions(indices, previous, native, historical, count=1024):
    excluded=np.unique(np.concatenate((np.asarray(previous).ravel(),np.asarray(native).ravel(),
        np.flatnonzero(np.isin(indices,historical)))))
    available=np.setdiff1d(np.arange(len(indices)),excluded)
    if len(available)<count:
        raise ValueError('Insufficient unused frames')
    return available[np.linspace(0,len(available)-1,count,dtype=int)]


def identity():
    root,indices,quality,shifts,reference,versions=inputs()
    prior=json.loads((PRIOR/'manifest.json').read_text())
    with np.load(PRIOR/'references.npz') as data:
        native=data['members']
    old=load_snapshot(root/'local.npz').provenance['local_alignment']['template_candidates']
    selected=unused_positions(indices,prior['selection_positions'],native,old)
    versions.update(matching_hashes())
    versions.update(replay_source=digest(__file__),reference_source=digest('tools/native_reference_ensemble.py'),
        prior_manifest=digest(PRIOR/'manifest.json'),references=digest(PRIOR/'references.npz'))
    if versions['references']!=prior['references_sha256']:
        raise ValueError('Frozen references changed')
    versions['sample_positions']=hashlib.sha256(selected.tobytes()).hexdigest()
    config=dict(names=NAMES,sample_count=1024,cohort_count=2,cohort_frames=512,shard_size=32,
        cohort='position_modulo_two',half='floor_position_over_two_modulo_two',
        fixed_reference=True,fixed_geometry=True,normalization=False,output_filtering=False)
    return root,indices,quality,shifts,reference,selected,versions,config


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    op=p.add_mutually_exclusive_group()
    op.add_argument('--prepare',action='store_true');op.add_argument('--combine',action='store_true')
    p.add_argument('--worker',type=int,default=0);p.add_argument('--workers',type=int,default=1)
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    if args.workers<1 or not 0<=args.worker<args.workers:
        p.error('Invalid worker')
    root,indices,quality,shifts,reference,selected,versions,config=identity()
    path=args.out/'manifest.json'
    if args.prepare:
        args.out.mkdir(parents=True,exist_ok=False)
        capture=capture_identity(); capture_hash=digest(CAPTURE)
        if capture!=capture_identity():
            raise ValueError('Capture changed')
        manifest=dict(hashes=versions,config=config,capture=capture,capture_sha256=capture_hash,
            selection_positions=selected.tolist(),frame_indices=indices[selected].tolist())
        path.write_text(json.dumps(manifest,indent=2)+'\n')
        print('Prepared two unused 512-frame sets',flush=True)
        return
    manifest=json.loads(path.read_text())
    if (manifest['hashes']!=versions or manifest['config']!=config or manifest['capture']!=capture_identity()
        or manifest['selection_positions']!=selected.tolist() or manifest['frame_indices']!=indices[selected].tolist()):
        raise ValueError('Prepared inputs changed')
    versions['capture']=manifest['capture_sha256']
    shape=(4,4,*reference.shape)
    if args.combine:
        total,support,stats=combine(args.out,1024,shape,versions,config)
        if digest(CAPTURE)!=versions['capture']:
            raise ValueError('Capture changed during replay')
        stats.sort(key=lambda r:r['position'])
        if [r['frame_index'] for r in stats]!=indices[selected].tolist():
            raise ValueError('Frame association mismatch')
        original=load_snapshot(root/'local.npz')
        _,text=read_png(root/'local.png'); mapping=json.loads(text)['mapping']
        export=ExportConfig('png16',mapping['black'],mapping['white'],1.)
        for cohort in (0,1):
            for name in NAMES:
                for suffix in ('.npz','.png','_half0.png','_half1.png'):
                    if (args.out/(f'c{cohort}_{name}'+suffix)).exists():
                        raise FileExistsError(name+suffix)
        for cohort in (0,1):
            slots=[cohort,cohort+2]
            ids=indices[selected[cohort::2]]
            for i,name in enumerate(NAMES):
                coverage=support[i,slots].sum(0)
                image=np.divide(total[i,slots].sum(0),coverage,out=np.zeros_like(coverage),where=coverage>0)
                provenance=dict(diagnostic_only=True,method=name,cohort=cohort,hashes=versions,config=config,frame_indices=ids.tolist())
                result=ReconstructionResult(image=image,coverage=coverage,validity=coverage>0,units=original.units,
                    channel_order='mono',backend='cuda',precision='float64',stage='diagnostic',incomplete=False,
                    reference_epoch=original.reference_epoch,n_used=512,provenance=provenance)
                stem=f'c{cohort}_{name}'
                save_snapshot(args.out/(stem+'.npz'),result);export_result(result,args.out/(stem+'.png'),export)
                for half,slot in enumerate(slots):
                    result.coverage=support[i,slot];result.validity=result.coverage>0
                    result.image=np.divide(total[i,slot],result.coverage,out=np.zeros_like(coverage),where=result.validity)
                    result.n_used=256
                    result.provenance=dict(provenance,half=half,frame_indices=ids[half::2].tolist())
                    export_result(result,args.out/f'{stem}_half{half}.png',export)
        np.savez_compressed(args.out/'halves.npz',signal=total,support=support)
        report=dict(n_used=1024,hashes=versions,config=config,statistics=stats,halves_sha256=digest(args.out/'halves.npz'),
            production_changed=False,normalization=False,output_filtering=False)
        (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print('Combined two complete 512-frame sets',flush=True)
        return
    torch.set_num_threads(4)
    model=RegularizedAPFit(reference,policies=['baseline'])
    with np.load(PRIOR/'references.npz') as data:
        references={name:data[name] for name in ('baseline','pooled')}
    states=NativeReferenceStates(model.engine,references);backend=TorchBackend()
    with SERSource(CAPTURE) as source:
        if source.color_mode()!='mono' or source.frame_shape()!=reference.shape:
            raise ValueError('Unexpected capture')
        for start in range(args.worker*32,1024,args.workers*32):
            path=args.out/f'shard_{start:05d}.npz'
            if path.exists():
                if args.resume:
                    with np.load(path) as d:
                        meta=json.loads(str(d['metadata']))
                        if meta['hashes']!=versions or meta['config']!=config:
                            raise ValueError('Changed checkpoint')
                    continue
                raise FileExistsError(path)
            positions=np.arange(start,start+32)
            total,support,rows=np.zeros(shape),np.zeros(shape),[]
            started=time.perf_counter()
            for pos in positions:
                at=selected[pos];frame=source.read_raw(int(indices[at]));proxy=LocalRegistration.proxy(frame)
                row=dict(position=int(pos),selection_position=int(at),frame_index=int(indices[at]),fits={},parity={})
                fields={}
                for state,suffix in [('baseline','base'),('pooled','pool')]:
                    states.select(state)
                    stages,records=trace(model.engine,proxy,shifts[at])
                    fields['production_'+suffix]=stages[-1]
                    obs=observations(model.engine,stages,records,shifts[at])
                    fields['coherent_'+suffix],row['fits']['coherent_'+suffix]=model.field('baseline',obs,shifts[at])
                    row['fits']['production_'+suffix]=dict(stage_acceptance=[r['accepted'] for r in records])
                    if pos==start:
                        expected=TorchCircularRegistration.displacement(model.engine,proxy,shifts[at],lambda:None)
                        torch.testing.assert_close(fields['production_'+suffix],expected,atol=1e-10,rtol=0)
                        row['parity']['production_'+suffix]=float((fields['production_'+suffix]-expected).abs().max())
                        expected=model.engine.displacement(proxy,shifts[at],lambda:None)
                        torch.testing.assert_close(fields['coherent_'+suffix],expected,atol=1e-10,rtol=0)
                        row['parity']['coherent_'+suffix]=float((fields['coherent_'+suffix]-expected).abs().max())
                slot=int(pos%4)
                for i,name in enumerate(NAMES):
                    signal,coverage,*_=backend.backproject(frame,fields[name],'mono')
                    weight=max(float(quality[at]),1e-12)
                    total[i,slot]+=weight*signal;support[i,slot]+=weight*coverage
                rows.append(row)
            temporary=path.with_suffix('.tmp')
            with temporary.open('xb') as stream:
                np.savez_compressed(stream,positions=positions,signal=total,support=support,
                    metadata=json.dumps(dict(hashes=versions,config=config,fit_statistics=rows)))
            temporary.replace(path)
            print('worker',args.worker,'completed',start,round(time.perf_counter()-started,1),'seconds',flush=True)


if __name__=='__main__':
    main()
