"""Fresh-noise and unused-phase check of peak refinement without cubic sampling."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from planetrecon.backends.torch_circular import sample
from planetrecon.pipeline.local_align import LocalRegistration
from tools.analytic_sampling_probe import components
from tools.ap_stability_validation import integrated
from tools.ap_noise_ladder import hashes, key, CASES
from tools.ap_fractional_phase import evaluate
from tools.regularized_ap_fit import RegularizedAPFit
from tools.joint_saturn_experiment import digest
from tools.validate_local_warp_centring import image_errors,save_png,weighted_field_errors

CONFIG=dict(seed=29033,positions=list(range(1,32,2)),levels=[.25,.5,1.,2.],policies=['baseline','refined'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepare',action='store_true')
    p.add_argument('--case',choices=list(CASES))
    args=p.parse_args()
    if args.prepare==bool(args.case):
        p.error('Choose prepare or a case')
    root=Path('out/ap-noise-ladder-validation')
    screen=Path('out/ap-noise-ladder/analysis.json')
    screen_report=json.loads(screen.read_text())
    if screen_report['manifest']['hashes']!=hashes():
        raise ValueError('Screen implementation changed')
    identity=dict(config=CONFIG,source_sha256=digest(__file__),hashes=hashes(),screen_sha256=digest(screen),
        geometry_sha256=digest('out/ap-oracle-observations/geometry.npz'),
        prior_hashes={case:digest(f'out/ap-fractional-validation/{case}.npz') for case in CASES})
    if args.prepare:
        root.mkdir(parents=True,exist_ok=False)
        (root/'manifest.json').write_text(json.dumps(identity,indent=2)+'\n')
        return
    if json.loads((root/'manifest.json').read_text())!=identity:
        raise ValueError('Validation changed')
    root=root/args.case
    root.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    with np.load('out/ap-oracle-observations/geometry.npz') as data:
        reference,mask=data['reference'],data['mask']
    with np.load(f'out/ap-fractional-validation/{args.case}.npz') as data:
        truth=data['truth']
    shape,count=reference.shape,len(CONFIG['positions'])
    model=RegularizedAPFit(reference,policies=['baseline'])
    engine=model.engine
    features=components('resolved',shape)
    sums,twins,fields,statistics=({key(p,s):[] if kind in ('fields','statistics') else np.zeros(shape)
        for p in CONFIG['policies'] for s in CONFIG['levels']} for kind in ('sums','twins','fields','statistics'))
    rng=np.random.default_rng(CONFIG['seed'])
    target=np.zeros(shape)
    for index,phase in enumerate(2*np.pi*np.arange(32)/32):
        unit_noise=rng.normal(size=shape)
        if index not in CONFIG['positions']:
            continue
        a,b=(0.,0.) if args.case=='stationary' else (1.5*np.sin(phase),1.2*np.cos(phase))
        if args.case=='short_offset':
            a,b=a+.8,b-.5
        shift=(.35*np.sin(phase+.4),.4*np.cos(phase))
        clean=integrated(features,shape,a,b,shift,CASES[args.case])
        global_shift=np.asarray(shift) if args.case=='short_offset' else truth[index][:,mask].mean(1)
        clean_tensor=torch.as_tensor(clean,device=engine.device)
        exact=torch.as_tensor(truth[index],device=engine.device)
        target+=sample(clean_tensor,engine.yy+exact[1],engine.xx+exact[0]).cpu().numpy()/count
        for sigma in CONFIG['levels']:
            frame=clean+sigma*unit_noise
            tensor=torch.as_tensor(frame,device=engine.device)
            proxy=LocalRegistration.proxy(frame)
            for policy in CONFIG['policies']:
                name=key(policy,sigma)
                field,stats=evaluate(model,proxy,global_shift,truth[index],mask,policy)
                fields[name].append(field.cpu().numpy())
                statistics[name].append(stats)
                for src,accum in ((tensor,sums),(clean_tensor,twins)):
                    accum[name]+=sample(src,engine.yy+field[1],engine.xx+field[0]).cpu().numpy()/count
        print(args.case,'completed',index,flush=True)
    report=dict(identity=identity,case=args.case,variants={})
    for policy in CONFIG['policies']:
        for sigma in CONFIG['levels']:
            name=key(policy,sigma)
            report['variants'][name]=dict(policy=policy,sigma_adu=sigma,
                image=image_errors(sums[name],target,mask),clean=image_errors(twins[name],target,mask),
                field=weighted_field_errors(np.asarray(fields[name]),truth[CONFIG['positions']],np.ones(count)/count,mask),
                statistics=statistics[name])
            save_png(root/f'{name}.png',sums[name])
    save_png(root/'target.png',target)
    np.savez_compressed(root/'arrays.npz',**sums,**{'clean_'+k:v for k,v in twins.items()},
        **{'field_'+k:np.asarray(v) for k,v in fields.items()},mask=mask,target=target,truth=truth[CONFIG['positions']])
    report['raw_sha256']=digest(root/'arrays.npz')
    if identity['hashes']!=hashes() or identity['source_sha256']!=digest(__file__):
        raise ValueError('Sources changed during validation')
    (root/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
