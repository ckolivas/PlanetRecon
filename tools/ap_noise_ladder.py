"""Paired factorial noise ladder for matching interpolation and peak refinement."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from planetrecon.backends.torch_circular import sample
from planetrecon.pipeline.local_align import LocalRegistration
from tools.analytic_sampling_probe import components
from tools.ap_stability_validation import integrated
from tools.ap_oracle_observations import CASES
from tools.ap_fractional_phase import hashes as matching_hashes, evaluate, POLICIES
from tools.regularized_ap_fit import RegularizedAPFit
from tools.joint_saturn_experiment import digest
from tools.validate_local_warp_centring import image_errors, save_png, weighted_field_errors

LEVELS = [0., .25, .5, 1., 2.]
POSITIONS = list(range(0,32,2))


def key(policy, sigma):
    return policy+'_s'+str(sigma).replace('.','p')


def hashes():
    return dict(matching_hashes(), **{'tools/ap_noise_ladder.py': digest(__file__)})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--case',choices=list(CASES))
    p.add_argument('--prepare',action='store_true')
    args=p.parse_args()
    if args.prepare == bool(args.case):
        p.error('Select exactly one of --prepare or --case')
    prior = Path('out/ap-fractional-validation')
    previous = json.loads((prior/'report.json').read_text())
    if previous['hashes'] != matching_hashes():
        raise ValueError('Prior implementation changed')
    config = dict(levels=LEVELS, positions=POSITIONS, seed=9033, policies={k:list(v) for k,v in POLICIES.items()})
    identity = dict(hashes=hashes(), config=config, prior_report_sha256=digest(prior/'report.json'),
        geometry_sha256=digest('out/ap-oracle-observations/geometry.npz'),
        prior_raw_sha256={case:digest(prior/f'{case}.npz') for case in CASES})
    if any(identity['prior_raw_sha256'][c] != previous['cases'][c]['raw_sha256'] for c in CASES):
        raise ValueError('Prior artifacts changed')
    if args.prepare:
        args.out.mkdir(parents=True,exist_ok=False)
        (args.out/'manifest.json').write_text(json.dumps(identity,indent=2)+'\n')
        return
    if json.loads((args.out/'manifest.json').read_text()) != identity:
        raise ValueError('Prepared run changed')
    root=args.out/args.case
    root.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    with np.load('out/ap-oracle-observations/geometry.npz') as data:
        reference, mask=data['reference'],data['mask']
    shape=reference.shape
    with np.load(prior/f'{args.case}.npz') as data:
        truths=data['truth']
        previous_fields={(policy,float(noise)):data[f'field_{policy}_noise{noise}'] for policy in ('baseline','cubic_refined') for noise in (0,2)}
    model=RegularizedAPFit(reference,policies=['baseline'])
    engine=model.engine
    features=components('resolved',shape)
    sums,twins,fields,statistics=({key(policy,noise):[] if kind in ('fields','statistics') else np.zeros(shape)
        for policy in POLICIES for noise in LEVELS} for kind in ('sums','twins','fields','statistics'))
    target=np.zeros(shape)
    rng=np.random.default_rng(config['seed'])
    reused=0
    count=len(POSITIONS)
    for index,phase in enumerate(2*np.pi*np.arange(32)/32):
        unit_noise=rng.normal(0,1.,shape)
        if index not in POSITIONS:
            continue
        a,b=(0.,0.) if args.case=='stationary' else (1.5*np.sin(phase),1.2*np.cos(phase))
        if args.case=='short_offset':
            a,b=a+.8,b-.5
        translation=(.35*np.sin(phase+.4),.4*np.cos(phase))
        clean=integrated(features,shape,a,b,translation,CASES[args.case])
        shift=np.asarray(translation) if args.case=='short_offset' else truths[index][:,mask].mean(1)
        clean_tensor=torch.as_tensor(clean,device=engine.device)
        truth_tensor=torch.as_tensor(truths[index],device=engine.device)
        target+=sample(clean_tensor,engine.yy+truth_tensor[1],engine.xx+truth_tensor[0]).cpu().numpy()/count
        for noise in LEVELS:
            frame=clean+noise*unit_noise
            proxy=LocalRegistration.proxy(frame)
            tensor=torch.as_tensor(frame,device=engine.device)
            for policy in POLICIES:
                name=key(policy,noise)
                if (policy,noise) in previous_fields:
                    field=torch.as_tensor(previous_fields[policy,noise][index],device=engine.device)
                    stats=dict(reused=True)
                    reused+=1
                    if index==0:
                        fresh,checked=evaluate(model,proxy,shift,truths[index],mask,policy)
                        torch.testing.assert_close(field,fresh,atol=1e-10,rtol=0)
                        stats['parity_max_error_px']=float((field-fresh).abs().max())
                else:
                    field,stats=evaluate(model,proxy,shift,truths[index],mask,policy)
                    stats['reused']=False
                fields[name].append(field.cpu().numpy())
                statistics[name].append(stats)
                for src,accum in ((tensor,sums),(clean_tensor,twins)):
                    accum[name]+=sample(src,engine.yy+field[1],engine.xx+field[0]).cpu().numpy()/count
        print(args.case,'completed position',index,flush=True)
    report=dict(identity=identity,case=args.case,reused_fields=reused,variants={})
    for policy in POLICIES:
        for noise in LEVELS:
            name=key(policy,noise)
            report['variants'][name]=dict(policy=policy,sigma_adu=noise,
                image=image_errors(sums[name],target,mask),clean=image_errors(twins[name],target,mask),
                field=weighted_field_errors(np.asarray(fields[name]),truths[POSITIONS],np.ones(count)/count,mask),
                statistics=statistics[name])
            save_png(root/f'{name}.png',sums[name])
    save_png(root/'target.png',target)
    artifact=root/'arrays.npz'
    np.savez_compressed(artifact,**sums,**{'clean_'+k:v for k,v in twins.items()},
        **{'field_'+k:np.asarray(v) for k,v in fields.items()},truth=truths[POSITIONS],mask=mask,target=target)
    report['raw_sha256']=digest(artifact)
    if identity['hashes']!=hashes():
        raise ValueError('Sources changed during run')
    (root/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
