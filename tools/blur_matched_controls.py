"""Evaluate estimated reference blur against identical known-motion clean twins."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.ndimage import binary_erosion
import torch

from planetrecon.backends.torch_circular import sample
from planetrecon.pipeline.circular_align import CircularMultiscaleRegistration
from tools.blur_matched_registration import BlurMatchedRegistration
from tools.validate_local_warp_centring import cases,make_frame,texture_scene,image_errors,save_png


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--scenes',nargs='+',choices=['saturn','texture'],default=['saturn','texture'])
    parser.add_argument('--cases',nargs='+',choices=list(cases(48)),
                        default=['static_noise','static_blur_noise','motion_blur_noise','offset_motion_blur'])
    parser.add_argument('--frames',type=int,default=48)
    parser.add_argument('--seed',type=int,default=9017)
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    if args.frames<8:parser.error('at least eight frames required')
    args.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    reference=np.load('out/saturn-controlled-alignment/reference.npy')
    report=dict(reference_sha256=hashlib.sha256(reference.tobytes()).hexdigest(),
                model_sha256=hashlib.sha256(Path('tools/blur_matched_registration.py').read_bytes()).hexdigest(),
                production_changed=False,normalization=False,output_filtering=False,scenes={})
    started=time.perf_counter()
    for scene_name,scene in [('saturn',reference),('texture',texture_scene(reference.shape))]:
        if scene_name not in args.scenes:continue
        engine=BlurMatchedRegistration(scene,CircularMultiscaleRegistration(scene),device=args.device)
        mask=binary_erosion(scene>.08*scene.max(),iterations=3)
        mask[:8]=mask[-8:]=False;mask[:,:8]=mask[:,-8:]=False
        report['scenes'][scene_name]={}
        for name,parameters in cases(args.frames).items():
            if name not in args.cases:continue
            aa,bb,blur,noise,weights=parameters;weights=weights/weights.sum()
            rng=np.random.default_rng(args.seed)
            images={};clean_images={};field_error={};stats=[]
            for a,b,sigma,weight in zip(aa,bb,blur,weights):
                clean,truth,_=make_frame(scene,a,b,sigma)
                frame=clean+rng.normal(0,noise,scene.shape)
                shift=truth[:,mask].mean(1)
                fields=engine.variants(frame,shift,lambda:None,known_sigma=float(sigma))
                fields['global']=torch.as_tensor(shift,device=args.device)[:,None,None].expand(2,*scene.shape)
                fields['oracle']=torch.as_tensor(truth,device=args.device)
                for key,field in fields.items():
                    raw=sample(torch.as_tensor(frame,device=args.device),engine.yy+field[1],engine.xx+field[0]).cpu().numpy()
                    twin=sample(torch.as_tensor(clean,device=args.device),engine.yy+field[1],engine.xx+field[0]).cpu().numpy()
                    images.setdefault(key,np.zeros_like(scene))[:] += weight*raw
                    clean_images.setdefault(key,np.zeros_like(scene))[:] += weight*twin
                    delta=field.cpu().numpy()[:,mask]-truth[:,mask]
                    field_error[key]=field_error.get(key,0.)+weight*np.mean(np.sum(delta*delta,axis=0))
                stats.append(engine.stats[-1])
            oracle=clean_images['oracle'];prefix=f'{scene_name}_{name}'
            result=dict(seed=args.seed,frames=args.frames,statistics=stats,variants={})
            for key in images:
                result['variants'][key]=dict(image=image_errors(images[key],oracle,mask),
                    clean_image=image_errors(clean_images[key],oracle,mask),field_vector_rmse_px=float(np.sqrt(field_error[key])))
                save_png(args.out/f'{prefix}_{key}.png',images[key])
            save_png(args.out/f'{prefix}_noiseless_oracle.png',oracle)
            np.savez_compressed(args.out/f'{prefix}.npz',**images,mask=mask,noiseless_oracle=oracle,
                                **{'clean_'+k:v for k,v in clean_images.items()})
            report['scenes'][scene_name][name]=result
            report['elapsed_seconds']=time.perf_counter()-started
            (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
            print(scene_name,name,'clean RMSE',
                  {k:round(v['clean_image']['rmse_adu'],5) for k,v in result['variants'].items()},flush=True)


if __name__=='__main__':main()
