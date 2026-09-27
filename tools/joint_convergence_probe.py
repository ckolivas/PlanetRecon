"""Bounded convergence check before choosing a joint-fit iteration budget."""
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
from tools.joint_registration import JointRegistration
from tools.validate_local_warp_centring import make_frame, texture_scene, image_errors


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    torch.set_num_threads(4)
    reference=np.load('out/saturn-controlled-alignment/reference.npy')
    report=dict(model_sha256=hashlib.sha256(Path('tools/joint_registration.py').read_bytes()).hexdigest(),frames=[])
    for scene_name,scene in [('saturn',reference),('texture',texture_scene(reference.shape))]:
        engine=JointRegistration(scene,CircularMultiscaleRegistration(scene))
        mask=binary_erosion(scene>.08*scene.max(),iterations=3)
        mask[:8]=mask[-8:]=False;mask[:,:8]=mask[:,-8:]=False
        for name,a,b,blur in [('stationary',0.,0.,1.5),('motion',1.3,-.9,1.),('offset',2.,-.6,.4)]:
            clean,truth,_=make_frame(scene,a,b,blur)
            frame=clean+np.random.default_rng(9017).normal(0,2.,scene.shape)
            shift=truth[:,mask].mean(1)
            oracle=sample(torch.as_tensor(clean,device=engine.device),
                          engine.yy+torch.tensor(truth[1],device=engine.device),
                          engine.xx+torch.tensor(truth[0],device=engine.device)).cpu().numpy()
            values=[]
            for iterations in (100,300,800):
                engine.maxiter=iterations
                start=time.perf_counter()
                field,stats=engine.fit(frame,shift,.3,lambda:None)
                warped=sample(torch.as_tensor(clean,device=engine.device),engine.yy+field[1],engine.xx+field[0]).cpu().numpy()
                values.append(dict(budget=iterations,seconds=time.perf_counter()-start,stats=stats,
                                   clean_image=image_errors(warped,oracle,mask)))
                print(scene_name,name,iterations,values[-1]['clean_image']['rmse_adu'],stats['success'],stats['iterations'],flush=True)
            report['frames'].append(dict(scene=scene_name,case=name,results=values))
            args.out.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
