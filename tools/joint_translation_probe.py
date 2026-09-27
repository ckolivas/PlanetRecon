"""Check local-motion claims when the only true motion is a known translation."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_erosion, gaussian_filter, shift as translate
import torch

from planetrecon.backends.torch_circular import sample
from planetrecon.pipeline.circular_align import CircularMultiscaleRegistration
from planetrecon.pipeline.local_align import LocalRegistration
from tools.joint_registration import JointRegistration
from tools.validate_local_warp_centring import texture_scene, image_errors


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--generator',choices=['scipy_spline','model'],default='scipy_spline')
    p.add_argument('--noise',type=float,nargs='+',default=[0.,2.])
    args=p.parse_args()
    torch.set_num_threads(4)
    reference=np.load('out/saturn-controlled-alignment/reference.npy')
    report=dict(model_sha256=hashlib.sha256(Path('tools/joint_registration.py').read_bytes()).hexdigest(),
                generator=args.generator,frames=[])
    for name,scene in [('saturn',reference),('texture',texture_scene(reference.shape))]:
        engine=JointRegistration(scene,CircularMultiscaleRegistration(scene))
        mask=binary_erosion(scene>.08*scene.max(),iterations=3)
        mask[:8]=mask[-8:]=False;mask[:,:8]=mask[:,-8:]=False
        for shift in ((.25,.5),(.6,-.4),(-.7,.35)):
            origin=torch.tensor(shift,device=engine.device,dtype=torch.float64)[:,None,None]
            for blur in (0.,1.):
                blurred=gaussian_filter(scene,blur) if blur else scene
                clean=(translate(blurred,shift[::-1],order=3,mode='reflect') if args.generator=='scipy_spline' else
                       engine.render(engine.bank[int(blur*4)],engine.xx-origin[0],engine.yy-origin[1]).cpu().numpy())
                oracle=sample(torch.tensor(clean,device=engine.device),engine.yy+origin[1],engine.xx+origin[0]).cpu().numpy()
                for noise in args.noise:
                    frame=clean+np.random.default_rng(9019).normal(0,noise,scene.shape)
                    field,stats=engine.fit(frame,shift,.3,lambda:None)
                    baseline=engine.engine.displacement(LocalRegistration.proxy(frame),shift,lambda:None)
                    values={}
                    for key,displacement in [('coherent',baseline),('joint_strong',field)]:
                        warped=sample(torch.tensor(clean,device=engine.device),
                                      engine.yy+displacement[1],engine.xx+displacement[0]).cpu().numpy()
                        values[key]=dict(clean_image=image_errors(warped,oracle,mask),
                            false_motion_rms_px=float((displacement-origin)[:,mask].square().sum(0).mean().sqrt()))
                    report['frames'].append(dict(scene=name,shift_xy=shift,blur=blur,noise=noise,
                                                  stats=stats,variants=values))
                print(name,shift,blur,'complete',flush=True)
                args.out.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
