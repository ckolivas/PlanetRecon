"""Check whether blur changes alone trigger local geometry corrections."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

from planetrecon.pipeline.circular_align import CircularMultiscaleRegistration
from planetrecon.pipeline.local_align import LocalRegistration
from planetrecon.backends.torch_circular import TorchCircularRegistration
from planetrecon.pipeline.align import phase_correlation_shift
from tools.local_warp_trace import trace


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--reference',type=Path,default=Path('out/saturn-controlled-alignment/reference.npy'))
    args=p.parse_args()
    if args.out.exists():raise FileExistsError(args.out)
    torch.set_num_threads(4)
    reference=np.load(args.reference)
    engine=TorchCircularRegistration(CircularMultiscaleRegistration(reference))
    report={'known_displacement_xy':[0.,0.],'added_noise':False,
        'reference_sha256':hashlib.sha256(reference.tobytes()).hexdigest(),
        'limits':'Reference contains its original texture/noise; no independent noise or geometric motion is added.',
        'frames':{}}
    for sigma in (0.,.5,1.,1.5,2.):
        frame=gaussian_filter(reference,sigma) if sigma else reference.copy()
        stages,records=trace(engine,LocalRegistration.proxy(frame),(0.,0.))
        field=stages[-1].cpu().numpy();left=field[:,175:230,130:225]
        detail={'measured_global_shift_xy':list(phase_correlation_shift(reference,frame)),
            'left_residual_rms_per_axis_px':np.sqrt(np.mean(left**2,axis=(1,2))).tolist(),
            'left_residual_max_norm_px':float(np.sqrt(np.sum(left**2,axis=0)).max()),
            'residual_xy_at_165_194':field[:,194,165].tolist(),
            'valid_AP_fraction_by_scale':[float(np.mean(r['measurements'][:,2])) for r in records]}
        report['frames'][str(sigma)]=detail
        print(sigma,detail,flush=True)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
