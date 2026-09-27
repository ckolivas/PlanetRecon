"""Read-only real-frame probe of agreement between whole-image and regional gates."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from planetrecon.io.ser import SERSource
from planetrecon.pipeline.circular_align import CircularMultiscaleRegistration
from tools.coherent_saturn_experiment import inputs
from tools.refit_saturn_experiment import cached_validation
from tools.regional_registration import RegionalRegistration


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--count',type=int,default=96)
    args=parser.parse_args()
    if args.out.exists():raise FileExistsError(args.out)
    _,indices,quality,shifts,reference,hashes=inputs()
    if not 1<=args.count<=len(indices):parser.error('invalid sample count')
    hashes['validated_registration']=hashlib.sha256(Path('tools/validated_registration.py').read_bytes()).hexdigest()
    hashes['regional_registration']=hashlib.sha256(Path('tools/regional_registration.py').read_bytes()).hexdigest()
    cached=cached_validation(Path('out/saturn-validated-field'),hashes,len(indices))
    selected=np.linspace(0,len(indices)-1,args.count,dtype=int)
    torch.set_num_threads(4)
    engine=RegionalRegistration(reference,CircularMultiscaleRegistration(reference))
    rows=[];start=time.perf_counter()
    with SERSource('2024-09-27-1154_3-CK-R-Sat.ser') as source:
        for i,p in enumerate(selected):
            engine.variants(source.read_raw(int(indices[p])),shifts[p],lambda:None)
            stats=engine.stats[-1]
            if stats['accepted_halves']!=cached[p]['accepted_halves']:
                raise ValueError('Whole-frame gate no longer matches baseline')
            rows.append(dict(position=int(p),frame_index=int(indices[p]),quality=float(quality[p]),**stats))
            if (i+1)%24==0:print(i+1,'frames',round(time.perf_counter()-start,1),'seconds',flush=True)
    global_pass=np.array([r['accepted_halves']>0 for r in rows])
    regions=np.array([r['region_accepted_halves'] for r in rows])
    summary=dict(frames=len(rows),whole_frame_rejected=int((~global_pass).sum()),
        whole_reject_but_some_region_any=int(np.sum((~global_pass)&(regions>=1).any(1))),
        whole_reject_but_some_region_both=int(np.sum((~global_pass)&(regions>=2).any(1))),
        region_any_mixed=int(np.sum((regions>=1).any(1)&(regions==0).any(1))),
        region_both_mixed=int(np.sum((regions>=2).any(1)&(regions<2).any(1))))
    report=dict(hashes=hashes,partition=engine.partition,selection='Evenly spaced positions in fixed 5738-frame selection',
                summary=summary,frames=rows,elapsed_seconds=time.perf_counter()-start,
                limits=['Regional predictive agreement is not known geometric truth.',
                        'This is a probe, not a reduced-frame stack or a grain comparison.'])
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2)+'\n')
    print(summary,flush=True)


if __name__=='__main__':main()
