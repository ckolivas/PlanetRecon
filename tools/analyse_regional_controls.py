"""Score exact sharpened regional-gate controls and verify the coherent baseline."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from tools.validate_local_warp_centring import image_errors


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();root=args.out
    report=json.loads((root/'report.json').read_text())
    sharp=json.loads((root/'sharpened/sharpening_report.json').read_text())
    prior=json.loads(Path('out/saturn-refit-full/sharpened/sharpening_report.json').read_text())
    for key in ('wavelet','deconvolution','implementation_sha256'):
        if sharp[key]!=prior[key]:raise ValueError('Sharpening recipe changed')
    report['sharpening']=sharp
    report['maximum_coherent_baseline_difference_adu']=0.
    for scene,cases in report['scenes'].items():
        for name,case in cases.items():
            prefix=f'{scene}_{name}'
            baseline=(Path('out/coherent-repeat') if case['seed']==9018 else
                      Path('out/coherent-validation' if scene=='saturn' else 'out/coherent-validation-texture'))
            with np.load(root/f'{prefix}.npz') as data,np.load(baseline/f'{prefix}.npz') as old:
                mask=data['mask']
                delta=float(np.max(abs(data['coherent']-old['local'])))
                report['maximum_coherent_baseline_difference_adu']=max(report['maximum_coherent_baseline_difference_adu'],delta)
                np.testing.assert_allclose(data['coherent'],old['local'],atol=1e-10,rtol=0)
            oracle_file=f'{prefix}_noiseless_oracle.png'
            digest=hashlib.sha256((root/oracle_file).read_bytes()).hexdigest()
            if digest!=hashlib.sha256((baseline/oracle_file).read_bytes()).hexdigest():
                raise ValueError('Noiseless oracle changed')
            case['oracle_sha256']=digest
            with np.load(root/'sharpened'/f'{prefix}_noiseless_oracle_stages.npz') as data:
                target=data['sharpened'].mean(2)
            for key,values in case['variants'].items():
                with np.load(root/'sharpened'/f'{prefix}_{key}_stages.npz') as data:
                    image=data['sharpened'].mean(2)
                error=image_errors(image,target,mask)
                values['sharpened']=dict(rmse_linear_0_1=error['rmse_adu'],
                    gradient_vector_rmse_linear_per_px=error['gradient_vector_rmse_adu_per_px'])
            counts=np.array([s['region_accepted_halves'] for s in case['statistics']])
            case['frames_with_regional_approvals']=dict(any=int((counts>=1).any(1).sum()),
                                                       both=int((counts>=2).any(1).sum()))
            print(scene,name,'clean / sharp',
                  {k:[round(v['clean_image']['rmse_adu'],6),round(v['sharpened']['rmse_linear_0_1'],6)]
                   for k,v in case['variants'].items() if k in ('coherent','full_any','regional_any','regional_both')},flush=True)
    (root/'analysis.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
