"""Score exact sharpened reference-blur controls and verify the coherent baseline."""
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
                for key, values in case['variants'].items():
                    geometry = (data['clean_'+key]-data['noiseless_oracle'])[mask]
                    noise = (data[key]-data['clean_'+key])[mask]
                    gm, nm, cross = (float(np.mean(geometry**2)), float(np.mean(noise**2)),
                                     float(2*np.mean(geometry*noise)))
                    total = float(np.mean((geometry+noise)**2))
                    np.testing.assert_allclose(total, gm+nm+cross, atol=1e-12, rtol=0)
                    values['raw_error_decomposition'] = dict(clean_geometry_mse_adu2=gm,
                        warped_noise_mse_adu2=nm, twice_cross_term_adu2=cross, total_mse_adu2=total)
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
            from tools.validate_local_warp_centring import cases as control_cases
            truth=control_cases(case['frames'])[name][2]
            case['blur_estimation']={}
            for pose in ('global','local'):
                selected=np.array([s[pose+'_sigma'] for s in case['statistics']])
                case['blur_estimation'][pose]=dict(rmse_sigma_px=float(np.sqrt(np.mean((selected-truth)**2))),
                    zero_count=int((selected==0).sum()),upper_bound_count=int((selected==2).sum()))
            print(scene,name,'clean / sharp',
                  {k:[round(v['clean_image']['rmse_adu'],6),round(v['sharpened']['rmse_linear_0_1'],6)]
                   for k,v in case['variants'].items() if k in ('coherent','matched_global','matched_local','known_blur')},flush=True)
    (root/'analysis.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
