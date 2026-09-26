"""Compare multiscale experiments after the exact PlanetaryTools sharpening."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import center_of_mass

from tools.alignment_noise_experiment import measurements, read_png
from tools.multiscale_patch_experiment import SIZES, NAMES, spread_subset
from planetrecon.runtime import apply_thread_limits


def overlap_counts(args, report):
    with np.load(args.single/'measurements.npz') as data:
        scalar=data['global_weights'].copy(); original=data['original_selected'].copy()
    with np.load(args.baseline/'global_subpixel.npz') as data:reference=data['image'].copy()
    cy,cx=np.round(center_of_mass(np.maximum(reference-reference.max()*.02,0))).astype(int)
    blend_by_region={'upper':[],'lower':[]}; selected=[]; subset=[]; offset=0
    for w in SIZES:
        path=args.single/'patch_stacks.npz' if w==65 else args.experiment/f'patch_stacks_{w}.npz'
        with np.load(path) as data:centres,foot=data['centres'].copy(),data['footprint'].copy()
        path=args.single/'selections.npz' if w==65 else args.experiment/f'selections_{w}.npz'
        with np.load(path) as data:selected.append(data['masks'][:3].copy())
        subset.extend((spread_subset(centres,report['sparse_ap_counts'][str(w)])+offset).tolist())
        offset+=len(centres)
        h=w//2
        for region,ylo in [('upper',cy-55),('lower',cy+35)]:
            yy,xx=np.indices((25,80)); yy+=ylo; xx+=cx-40
            dy,dx=yy[None]-centres[:,0,None,None]+h,xx[None]-centres[:,1,None,None]+h
            good=(dy>=0)&(dy<w)&(dx>=0)&(dx<w)
            blend_by_region[region].append((foot[dy.clip(0,w-1),dx.clip(0,w-1)]*good).reshape(len(centres),-1))
    selected=np.concatenate(selected,axis=2)
    for j,name in enumerate(NAMES):
        weights=np.column_stack([selected[j],original])*scalar[:,None]
        weights/=weights.sum(axis=0)
        covariance=weights.T@weights
        for layout,keep in [('multiscale',np.arange(offset)),('multiscale55',np.array(subset))]:
            ids=np.r_[keep,offset]
            counts={}
            for region,pieces in blend_by_region.items():
                blend=np.concatenate(pieces)[keep]
                coverage=blend.sum(axis=0)
                blend=np.vstack([blend,np.maximum(1.-coverage,0.)])/np.maximum(coverage,1.)
                variance=np.sum(blend*(covariance[np.ix_(ids,ids)]@blend),axis=0)
                counts[region]=np.percentile(1/variance,[10,50,90]).tolist()
            report['variants'][f'{layout}_{name}']['frame_weight_effective_counts_p10_median_p90']=counts


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment',type=Path,required=True)
    p.add_argument('--single',type=Path,default=Path('out/saturn-patch-stacking'))
    p.add_argument('--baseline',type=Path,default=Path('out/saturn-controlled-alignment'))
    args=p.parse_args(); apply_thread_limits(8)
    report=json.loads((args.experiment/'report.json').read_text())
    report['date']='2026-09-27'
    report['sharpening']=json.loads((args.experiment/'sharpened'/'sharpening_report.json').read_text())
    report['comparison']={}
    for name,path in [('PR_production',Path('2024-09-27-1154_3-CK-R-SatPs.png')),
                      ('AS_manual64',Path('AS_F5738/2024-09-27-1154_3-CK-R-Sat_r64_lapl4_ap54s.png'))]:
        image,_=read_png(path)
        report['comparison'][name]=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                       sharpened=measurements(image))
    for name,item in report['variants'].items():
        image,_=read_png(args.experiment/'sharpened'/f'{name}_sharpened.png')
        item['sharpened']=measurements(image)
    overlap_counts(args,report)
    report['limits']=[
        'Tests independent patch stacking at multiple sizes, not the private AutoStakkert algorithm.',
        'Fixed 64-frame reference, 25476 screened frames, and 5738 contributions per AP.',
        '120 circular APs at 65/93/131/185px, plus a spatially spread 55-AP subset count control.',
        'The 55-AP subset does not reproduce the exact positions or size distribution in the AS screenshot.',
        'Same original scalar quality weights and absolute frame brightness; no normalization or denoising.',
        'Completed-patch registration adds another bilinear interpolation.',
        'Frame-weight effective counts omit covariance effects from differing subpixel samples.',
        'Fine-scale variation includes detail and artifacts; it is not a pure sensor-noise measurement.',
        'The 65px controls are reconstructed from the preceding experiment and must match its floating images.',
        'AS comparison uses the supplied manual-64 reference output, not a new AS run.']
    (args.experiment/'analysis.json').write_text(json.dumps(report,indent=2)+'\n')
    for name,item in list(report['comparison'].items())+list(report['variants'].items()):
        m=item['sharpened']
        print(name,*(round(m[k]['highpass_percent'],4) for k in ('upper','lower')))


if __name__=='__main__':main()
