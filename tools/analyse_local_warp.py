"""Analyse scale-prefix and mean-field ablations after the fixed sharpening."""
import argparse
import json
from pathlib import Path

import numpy as np
from PySide6.QtGui import QGuiApplication, QImage, QPainter, QColor, QFont
from PySide6.QtCore import Qt
from scipy.ndimage import center_of_mass

from tools.alignment_noise_experiment import read_png, measurements
from tools.analyse_refined_registration import ring_edges, edge_sensitivity


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=Path('out/saturn-local-warp-trace'))
    args=p.parse_args(); out=args.out
    report=json.loads((out/'report.json').read_text())
    report['sharpening']=json.loads((out/'sharpened'/'sharpening_report.json').read_text())
    with np.load(out/'traces.npz') as d:
        fields,quality,points=(d[k].copy() for k in ('point_fields','quality','points_xy'))
        stage_accepted=d['stage_accepted'].copy()
    weights=quality/quality.sum()
    means=np.einsum('n,nsap->sap',weights,fields)
    variance=np.einsum('n,nsap->sap',weights,(fields-means[None])**2)
    report['point_motion']={'points_xy':points.tolist(),'stage_sizes':report['sizes'],
                            'mean_xy':means.tolist(),'std_xy':np.sqrt(variance).tolist()}
    report['ap_summary']={}
    for size in report['sizes']:
        path=out/f'ap_{size}.npz'
        if not path.exists():continue
        with np.load(path) as d:
            a,c=d['measurements'].copy(),d['centres_xy'].copy()
        valid_before_guard=a[:,:,2]>0
        valid=valid_before_guard & stage_accepted[:,report['sizes'].index(size),None]
        region={}
        for name,lo,hi in [('left',110,245),('right',450,585)]:
            ids=(c[:,0]>=lo)&(c[:,0]<hi)&(c[:,1]>=165)&(c[:,1]<235)
            weighted=weights[:,None]*valid[:,ids]
            n=float(weighted.sum())
            deltas=a[:,ids,:2]
            mean=(weighted[:,:,None]*deltas).sum((0,1))/max(n,1e-30)
            std=np.sqrt((weighted[:,:,None]*(deltas-mean)**2).sum((0,1))/max(n,1e-30))
            region[name]={'points':c[ids].tolist(),'valid_fraction':float(valid[:,ids].mean()) if ids.any() else None,
                          'accepted_delta_mean_xy':mean.tolist(),'accepted_delta_std_xy':std.tolist()}
        report['ap_summary'][str(size)]={'active_points':len(c),'valid_fraction':float(valid.mean()),
            'valid_fraction_before_field_guard':float(valid_before_guard.mean()),'regions':region}
    paths={
        'global':Path('out/saturn-refined-registration/sharpened/existing_bilinear_as_transfer_sharpened.png'),
        'AS':Path('out/saturn-matched-transfer/sharpened/as_manual64_sharpened.png'),
        **{name:out/'sharpened'/f'{name}_as_transfer_sharpened.png' for name in report['names']}}
    arrays={k:read_png(v)[0] for k,v in paths.items()}
    report['comparison']={}
    for name,a in arrays.items():
        report['comparison'][name]={'sharpened':measurements(a),'ring_edges':ring_edges(a)}
        if name not in ('global','AS'):
            report['comparison'][name]['edge_sensitivity']=edge_sensitivity({
                'production_local':a,'existing_bilinear':arrays['global'],'AS_manual64':arrays['AS']})
        m=report['comparison'][name]
        print(name,'variation',*[round(v['highpass_percent'],5) for v in m['sharpened'].values()],
              'edge widths',*[round(v['width_10_90_px'],5) for v in m['ring_edges'].values()])
    app=QGuiApplication.instance() or QGuiApplication([])
    canvas=QImage(2000,510*((len(paths)+1)//2),QImage.Format.Format_RGB32);canvas.fill(QColor('black'))
    painter=QPainter(canvas);painter.setPen(QColor('white'));painter.setFont(QFont('Sans',18))
    for i,(name,path) in enumerate(paths.items()):
        a=arrays[name];cy,cx=np.round(center_of_mass(np.maximum(a-a.max()*.02,0))).astype(int)
        crop=QImage(str(path)).copy(int(cx-250),int(cy-115),500,230)
        crop=crop.scaled(1000,460,Qt.AspectRatioMode.IgnoreAspectRatio,Qt.TransformationMode.FastTransformation)
        x,y=(i%2)*1000,(i//2)*510;painter.drawText(x+15,y+32,name+' (same sharpening)');painter.drawImage(x,y+45,crop)
    painter.end();assert canvas.save(str(out/'comparison.png'))
    report['limits']=[
        'Same 5738 frames, scalar weights and fixed reference; no frame brightness normalization.',
        'All PR displayed variants receive measured AS sigma-1 radius-3 smoothing before fixed sharpening.',
        'Per-AP residuals are in the preceding warped coordinate system, not final sensor displacements.',
        'Mean-only and zero-mean fields isolate static versus varying components; neither is a validated production method.',
        'Ring widths include physical geometry and processing; variation includes detail, not pure noise.',
        'This is one capture, not a general test of local alignment benefits.']
    blur=out/'zero_motion_blur_verified.json'
    if blur.exists():
        report['zero_motion_blur']=json.loads(blur.read_text())
        report['reference_sha256']=report['zero_motion_blur']['reference_sha256']
    (out/'analysis.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Point means by stage:',np.round(means,4).tolist())
    print('Point standard deviations by stage:',np.round(np.sqrt(variance),4).tolist())


if __name__=='__main__':main()
