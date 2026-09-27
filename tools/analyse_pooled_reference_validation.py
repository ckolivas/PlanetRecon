"""Audit the frozen pooled-reference validation on two disjoint frame cohorts."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import center_of_mass
from planetrecon.result import load_snapshot
from tools.alignment_noise_experiment import read_png,measurements
from tools.analyse_joint_saturn import sharp_identity
from tools.analyse_refined_registration import ring_edges,edge_sensitivity
from tools.analyse_regularized_ap import fit_summary
from tools.coherent_saturn_experiment import inputs
from tools.joint_saturn_experiment import digest,CAPTURE,capture_identity
from tools.stack_repeatability import analyse_pair
from tools.validate_pooled_reference import identity,NAMES,PRIOR


def gallery(root,centres):
    import os
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QGuiApplication,QImage,QPainter,QColor,QFont
    app=QGuiApplication.instance() or QGuiApplication([])
    canvas=QImage(2000,2140,QImage.Format.Format_RGB32);canvas.fill(QColor('black'))
    painter=QPainter(canvas);painter.setPen(QColor('white'));painter.setFont(QFont('Sans',18))
    for cohort in (0,1):
        for kind,matcher in enumerate(('production','coherent')):
            cy,cx=centres[cohort,matcher]
            for column,ref in enumerate(('base','pool')):
                path=root/'sharpened'/f'c{cohort}_{matcher}_{ref}_sharpened.png'
                crop=QImage(str(path)).copy(int(cx-250),int(cy-115),500,230)
                crop=crop.scaled(1000,460,Qt.AspectRatioMode.IgnoreAspectRatio,Qt.TransformationMode.FastTransformation)
                x,y=column*1000,(cohort*2+kind)*510
                painter.drawText(x+12,y+32,f'Set {cohort+1}, {matcher}: '+('original reference' if ref=='base' else 'pooled 256 reference'))
                painter.drawImage(x,y+45,crop)
    painter.drawText(12,2060,'512 identical frames within each set; sets and reference inputs are disjoint')
    painter.drawText(12,2105,'Wavelet 27/0/0/0 + Adaptive Deconvolution 15.6, Contrast Adaptive; raw pixels unfiltered')
    painter.end()
    if not canvas.save(str(root/'comparison.png')):raise OSError('Gallery export failed')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--record',type=Path,required=True)
    args=p.parse_args();root=args.out
    report=json.loads((root/'report.json').read_text());manifest=json.loads((root/'manifest.json').read_text())
    _,indices,quality,_,_,selected,versions,config=identity()
    prior=json.loads((PRIOR/'manifest.json').read_text())
    for key,value in inputs()[-1].items():
        if prior['hashes'][key]!=value:raise ValueError('Prior input identity changed')
    if (manifest['hashes']!=versions or manifest['config']!=config or manifest['capture']!=capture_identity()
        or manifest['capture_sha256']!=digest(CAPTURE) or manifest['frame_indices']!=indices[selected].tolist()
        or manifest['selection_positions']!=selected.tolist()):
        raise ValueError('Prepared inputs changed')
    versions['capture']=manifest['capture_sha256']
    stats=report.pop('statistics')
    if (report['hashes']!=versions or report['config']!=config or report['n_used']!=1024
        or [r['position'] for r in stats]!=list(range(1024))
        or [r['selection_position'] for r in stats]!=selected.tolist()
        or [r['frame_index'] for r in stats]!=indices[selected].tolist()
        or report['halves_sha256']!=digest(root/'halves.npz')):
        raise ValueError('Incomplete or mismatched replay')
    report['original_matcher_parity']={}
    for name in NAMES:
        values=[r['parity'][name] for r in stats if name in r['parity']]
        if len(values)!=32 or max(values)>1e-10:raise ValueError('Matcher parity failed')
        report['original_matcher_parity'][name]=dict(count=len(values),max_error_px=max(values))
    report['coherent_fit_summary']={n:fit_summary([r['fits'][n] for r in stats]) for n in NAMES if n.startswith('coherent')}
    report['production_stage_rejections']={n:np.sum([np.logical_not(r['fits'][n]['stage_acceptance']) for r in stats],axis=0).tolist()
        for n in NAMES if n.startswith('production')}
    report['raw_report_sha256']=digest(root/'report.json');report['analyzer_sha256']=digest(__file__)
    recipe=json.loads(Path('out/saturn-coherent-field/sharpened/sharpening_report.json').read_text())
    report['sharpening']=sharp_identity(root,recipe)
    if len(report['sharpening']['images'])!=24:raise ValueError('Expected 24 sharpened images')
    report['cohorts']={};centres={}
    with np.load(root/'halves.npz') as data:
        for cohort in (0,1):
            rows={};arrays={};slots=[cohort,cohort+2]
            for i,name in enumerate(NAMES):
                sums,weights=data['signal'][i,slots],data['support'][i,slots]
                halves=np.divide(sums,weights,out=np.zeros_like(sums),where=weights>0)
                coverage=weights.sum(0)
                mean=np.divide(sums.sum(0),coverage,out=np.zeros_like(coverage),where=coverage>0)
                stem=f'c{cohort}_{name}';snapshot=load_snapshot(root/(stem+'.npz'))
                np.testing.assert_array_equal(snapshot.image,mean)
                if snapshot.n_used!=512 or snapshot.provenance['frame_indices']!=indices[selected[cohort::2]].tolist():
                    raise ValueError('Snapshot cohort mismatch')
                arrays[name]=read_png(root/'sharpened'/(stem+'_sharpened.png'))[0]
                rows[name]=dict(raw=measurements(mean),half_difference_over_two=measurements(mean,difference=(halves[0]-halves[1])/2),
                    repeatability=analyse_pair(*halves),sharp=measurements(arrays[name]),ring_edges=ring_edges(arrays[name]),
                    snapshot_sha256=digest(root/(stem+'.npz')))
            paired={}
            for matcher in ('production','coherent'):
                a,b=matcher+'_base',matcher+'_pool'
                centre=np.rint(center_of_mass(np.maximum(arrays[a]-.02*arrays[a].max(),0))).astype(int)
                centres[cohort,matcher]=centre
                images=dict(production_local=arrays[b],existing_bilinear=arrays[a],AS_manual64=arrays[a])
                paired[matcher]=dict(
                    sharp_variation_change_percent={reg:100*(rows[b]['sharp'][reg]['highpass_percent']/rows[a]['sharp'][reg]['highpass_percent']-1) for reg in ('upper','lower')},
                    half_difference_change_percent={reg:100*(rows[b]['half_difference_over_two'][reg]['highpass_percent']/rows[a]['half_difference_over_two'][reg]['highpass_percent']-1) for reg in ('upper','lower')},
                    edges={side:dict(paired_windows=v['paired_windows'],pool_minus_base_width_px_min_median_max=v['local_minus_global_width_px_min_median_max'])
                        for side,v in edge_sensitivity(images,{k:centre for k in images}).items()})
            q=np.maximum(quality[selected[cohort::2]],1e-12)
            report['cohorts'][str(cohort)]=dict(frame_indices=indices[selected[cohort::2]].tolist(),
                effective_frame_count=float(q.sum()**2/np.sum(q*q)),half_weight_fractions=[float(q[h::2].sum()/q.sum()) for h in (0,1)],
                methods=rows,paired=paired)
    previous=json.loads((PRIOR/'analysis.json').read_text())
    if previous['hashes']['references']!=versions['references']:raise ValueError('Previous reference identity differs')
    report['previous_pilot']=dict(analysis_sha256=digest(PRIOR/'analysis.json'),
        coherent_base=previous['raw_halves']['baseline']['repeatability'],coherent_pool=previous['raw_halves']['pooled']['repeatability'])
    report['stage_sha256']={p.name:digest(p) for p in sorted((root/'sharpened').glob('*_stages.npz'))}
    report['limits']=[
        'Two disjoint frame sets from the same capture, not independent observing sessions or atmospheric realizations.',
        'Previous replay frames and both historical/native reference inputs are excluded from both new sets.',
        'References, global shifts, AP geometry and all numerical settings were frozen before validation.',
        'Production here means its local circular matcher with frozen experiment inputs, not a fresh GUI pipeline run.',
        'Shared reference biases cancel in half differences and may increase repeatability; true detail is not known.',
        'Variation mixes signal and artifacts; ring widths and correlations are descriptive without calibrated significance.',
        'Historical and pooled references differ in preparation and membership; this is not a pure reference-count test.',
        'No normalization, output filter or production setting was changed.']
    text=json.dumps(report,indent=2)+'\n';(root/'analysis.json').write_text(text);args.record.write_text(text)
    gallery(root,centres)
    for cohort,values in report['cohorts'].items():
        print('Cohort',cohort)
        for matcher,pair in values['paired'].items():
            print(matcher,'sharp',pair['sharp_variation_change_percent'],'halves',pair['half_difference_change_percent'])
            for ref in ('base','pool'):
                print(ref,'ring correlation',[values['methods'][matcher+'_'+ref]['repeatability'][side]['bands'][2]['correlation'] for side in ('left_ring','right_ring')])


if __name__=='__main__':main()
