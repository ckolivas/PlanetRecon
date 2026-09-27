"""Collect the regional-gate experiment without duplicating per-frame traces."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source(path):
    return dict(path=str(path),sha256=digest(path))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    result=dict(production_changed=False,brightness_normalization=False,output_smoothing=False,cases=[],
        code_sha256={name:digest('tools/'+name+'.py') for name in
                     ('regional_registration','validated_registration','coherent_registration','refit_registration')})
    recipe=None
    for name in ('regional-saturn-controls','regional-texture-controls','regional-repeat-controls'):
        path=Path('out')/name/'analysis.json'
        report=json.loads(path.read_text())
        if report['model_sha256']!=result['code_sha256']['regional_registration']:
            raise ValueError('Synthetic model changed')
        current={k:report['sharpening'][k] for k in ('wavelet','deconvolution','implementation_sha256')}
        if recipe is None:recipe=current
        elif recipe!=current:raise ValueError('Sharpening changed')
        for scene,cases in report['scenes'].items():
            for condition,case in cases.items():
                values={k:v for k,v in case.items() if k!='statistics'}
                values.update(scene=scene,condition=condition,source=source(path),
                              coherent_baseline_max_difference_adu=report['maximum_coherent_baseline_difference_adu'])
                values['guard_rejections']={p:sum(not s[p+'_guard_accepted'] for s in case['statistics'])
                                            for p in ('regional_any','regional_both')}
                result['cases'].append(values)
    if len(result['cases'])!=16:raise ValueError('Incomplete control set')
    result['synthetic_frames']=sum(c['frames'] for c in result['cases'])
    result['sharpening']=recipe
    probe_path=Path('out/regional-saturn-probe.json')
    probe=json.loads(probe_path.read_text())
    if probe['hashes']['regional_registration']!=result['code_sha256']['regional_registration']:
        raise ValueError('Probe model differs')
    whole=np.array([r['accepted_halves']>0 for r in probe['frames']])
    region=np.array([r['region_accepted_halves'] for r in probe['frames']])
    result['probe']=dict(source=source(probe_path),summary=probe['summary'],partition=probe['partition'],
        regional_counts=[dict(any=int((region[:,i]>=1).sum()),both=int((region[:,i]>=2).sum()),
                              both_while_whole_rejected=int(((region[:,i]>=2)&~whole).sum())) for i in range(6)])
    real_path=Path('out/saturn-regional-full/analysis.json')
    real=json.loads(real_path.read_text())
    if real['n_used']!=5738 or real['hashes']['regional_registration']!=result['code_sha256']['regional_registration']:
        raise ValueError('Unexpected full replay identity')
    if {k:real['sharpening'][k] for k in recipe}!=recipe:raise ValueError('Real sharpening differs')
    result['real_source']=source(real_path)
    result['real']=real
    result['baseline_replay_max_difference_adu']={}
    for name in ('coherent','full_any'):
        with np.load(real_path.parent/f'{name}.npz') as now,np.load(f'out/saturn-refit-full/{name}.npz') as old:
            result['baseline_replay_max_difference_adu'][name]=float(np.max(abs(now['image']-old['image'])))
            np.testing.assert_allclose(now['image'],old['image'],atol=1e-10,rtol=0)
            np.testing.assert_allclose(now['coverage'],old['coverage'],atol=1e-10,rtol=0)
    region_counts=[];global_counts=[]
    for path in sorted(real_path.parent.glob('shard_*.npz')):
        with np.load(path) as data:
            stats=json.loads(str(data['metadata']))['fit_statistics']
            region_counts.extend(s['region_accepted_halves'] for s in stats)
            global_counts.extend(s['accepted_halves'] for s in stats)
    rc=np.array(region_counts);gc=np.array(global_counts)
    if rc.shape!=(5738,6):raise ValueError('Incomplete regional statistics')
    result['real_regional_counts']=dict(whole_rejected=int((gc==0).sum()),
        whole_reject_but_region_both=int(((gc==0)&(rc>=2).any(1)).sum()),
        no_region_any=int((rc==0).all(1).sum()),no_region_both=int((rc<2).all(1).sum()),
        per_region_both=(rc>=2).sum(0).tolist())
    result['limits']=[
        'A fixed six-sector partition is a heuristic, not adaptive planetary segmentation.',
        'Independent motion selection is followed by a full-data refit; the final field is not independently validated.',
        'Regional fitting changes nuisance-model flexibility and sample count as well as spatial selectivity.',
        'One real capture has no known geometric truth; variation and edge widths do not establish resolution.',
        'Only confidence in the displacement field is blended; no output smoothing or normalization is added.']
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(result['synthetic_frames'],'synthetic frames; 5738 real frames; both baseline stacks reproduced')


if __name__=='__main__':main()
