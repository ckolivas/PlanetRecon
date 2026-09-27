"""Audit the paired noise ladder and report conditional factorial effects."""
import json
from pathlib import Path

import numpy as np

from tools.ap_noise_ladder import hashes, CASES, LEVELS, POSITIONS, POLICIES, key
from tools.analyse_joint_saturn import sharp_identity
from tools.joint_saturn_experiment import digest
from tools.validate_local_warp_centring import image_errors, weighted_field_errors


def effects(values):
    """Differences in the supplied metric, not fractions of physical noise."""
    b,c,r,cr=(values[k] for k in ('baseline','cubic','refined','cubic_refined'))
    return dict(cubic_with_original_peak=c-b,cubic_with_refined_peak=cr-r,
                refinement_with_bilinear=r-b,refinement_with_cubic=cr-c,
                interaction=cr-c-r+b)


def crossings(levels,differences):
    """Report observed sign-change brackets; do not assume monotonicity."""
    if len(levels)!=len(differences) or any(b<=a for a,b in zip(levels,levels[1:])):
        raise ValueError('Increasing paired levels required')
    return [dict(lower_sigma=a,upper_sigma=b,lower_difference=x,upper_difference=y)
            for a,b,x,y in zip(levels,levels[1:],differences,differences[1:]) if x*y<0]


def main():
    root=Path('out/ap-noise-ladder')
    manifest=json.loads((root/'manifest.json').read_text())
    if manifest['hashes']!=hashes():
        raise ValueError('Sources changed')
    config=dict(levels=LEVELS,positions=POSITIONS,seed=9033,policies={k:list(v) for k,v in POLICIES.items()})
    if manifest['config']!=config or manifest['prior_report_sha256']!=digest('out/ap-fractional-validation/report.json'):
        raise ValueError('Configuration/input changed')
    if manifest['geometry_sha256']!=digest('out/ap-oracle-observations/geometry.npz'):
        raise ValueError('Reference geometry changed')
    for case,path_hash in manifest['prior_raw_sha256'].items():
        if digest(f'out/ap-fractional-validation/{case}.npz')!=path_hash:
            raise ValueError('Prior fields changed')
    report=dict(manifest=manifest,analyzer_sha256=digest(__file__),cases={})
    recipe=json.loads(Path('out/saturn-coherent-field/sharpened/sharpening_report.json').read_text())
    metrics=['field','clean','sharpened']
    for case in CASES:
        directory=root/case
        values=json.loads((directory/'report.json').read_text())
        if values['identity']!=manifest or values['case']!=case or values['reused_fields']!=64:
            raise ValueError('Case association/incomplete field count')
        if values['raw_sha256']!=digest(directory/'arrays.npz'):
            raise ValueError('Changed arrays')
        values['raw_report_sha256']=digest(directory/'report.json')
        values['sharpening']=sharp_identity(directory,recipe)
        expected={key(p,s) for p in POLICIES for s in LEVELS}
        if set(values['variants'])!=expected or set(values['sharpening']['images'])!=expected|{'target'}:
            raise ValueError('Incomplete factorial or sharpening')
        with np.load(directory/'arrays.npz') as data:
            mask,target,truth=data['mask'],data['target'],data['truth']
            if len(truth)!=len(POSITIONS):
                raise ValueError('Incorrect frame count')
            for name,variant in values['variants'].items():
                checks=dict(image=image_errors(data[name],target,mask),clean=image_errors(data['clean_'+name],target,mask),
                    field=weighted_field_errors(data['field_'+name],truth,np.ones(len(POSITIONS))/len(POSITIONS),mask))
                if any(variant[k]!=v for k,v in checks.items()):
                    raise ValueError('Metrics differ from saved arrays')
                stats=variant.pop('statistics')
                if len(stats)!=len(POSITIONS):
                    raise ValueError('Incomplete diagnostics')
                variant['fits']=dict(reused=sum(s['reused'] for s in stats),
                    new_guard_rejections=sum(not s['fit']['field_guard_accepted'] for s in stats if not s['reused']),
                    new_insufficient_points=sum(s['fit']['fallback'] for s in stats if not s['reused']),
                    endpoint_parity_errors=[s['parity_max_error_px'] for s in stats if 'parity_max_error_px' in s])
        with np.load(directory/'sharpened/target_stages.npz') as data:
            target=data['sharpened'].mean(2)
        values['stage_sha256']={'target':digest(directory/'sharpened/target_stages.npz')}
        for name,variant in values['variants'].items():
            path=directory/'sharpened'/f'{name}_stages.npz'
            values['stage_sha256'][name]=digest(path)
            with np.load(path) as data:
                variant['sharpened_error_linear_0_1']=image_errors(data['sharpened'].mean(2),target,mask)
        values['factorial']={}
        values['sign_change_brackets']={}
        for metric in metrics:
            def value(policy,sigma):
                v=values['variants'][key(policy,sigma)]
                return v['field']['total_vector_rmse_px'] if metric=='field' else (
                    v['clean']['rmse_adu'] if metric=='clean' else v['sharpened_error_linear_0_1']['rmse_adu'])
            values['factorial'][metric]=[dict(sigma_adu=sigma,values={p:value(p,sigma) for p in POLICIES},
                effects=effects({p:value(p,sigma) for p in POLICIES})) for sigma in LEVELS]
            values['sign_change_brackets'][metric]={p:crossings(LEVELS,[value(p,s)-value('baseline',s) for s in LEVELS])
                for p in POLICIES if p!='baseline'}
        report['cases'][case]=values
        print(case,flush=True)
        for row in values['factorial']['field']:
            print(' field',row['sigma_adu'],row['values'],flush=True)
        for row in values['factorial']['sharpened']:
            print(' sharp',row['sigma_adu'],row['values'],flush=True)
    report['limits']=[
        'Sixteen fixed poses, one scene and one paired noise seed provide an exploratory screen, not a population threshold.',
        'Noise levels use the same underlying random samples; changing dose does not change realization.',
        'Sigma values are synthetic ADU amplitudes, not inferred Saturn read-noise or photon-noise levels.',
        'Sign-change intervals are observed brackets; acceptance gates can make effects nonmonotonic.',
        'Factorial metric differences are descriptive and do not partition physical noise into additive causal fractions.',
        'Final image sampling, raw brightness, reference, AP geometry and field regularization stay fixed.',
        'All comparisons use sixteen frames; full-count records from previous turns are not mixed into image metrics.',
        'No production behavior, output filtering or normalization changes.']
    (root/'analysis.json').write_text(json.dumps(report,indent=2)+'\n')
    Path('results/registration/ap-noise-ladder.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
