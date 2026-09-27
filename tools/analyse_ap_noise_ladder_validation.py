"""Audit the independent refinement-only check and compare development effects."""
import json
from pathlib import Path

import numpy as np

from tools.ap_noise_ladder import hashes,key,CASES
from tools.validate_ap_noise_ladder import CONFIG
from tools.analyse_joint_saturn import sharp_identity
from tools.joint_saturn_experiment import digest
from tools.validate_local_warp_centring import image_errors,weighted_field_errors


def main():
    root=Path('out/ap-noise-ladder-validation')
    manifest=json.loads((root/'manifest.json').read_text())
    screen_path=Path('out/ap-noise-ladder/analysis.json')
    screen=json.loads(screen_path.read_text())
    if (manifest['hashes']!=hashes() or manifest['source_sha256']!=digest('tools/validate_ap_noise_ladder.py')
            or manifest['config']!=CONFIG or manifest['screen_sha256']!=digest(screen_path)
            or set(CONFIG['positions'])&set(screen['manifest']['config']['positions'])
            or CONFIG['seed']==screen['manifest']['config']['seed']):
        raise ValueError('Invalid validation identity or separation')
    if manifest['geometry_sha256']!=digest('out/ap-oracle-observations/geometry.npz'):
        raise ValueError('Geometry changed')
    for case in CASES:
        if manifest['prior_hashes'][case]!=digest(f'out/ap-fractional-validation/{case}.npz'):
            raise ValueError('Truth data changed')
    report=dict(manifest=manifest,analyzer_sha256=digest(__file__),cases={})
    recipe=json.loads(Path('out/saturn-coherent-field/sharpened/sharpening_report.json').read_text())
    for case in CASES:
        directory=root/case
        values=json.loads((directory/'report.json').read_text())
        if values['identity']!=manifest or values['case']!=case or values['raw_sha256']!=digest(directory/'arrays.npz'):
            raise ValueError('Incomplete or changed validation')
        values['raw_report_sha256']=digest(directory/'report.json')
        values['sharpening']=sharp_identity(directory,recipe)
        names={key(p,s) for p in CONFIG['policies'] for s in CONFIG['levels']}
        if set(values['variants'])!=names or set(values['sharpening']['images'])!=names|{'target'}:
            raise ValueError('Incomplete sharpening')
        with np.load(directory/'arrays.npz') as data:
            mask,target,truth=data['mask'],data['target'],data['truth']
            for name,variant in values['variants'].items():
                expected=dict(image=image_errors(data[name],target,mask),clean=image_errors(data['clean_'+name],target,mask),
                    field=weighted_field_errors(data['field_'+name],truth,np.ones(16)/16,mask))
                if any(variant[k]!=v for k,v in expected.items()):
                    raise ValueError('Saved array/metric mismatch')
                rows=variant.pop('statistics')
                if len(rows)!=16 or len(truth)!=16:
                    raise ValueError('Incomplete frame count')
                variant['fits']=dict(guard_rejections=sum(not r['fit']['field_guard_accepted'] for r in rows),
                    insufficient_points=sum(r['fit']['fallback'] for r in rows))
        with np.load(directory/'sharpened/target_stages.npz') as data:
            target=data['sharpened'].mean(2)
        values['stage_sha256']={'target':digest(directory/'sharpened/target_stages.npz')}
        for name,variant in values['variants'].items():
            path=directory/'sharpened'/f'{name}_stages.npz'
            values['stage_sha256'][name]=digest(path)
            with np.load(path) as data:
                variant['sharpened_error_linear_0_1']=image_errors(data['sharpened'].mean(2),target,mask)
        values['relative_change_percent']={}
        for metric in ('field','clean','sharpened'):
            def value(variants,policy,sigma):
                v=variants[key(policy,sigma)]
                return v['field']['total_vector_rmse_px'] if metric=='field' else (
                    v['clean']['rmse_adu'] if metric=='clean' else v['sharpened_error_linear_0_1']['rmse_adu'])
            values['relative_change_percent'][metric]=[dict(sigma_adu=s,
                development=100*(value(screen['cases'][case]['variants'],'refined',s)/value(screen['cases'][case]['variants'],'baseline',s)-1),
                validation=100*(value(values['variants'],'refined',s)/value(values['variants'],'baseline',s)-1)) for s in CONFIG['levels']]
        print(case,json.dumps(values['relative_change_percent']),flush=True)
        report['cases'][case]=values
    report['limits']=[
        'This validates one policy on a fresh seed and unused phases, with the same scene and constant PSF.',
        'Noise seed and motion phases change together; their individual contributions to development/validation differences are not separated.',
        'Each stack has sixteen poses; no statistical confidence interval or real-capture noise threshold is inferred.',
        'The refinement-only policy was selected after the development ladder; this validation did not retune it.',
        'Sharpened image error, clean-image bias and motion accuracy can disagree.',
        'No production, output filtering or brightness normalization changes.']
    (root/'analysis.json').write_text(json.dumps(report,indent=2)+'\n')
    Path('results/registration/ap-noise-ladder-validation.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
