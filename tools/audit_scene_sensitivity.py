"""Prospective observed-data prior/domain/sampling/photon development studies."""
import argparse
from pathlib import Path
import sys,time,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from tools.scene_study import (ObservedSceneData,TRAIN_SMALL,TRAIN_LARGE,SELECTION,ASSESSMENT,
    CellFFTBatch,prior_coefficient,prediction_score,select_candidate)
from tools.scene_quadratic import SceneQuadratic,solve
from tools.experiment_stages import StageStore
from tools.study_io import identities,open_study,write_json,file_hash


def run(path,directory,*,phase='prior',prior_report=None,device='cpu',resume=False,
        budgets=(300,600),wall_budget_s=1800.,fit_budget_s=300.,strengths=(.0003,.003,.03)):
    from planetrecon.runtime import apply_thread_limits
    from planetrecon.physics_audit import peak_rss_bytes
    apply_thread_limits(2)
    if phase not in ('prior','domain','sampling','photons') or len(budgets)!=2 or budgets[0]<1 or budgets[1]<=budgets[0]:
        raise ValueError('invalid phase or budgets')
    if not np.isfinite([wall_budget_s,fit_budget_s]).all() or min(wall_budget_s,fit_budget_s)<=0:
        raise ValueError('positive finite time budgets required')
    if not strengths or any(not np.isfinite(v) or v<=0 for v in strengths) or len(set(strengths))!=len(strengths):
        raise ValueError('distinct positive finite prior strengths required')
    strength=None
    if phase!='prior':
        if prior_report is None:raise ValueError('qualified prior report required')
        prior=json.loads(Path(prior_report).read_text())
        if prior['status']!='valid' or prior['selected_strength'] is None:raise ValueError('prior selection is incomplete')
        if prior['input_sha256']!=file_hash(path):raise ValueError('prior report input mismatch')
        strength=prior['selected_strength']
    dependencies=[Path(__file__),Path('tools/scene_fft.py'),Path('tools/scene_study.py'),Path('tools/scene_quadratic.py'),Path('tools/scene_iteration_state.py'),Path('docs/scene-sensitivity-protocol.md'),Path('docs/scene-prior-extension-protocol.md')]
    identity=identities(dependencies)
    protocol={'phase':phase,'identities':identity,'input_sha256':file_hash(path),
              'prior_report_sha256':None if prior_report is None else file_hash(prior_report),
              'fixed_strength':strength,'strength_grid':list(strengths),'device':device,'cache_bytes':256*1024**2,'threads':2,
              'train_small':list(TRAIN_SMALL),'train_large':list(TRAIN_LARGE),'selection':list(SELECTION),'assessment':list(ASSESSMENT),
              'budgets':list(budgets),'wall_budget_s':wall_budget_s,'fit_budget_s':fit_budget_s,
              'solution_tolerance':1e-5,'image_stability_tolerance':1e-4,'material_image_change_threshold':.01,
              'scope':'Development pilot; fixed scalar observed-data variance, zero prior mean, no latent truth or expected-image values. No Gate-1/Q2/Q3 or production-setting qualification.',
              'q3_authorized':False}
    directory=open_study(directory,protocol,resume=resume)
    start=time.monotonic();deadline=start+wall_budget_s
    train=TRAIN_LARGE if phase=='photons' else TRAIN_SMALL
    # Assessment observations are deliberately absent until after prior selection.
    data=ObservedSceneData(path,tuple(train)+SELECTION)
    if phase=='prior':configs=[(f'ridge-{v}',None,1,TRAIN_SMALL,v) for v in strengths]
    elif phase=='domain':configs=[(f'margin-{m}',m,1,TRAIN_SMALL,strength) for m in (None,0,32,64)]
    elif phase=='sampling':configs=[(f'factor-{f}',64,f,TRAIN_SMALL,strength) for f in (1,2,4)]
    else:configs=[(f'frames-{len(t)}',64,1,t,strength) for t in (TRAIN_SMALL,TRAIN_LARGE)]
    store=StageStore(directory/'stages',protocol,deadline=deadline)
    rows=[];detectors={};latent={}
    for name,margin,factor,indices,lam in configs:
        ops=data.operators(indices,margin=margin,factor=factor)
        batch=CellFFTBatch(ops,device=device,cache_bytes=protocol['cache_bytes'])
        images=[data.images[i] for i in indices];variances=[data.variances[i] for i in indices]
        ridge=prior_coefficient(lam,len(indices),factor)
        problem=SceneQuadratic(ops,images,variances,ridge=ridge,batch=batch)
        reference=SceneQuadratic(ops,images,variances,ridge=ridge)
        runs=[];outputs=[]
        for budget in budgets:
            def compute():
                def callback(n,x,cert):
                    write_json(directory/'progress.json',{'case':name,'budget':budget,'iteration':n,'certificate':cert})
                from tools.scene_iteration_state import IterationCheckpoint
                checkpoint=IterationCheckpoint(directory/'iterations'/f'{name}-{budget}.npz',
                    {'protocol':protocol,'case':name,'mean_cell_ridge':ridge})
                x,info=solve(problem,maxiter=budget,tolerance=1e-5,callback=callback,
                             deadline=min(deadline,time.monotonic()+fit_budget_s),iteration_checkpoint=checkpoint)
                info['reference_certificate']=reference.certificate(x)
                if not info['converged'] and info['reason']=='wall_budget':
                    write_json(directory/f'incomplete-{name}-{budget}-{time.time_ns()}.json',info)
                    raise TimeoutError('wall budget exhausted; compatible iteration state retained')
                return x,info
            try:
                x,info=store.run(f'{name}-{budget}',compute)
                runs.append(info);outputs.append(x)
            except TimeoutError as exc:
                runs.append({'converged':False,'reason':str(exc),'maxiter':budget})
        change=None if len(outputs)!=2 else float(np.linalg.norm(outputs[0]-outputs[1])/max(np.linalg.norm(outputs[1]),1.))
        numerical=(len(outputs)==2 and all(r['converged'] and r['reference_certificate']['relative_solution_error_bound']<=1e-5 for r in runs)
                   and change<=protocol['image_stability_tolerance'])
        score=None
        if outputs:
            x=outputs[-1]
            selection_ops=data.operators(SELECTION,margin=margin,factor=factor)
            selection_batch=CellFFTBatch(selection_ops,device=device,cache_bytes=protocol['cache_bytes'])
            score=prediction_score(selection_ops,[data.images[i] for i in SELECTION],[data.variances[i] for i in SELECTION],x,selection_batch)
            selection_batch.clear_cache()
            detectors[name]=ops[0].detector_scene(x)
            latent[name]=x
        row={'case':name,'margin':margin,'factor':factor,'indices':list(indices),
             'sum_native_strength':lam,'mean_cell_ridge':ridge,'native_shape':list(ops[0].base.scene_shape),
             'scene_shape':list(ops[0].scene_shape),'numerical_passed':bool(numerical),
             'runs':runs,'image_relative_budget_change':change,'selection_score':score,'cache':batch.cache_info()}
        rows.append(row);write_json(directory/f'{name}.json',row)
        print(phase,name,'passed=',numerical,'iterations=',[r.get('n_iter') for r in runs],'selection=',score,flush=True)
        batch.clear_cache()
        del batch,problem,reference
    selected=None;assessment=None
    if phase=='prior' and all(r['numerical_passed'] for r in rows):
        selected=select_candidate(rows)
        chosen=next(r for r in rows if r['sum_native_strength']==selected)
        held=ObservedSceneData(path,ASSESSMENT)
        ops=held.operators(ASSESSMENT)
        batch=CellFFTBatch(ops,device=device,cache_bytes=protocol['cache_bytes'])
        assessment={'indices':list(ASSESSMENT),'selected_strength':selected,
                    'score':prediction_score(ops,[held.images[i] for i in ASSESSMENT],
                        [held.variances[i] for i in ASSESSMENT],latent[chosen['case']],batch)}
        batch.clear_cache()
    comparisons={}
    if len(detectors)==len(configs) and phase!='prior':
        anchor=detectors[configs[0][0]]
        comparisons={name:float(np.linalg.norm(image-anchor)/max(np.linalg.norm(anchor),1.)) for name,image in detectors.items()}
    unchanged=identity==identities(dependencies) and protocol['input_sha256']==file_hash(path)
    passed=unchanged and all(r['numerical_passed'] for r in rows)
    if phase=='domain' and comparisons.get('margin-64',float('inf'))>1e-4:passed=False
    result={'status':'valid' if passed else 'incomplete','scope':protocol['scope'],
            'input_sha256':protocol['input_sha256'],'source_input_unchanged':unchanged,
            'selected_strength':selected if phase=='prior' else strength,'assessment':assessment,
            'rows':rows,'detector_relative_changes':comparisons,'q3_authorized':False,
            'wall_s':time.monotonic()-start,'process_peak_rss_bytes':peak_rss_bytes()}
    write_json(directory/f'attempt-{time.time_ns()}.json',result);write_json(directory/'report.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--phase',choices=('prior','domain','sampling','photons'),default='prior')
    p.add_argument('--prior-report',type=Path);p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    p.add_argument('--budgets',nargs=2,type=int,default=(300,600));p.add_argument('--wall-budget-s',type=float,default=1800.)
    p.add_argument('--strengths',type=float,nargs='+',default=(.0003,.003,.03))
    p.add_argument('--fit-budget-s',type=float,default=300.);p.add_argument('--resume',action='store_true')
    a=p.parse_args();r=run(a.input,a.out,phase=a.phase,prior_report=a.prior_report,device=a.device,
        resume=a.resume,budgets=a.budgets,wall_budget_s=a.wall_budget_s,fit_budget_s=a.fit_budget_s,strengths=a.strengths)
    sys.exit(0 if r['status']=='valid' else 1)
