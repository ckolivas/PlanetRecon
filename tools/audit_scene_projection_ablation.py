"""Bounded window/projection ablation and gated full-count endpoint."""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from tools.audit_scene_selections import validate_case
from tools.scene_study import ObservedSceneData
from tools.scene_quadratic import SceneQuadratic
from tools.scene_projected_newton import solve as newton
from tools.scene_gradient_projection import solve as projection
from tools.scene_parallel_reference import ParallelSceneBatch
from tools.scene_window_preconditioner import WindowAveragedPreconditioner
from tools.scene_retained_cache import RetainedCellFFTBatch
from tools.experiment_stages import StageStore
from tools.study_io import identities,open_study,write_json,file_hash


def run(path,manifest_path,directory,*,fraction=5,resume=False):
    if fraction not in (5,100): raise ValueError('only declared endpoint fractions are supported')
    import torch
    from planetrecon.runtime import apply_thread_limits
    from planetrecon.physics_audit import peak_rss_bytes
    apply_thread_limits(2)
    names=['audit_scene_selections','scene_study','scene_fft','scene_quadratic',
           'scene_projected_newton','scene_gradient_projection','scene_conditioning',
           'scene_parallel_reference','scene_periodic_preconditioner',
           'scene_window_preconditioner','scene_retained_cache']
    deps=[Path(__file__),Path('docs/scene-projection-ablation-protocol.md'),
          Path('docs/scene-gradient-projection.md'),Path('docs/scene-projection-full-count-protocol.md')]+[Path('tools')/(n+'.py') for n in names]
    identity=identities(deps)
    case=validate_case(json.loads(manifest_path.read_text()),1001,4.,'feature')
    selection=next(s for s in case['selections'] if s['fraction']==fraction)
    modes=[('window_newton',newton),('window_projection',projection)] if fraction==5 else [('window_projection',projection)]
    qualification_path=Path('results/p2-projection-ablation/report.json') if fraction==100 else None
    if qualification_path is not None:
        qualified=json.loads(qualification_path.read_text())
        previous=json.loads(qualification_path.with_name('protocol.json').read_text())
        if not qualified['source_input_unchanged'] or 'window_projection' not in qualified['qualified_modes']:
            raise ValueError('25-frame candidate prerequisite not met')
        if previous['identities']['package_source_hash']!=identity['package_source_hash']:
            raise ValueError('qualified package source changed')
        for name,digest in previous['identities']['dependencies'].items():
            if name=='tools/audit_scene_projection_ablation.py': continue  # explicitly declared harness refactor
            if file_hash(Path(name))!=digest: raise ValueError('qualified numerical source changed: '+name)
    if file_hash(path)!=case['input_sha256']: raise ValueError('input does not match observed manifest')
    protocol={'identities':identity,'input_sha256':file_hash(path),'manifest_sha256':file_hash(manifest_path),
              'selection':selection,'case':[1001,4.,'feature'],'modes':[name for name,_ in modes],'fraction':fraction,
              'qualification_sha256':file_hash(qualification_path) if qualification_path is not None else None,
              'budgets':[750,1500],'sum_native_strength':.0003,'margin':64,'cell_factor':1,
              'tolerance':1e-5,'image_tolerance':1e-4,'fit_budget_s':300.,'wall_budget_s':1200. if fraction==5 else 900.,
              'device':'cuda','threads':2,'cpu_frame_workers':8,'cache_bytes':3*1024**3,'headroom_bytes':1024**3,
              'inner_steps':32,'projection_steps':8,'exact_iteration_resume':False,
              'scope':f'One-case {len(selection["indices"])}-frame projection endpoint; no full-family or scientific qualification.',
              'q3_authorized':False}
    directory=open_study(directory,protocol,resume=resume);started=time.monotonic();deadline=started+protocol['wall_budget_s']
    rows=[];failures=[];batch=None
    try:
        if torch.cuda.mem_get_info()[0]<protocol['cache_bytes']+protocol['headroom_bytes']:
            raise MemoryError('insufficient declared GPU headroom')
        data=ObservedSceneData(path,range(500),'feature');indices=selection['indices']
        ops=data.operators(indices,margin=64);images=[data.images[i] for i in indices]
        variances=[data.variances[i] for i in indices]
        batch=RetainedCellFFTBatch(ops,device='cuda',cache_bytes=protocol['cache_bytes'])
        problem=SceneQuadratic(ops,images,variances,ridge=selection['mean_ridge'],batch=batch)
        reference_batch=ParallelSceneBatch(ops,workers=8)
        reference=SceneQuadratic(ops,images,variances,ridge=selection['mean_ridge'],batch=reference_batch)
        inverse=WindowAveragedPreconditioner(problem,workers=8)
        for mode,solve in modes:
            store=StageStore(directory/'stages'/mode,{'protocol':protocol,'mode':mode},deadline=deadline)
            outputs=[];fits=[]
            for cap in protocol['budgets']:
                def compute():
                    trace=[];inner_trace=[]
                    def progress(row,x):
                        trace.append(row);write_json(directory/f'trace-{mode}-{cap}.json',{'trace':trace})
                    def inner_progress(row):
                        inner_trace.append(row);write_json(directory/f'inner-trace-{mode}-{cap}.json',{'trace':inner_trace})
                    x,info=solve(problem,max_products=cap,inner_steps=32,tolerance=protocol['tolerance'],
                                 preconditioner=inverse,callback=progress,inner_callback=inner_progress,
                                 deadline=min(deadline,time.monotonic()+protocol['fit_budget_s']))
                    info['reference_certificate']=reference.certificate(x)
                    return x,info
                try:
                    x,info=store.run(f'budget-{cap}',compute);outputs.append(x);fits.append(info)
                    print(mode,cap,'bound=',info['reference_certificate']['relative_solution_error_bound'],flush=True)
                except TimeoutError as exc:
                    fits.append({'max_products':cap,'converged':False,'reason':str(exc)})
            changes={}
            if len(outputs)==2:
                changes['latent']=float(np.linalg.norm(outputs[0]-outputs[1])/max(np.linalg.norm(outputs[1]),1.))
                a,b=(ops[0].detector_scene(x) for x in outputs)
                changes['detector']=float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1.))
            passed=len(outputs)==2 and all(f['converged'] and f['reference_certificate']['feasible'] and
                f['reference_certificate']['relative_solution_error_bound']<=protocol['tolerance'] for f in fits) and max(changes.values())<=protocol['image_tolerance']
            row={'mode':mode,'n_used':len(indices),'numerical_passed':bool(passed),'runs':fits,'relative_changes':changes,
                 'preconditioner':inverse.info(),'cache':batch.cache_info(),'cpu_reference':reference_batch.cache_info(),
                 'process_peak_rss_bytes':peak_rss_bytes()}
            rows.append(row);write_json(directory/(mode+'.json'),row)
    except Exception as exc:
        failures.append(repr(exc));print(repr(exc),flush=True)
    finally:
        if batch is not None: batch.clear_cache()
    unchanged=identity==identities(deps) and file_hash(path)==protocol['input_sha256'] and file_hash(manifest_path)==protocol['manifest_sha256']
    if qualification_path is not None:
        unchanged=unchanged and file_hash(qualification_path)==protocol['qualification_sha256']
    complete=len(rows)==len(modes) and not failures
    qualified=[r['mode'] for r in rows if r['numerical_passed']] if unchanged and not failures else []
    report={'status':'valid' if complete and len(qualified)==len(modes) else 'incomplete',
            'complete':complete,'source_input_unchanged':unchanged,'qualified_modes':qualified,
            'rows':rows,'failures':failures,'wall_s':time.monotonic()-started,'q3_authorized':False,'scope':protocol['scope']}
    write_json(directory/f'attempt-{time.time_ns()}.json',report);write_json(directory/'report.json',report)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--fraction',type=int,choices=(5,100),default=5)
    p.add_argument('--resume',action='store_true');a=p.parse_args()
    sys.exit(0 if run(a.input,a.manifest,a.out,fraction=a.fraction,resume=a.resume)['status']=='valid' else 1)
