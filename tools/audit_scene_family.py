"""Complete seed/regime/crop numerical pilot on observed-only optical scenes."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing
from pathlib import Path
import sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from tools.scene_study import ObservedSceneData,TRAIN_LARGE,CellFFTBatch,prior_coefficient
from tools.scene_quadratic import SceneQuadratic,solve
from tools.scene_iteration_state import IterationCheckpoint
from tools.experiment_stages import StageStore
from tools.study_io import identities,open_study,write_json,file_hash


def complete_cases(rows):
    expected={(s,d,c) for s in (1001,1002,1003) for d in (4.,8.) for c in ('feature','bland')}
    actual=[(r['seed'],r['dr0'],r['crop']) for r in rows]
    return len(actual)==len(expected) and set(actual)==expected


def case(payload):
    from planetrecon.runtime import apply_thread_limits
    from planetrecon.physics_audit import peak_rss_bytes
    apply_thread_limits(2)
    path,crop,directory,protocol,family_deadline,resume=payload
    start=time.monotonic();deadline=min(family_deadline,start+protocol['case_budget_s'])
    if time.monotonic()>=deadline:raise TimeoutError('family budget exhausted before case')
    data=ObservedSceneData(path,TRAIN_LARGE,crop)
    name=f'{data.cfg.seed}-{int(data.cfg.dr0)}-{crop}'
    case_directory=directory/name
    directory=open_study(case_directory,{'family':protocol,'input_sha256':file_hash(path),'crop':crop},resume=resume and case_directory.exists())
    ops=data.operators(TRAIN_LARGE,margin=64)
    images=[data.images[i] for i in TRAIN_LARGE];variances=[data.variances[i] for i in TRAIN_LARGE]
    ridge=prior_coefficient(protocol['sum_native_strength'],len(TRAIN_LARGE))
    batch=CellFFTBatch(ops,device=protocol['device'],cache_bytes=protocol['cache_bytes'])
    problem=SceneQuadratic(ops,images,variances,ridge=ridge,batch=batch)
    reference=SceneQuadratic(ops,images,variances,ridge=ridge)
    store=StageStore(directory/'stages',{'family':protocol,'case':name,'input_sha256':file_hash(path)},deadline=deadline)
    rows=[];images_out=[]
    for budget in protocol['budgets']:
        def compute():
            checkpoint=IterationCheckpoint(directory/'iterations'/f'{budget}.npz',
                {'family':protocol,'case':name,'mean_ridge':ridge})
            def callback(n,x,certificate):
                write_json(directory/'progress.json',{'budget':budget,'iteration':n,'certificate':certificate})
            x,info=solve(problem,maxiter=budget,tolerance=protocol['tolerance'],deadline=deadline,
                         iteration_checkpoint=checkpoint,checkpoint_interval=100,callback=callback)
            info['reference_certificate']=reference.certificate(x)
            if not info['converged'] and info['reason']=='wall_budget':
                write_json(directory/f'incomplete-{budget}-{time.time_ns()}.json',info)
                raise TimeoutError('wall budget exhausted; compatible iteration state retained')
            return x,info
        try:
            x,info=store.run(f'budget-{budget}',compute)
            rows.append(info);images_out.append(x)
        except TimeoutError as exc:
            rows.append({'maxiter':budget,'converged':False,'reason':str(exc)})
    changes={}
    if len(images_out)==2:
        a,b=images_out
        changes['latent']=float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1.))
        a,b=(ops[0].detector_scene(x) for x in images_out)
        changes['detector']=float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1.))
    passed=(len(images_out)==2 and all(r['converged'] and r['reference_certificate']['relative_solution_error_bound']<=protocol['tolerance'] for r in rows)
            and max(changes.values())<=protocol['image_tolerance'])
    result={'seed':data.cfg.seed,'dr0':data.cfg.dr0,'crop':crop,'status':'valid' if passed else 'incomplete',
            'numerical_passed':bool(passed),'indices':list(TRAIN_LARGE),'n_captured':data.cfg.n_frames,
            'observed_electron_sum':float(sum(y.sum() for y in images)),
            'native_shape':list(ops[0].base.scene_shape),'mean_ridge':ridge,'runs':rows,'relative_changes':changes,
            'cache':batch.cache_info(),'wall_s':time.monotonic()-start,'process_peak_rss_bytes':peak_rss_bytes(),
            'input_unchanged':protocol['input_sha256'][path.name]==file_hash(path),'q3_authorized':False}
    if not result['input_unchanged']:result.update(status='incomplete',numerical_passed=False)
    write_json(directory/'report.json',result)
    batch.clear_cache()
    return result


def run(inputs,directory,*,device='cpu',workers=2,resume=False):
    from planetrecon.hdf5io import filename
    if not 1<=workers<=2:raise ValueError('one or two workers supported by this protocol')
    deps=[Path(__file__),Path('tools/scene_study.py'),Path('tools/scene_fft.py'),Path('tools/scene_quadratic.py'),
          Path('tools/scene_iteration_state.py'),Path('docs/scene-family-pilot-protocol.md')]
    identity=identities(deps)
    paths=[inputs/filename(s,d) for s in (1001,1002,1003) for d in (4.,8.)]
    protocol={'question':'Do all 12 development seed/regime/crop cases certify the frozen 11-frame numerical objective?',
        'identities':identity,'input_sha256':{p.name:file_hash(p) for p in paths},
        'indices':list(TRAIN_LARGE),'sum_native_strength':.0003,'margin':64,'cell_factor':1,
        'variance':'max(observed frame mean,0)+read variance','initialization':'zero',
        'budgets':[750,1500],'tolerance':1e-5,'image_tolerance':1e-4,'workers':workers,'threads_per_worker':2,
        'cache_bytes':256*1024**2,'device':device,'case_budget_s':900.,'family_budget_s':3600.,
        'q3_authorized':False,'scope':'Complete development PILOT matrix, 11 of 500 frames; not complete frame selections, prior qualification, Gate-1, Q2 or production acceptance.'}
    directory=open_study(directory,protocol,resume=resume)
    started=time.monotonic();rows=[];failures=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        pending={pool.submit(case,(p,c,directory,protocol,started+protocol['family_budget_s'],resume)):(p,c)
                 for p in paths for c in ('feature','bland')}
        for future in as_completed(pending):
            p,c=pending[future]
            try:
                row=future.result();rows.append(row)
                print(row['seed'],row['dr0'],c,'passed=',row['numerical_passed'],'iterations=',[r.get('n_iter') for r in row['runs']],flush=True)
            except Exception as exc:
                failures.append({'input':p.name,'crop':c,'error':repr(exc)})
                print(p.name,c,repr(exc),flush=True)
            write_json(directory/'progress.json',{'cases':rows,'failures':failures})
    unchanged=identity==identities(deps) and all(file_hash(p)==protocol['input_sha256'][p.name] for p in paths)
    complete=complete_cases(rows) and not failures
    passed=complete and unchanged and all(r['numerical_passed'] for r in rows)
    report={'status':'valid' if passed else 'incomplete','complete':complete,'source_input_unchanged':unchanged,
        'cases':sorted(rows,key=lambda r:(r['seed'],r['dr0'],r['crop'])),'failures':failures,
        'wall_s':time.monotonic()-started,'q3_authorized':False,'scope':protocol['scope']}
    write_json(directory/f'attempt-{time.time_ns()}.json',report);write_json(directory/'report.json',report)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--inputs',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',choices=('cpu','cuda'),default='cpu');p.add_argument('--workers',type=int,default=2);p.add_argument('--resume',action='store_true')
    a=p.parse_args();r=run(a.inputs,a.out,device=a.device,workers=a.workers,resume=a.resume);sys.exit(0 if r['status']=='valid' else 1)
