"""Gated per-case execution of the frozen complete observed-selection family."""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from tools.audit_scene_selections import validate_case
from tools.scene_study import ObservedSceneData
from tools.scene_quadratic import SceneQuadratic,solve
from tools.scene_cropped_fft import CroppedCellFFTBatch
from tools.scene_parallel_reference import ParallelSceneBatch
from tools.scene_iteration_state import IterationCheckpoint
from tools.experiment_stages import StageStore
from tools.study_io import identities,open_study,file_hash,write_json


def select_case(manifest,index):
    expected=[(seed,dr0,crop) for seed in (1001,1002,1003) for dr0 in (4.,8.) for crop in ('feature','bland')]
    if int(index)!=index or not 0<=index<len(expected): raise ValueError('declared case index required')
    actual=[(c['seed'],c['dr0'],c['crop']) for c in manifest['cases']]
    if actual!=expected: raise ValueError('frozen complete case catalog/order required')
    return validate_case(manifest,*expected[int(index)])


def require_qualified_stability(proof,protocol):
    if proof.get('status')!='valid' or proof.get('source_input_unchanged') is not True:
        raise ValueError('qualified unchanged full-count reference stability required')
    expected={'caps':[1500,3000],'margin':64,'cell_factor':1,'sum_native_strength':.0003,
              'tolerance':1e-5,'image_tolerance':1e-4,'cache_bytes':4*1024**3,'upper_initialization':'zero'}
    if any(protocol.get(k)!=v for k,v in expected.items()):raise ValueError('qualified numerical contract differs')
    upper=proof.get('upper_fit',{})
    if not upper.get('converged') or upper.get('solver')!='extended_scene_projected_acceleration_v4' or upper.get('maxiter')!=3000:
        raise ValueError('qualified independent upper fit required')
    for cert in (proof.get('lower_certificate',{}),upper.get('reference_certificate',{})):
        bound=cert.get('relative_solution_error_bound',float('inf'))
        if cert.get('feasible') is not True or not np.isfinite(bound) or not 0<=bound<=1e-5:
            raise ValueError('qualified independent distance bounds required')
    for key in ('latent','detector'):
        change=proof.get('relative_changes',{}).get(key,float('inf'))
        if not np.isfinite(change) or not 0<=change<=1e-4:
            raise ValueError('qualified image stability required')


def run(path,manifest_path,directory,*,case_index,resume=False):
    import torch
    from planetrecon.runtime import apply_thread_limits
    from planetrecon.physics_audit import peak_rss_bytes
    apply_thread_limits(2)
    case=select_case(json.loads(manifest_path.read_text()),case_index)
    if file_hash(path)!=case['input_sha256']: raise ValueError('input does not match frozen case')
    proof_root=Path('results/p2-reference-stability')
    proof_paths=[proof_root/'report.json',proof_root/'protocol.json']
    proof=json.loads(proof_paths[0].read_text());previous=json.loads(proof_paths[1].read_text())
    require_qualified_stability(proof,previous)
    for name,digest in previous['identities']['dependencies'].items():
        if file_hash(Path(name))!=digest:raise ValueError('qualified numerical source changed: '+name)
    names=['audit_scene_selections','scene_study','scene_fft','scene_quadratic','scene_cropped_fft',
           'scene_retained_cache','scene_parallel_reference','scene_iteration_state']
    deps=[Path(__file__),Path('docs/scene-selection-family-protocol.md')]+[Path('tools')/(n+'.py') for n in names]
    # Include the prerequisite closure too, so changes during a case cannot be
    # hidden merely because its loader code was only used at startup.
    deps+=list(map(Path,previous['identities']['dependencies']))
    identity=identities(deps)
    if identity['package_source_hash']!=previous['identities']['package_source_hash']:raise ValueError('qualified package changed')
    proof_hashes={str(p):file_hash(p) for p in proof_paths}
    protocol={'identities':identity,'qualification_hashes':proof_hashes,'case_index':case_index,
              'case':[case['seed'],case['dr0'],case['crop']],'input_sha256':file_hash(path),'manifest_sha256':file_hash(manifest_path),
              'fractions':[5,10,25,50,100],'budgets':[1500,3000],'budget_unit':'iterations',
              'margin':64,'cell_factor':1,'sum_native_strength':.0003,'tolerance':1e-5,'image_tolerance':1e-4,
              'cache_bytes':4*1024**3,'headroom_bytes':1024**3,'threads':2,'cpu_frame_workers':8,
              'fit_budget_s':{'5':300.,'10':300.,'25':600.,'50':900.,'100':900.},'wall_budget_s':3600.,
              'scope':'One case of the complete 12-case/60-selection numerical family; no scientific qualification.','q3_authorized':False}
    directory=open_study(directory,protocol,resume=resume);started=time.monotonic();deadline=started+3600
    rows=[];failures=[];batch=None
    try:
        data=ObservedSceneData(path,range(500),case['crop'])
        for selection in case['selections']:
            if time.monotonic()>=deadline:raise TimeoutError('case budget exhausted before next fraction')
            fraction=selection['fraction'];indices=selection['indices']
            torch.cuda.empty_cache()
            if torch.cuda.mem_get_info()[0]<protocol['cache_bytes']+protocol['headroom_bytes']:
                raise MemoryError('insufficient declared GPU headroom')
            ops=data.operators(indices,margin=64);images=[data.images[i] for i in indices];variances=[data.variances[i] for i in indices]
            batch=CroppedCellFFTBatch(ops,device='cuda',cache_bytes=protocol['cache_bytes'])
            problem=SceneQuadratic(ops,images,variances,ridge=selection['mean_ridge'],batch=batch)
            independent_batch=ParallelSceneBatch(ops,workers=8)
            independent=SceneQuadratic(ops,images,variances,ridge=selection['mean_ridge'],batch=independent_batch)
            store=StageStore(directory/'stages'/str(fraction),{'protocol':protocol,'selection':selection},deadline=deadline)
            outputs=[];fits=[]
            for cap in protocol['budgets']:
                def compute():
                    trace=[]
                    def progress(n,x,cert):
                        trace.append({'iteration':n,**cert});write_json(directory/f'trace-{fraction}-{cap}.json',{'trace':trace})
                    checkpoint=IterationCheckpoint(directory/'iterations'/f'{fraction}-{cap}.npz',{'protocol':protocol,'selection':selection})
                    x,info=solve(problem,maxiter=cap,tolerance=1e-5,callback=progress,iteration_checkpoint=checkpoint,
                                 checkpoint_interval=100,deadline=min(deadline,time.monotonic()+protocol['fit_budget_s'][str(fraction)]))
                    info['trace']=trace;info['reference_certificate']=independent.certificate(x)
                    return x,info
                try:
                    x,info=store.run(f'budget-{cap}',compute);outputs.append(x);fits.append(info)
                    print(case_index,fraction,cap,'bound=',info['reference_certificate']['relative_solution_error_bound'],flush=True)
                except TimeoutError as exc:fits.append({'maxiter':cap,'converged':False,'reason':str(exc)})
            changes={}
            if len(outputs)==2:
                changes['latent']=float(np.linalg.norm(outputs[0]-outputs[1])/max(np.linalg.norm(outputs[1]),1.))
                a,b=(ops[0].detector_scene(x) for x in outputs)
                changes['detector']=float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1.))
            passed=bool(len(outputs)==2 and all(f['converged'] and f['reference_certificate']['feasible'] and
                        f['reference_certificate']['relative_solution_error_bound']<=1e-5 for f in fits) and
                        all(np.isfinite(v) and v<=1e-4 for v in changes.values()))
            row={'fraction':fraction,'indices':indices,'n_used':len(indices),'n_captured':500,'mean_ridge':selection['mean_ridge'],
                 'observed_electron_sum':float(sum(y.sum() for y in images)),'runs':fits,'relative_changes':changes,
                 'numerical_passed':passed,'cache':batch.cache_info(),'process_peak_rss_bytes':peak_rss_bytes()}
            rows.append(row);write_json(directory/f'fraction-{fraction}.json',row)
            batch.clear_cache();batch=None
            del ops,problem,independent,independent_batch,outputs
    except Exception as exc:failures.append(repr(exc));print(repr(exc),flush=True)
    finally:
        if batch is not None:batch.clear_cache()
    unchanged=identity==identities(deps) and file_hash(path)==protocol['input_sha256'] and file_hash(manifest_path)==protocol['manifest_sha256'] and all(file_hash(Path(p))==h for p,h in proof_hashes.items())
    complete=[r['fraction'] for r in rows]==protocol['fractions'] and not failures
    report={'status':'valid' if complete and unchanged and all(r['numerical_passed'] for r in rows) else 'incomplete',
            'complete':complete,'source_input_unchanged':unchanged,'case_index':case_index,'rows':rows,'failures':failures,
            'wall_s':time.monotonic()-started,'scope':protocol['scope'],'q3_authorized':False}
    write_json(directory/'report.json',report);return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--case-index',type=int,choices=range(12),required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
    sys.exit(0 if run(a.input,a.manifest,a.out,case_index=a.case_index,resume=a.resume)['status']=='valid' else 1)
