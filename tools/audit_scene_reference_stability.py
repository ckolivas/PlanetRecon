"""Frozen certified 1500-cap reference versus a fresh 3000-cap full-count fit."""
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
from tools.scene_state_diagnostics import reference_stage
from tools.experiment_stages import StageStore
from tools.study_io import identities,open_study,file_hash,write_json

FROZEN='abbdcdaad5a4df0099a3fad842e8c9992a8e157531c9a0a0229177b1f11e18aa'


def run(path,manifest_path,directory,*,resume=False):
    import torch
    from planetrecon.runtime import apply_thread_limits
    from planetrecon.physics_audit import peak_rss_bytes
    apply_thread_limits(2)
    old_root=Path('results/p2-cropped-reference');stage=Path('out/p2-cropped-reference/stages/100')
    original=json.loads((old_root/'protocol.json').read_text())
    report=json.loads((old_root/'report.json').read_text())
    case=validate_case(json.loads(manifest_path.read_text()),1001,4.,'feature')
    selection=next(s for s in case['selections'] if s['fraction']==100)
    if not report['source_input_unchanged'] or original['method']!='reference': raise ValueError('original reference evidence required')
    if file_hash(path)!=original['input_sha256'] or file_hash(manifest_path)!=original['manifest_sha256']:
        raise ValueError('original input/selection identity required')
    expected={'sum_native_strength':.0003,'margin':64,'cell_factor':1,'tolerance':1e-5,'image_tolerance':1e-4,'cache_bytes':4*1024**3}
    if any(original[k]!=v for k,v in expected.items()): raise ValueError('original numerical contract differs')
    for name,digest in original['identities']['dependencies'].items():
        if file_hash(Path(name))!=digest: raise ValueError('original numerical source changed: '+name)
    lower=reference_stage(stage,original,selection,FROZEN,cap=1500,iterations=1410)
    files=[path,manifest_path,old_root/'report.json',old_root/'protocol.json',
           *[stage/name for name in ('identity.json','budget-1500.json','budget-1500.npz')]]
    hashes={str(p):file_hash(p) for p in files}
    names=['audit_scene_selections','scene_study','scene_fft','scene_quadratic','scene_cropped_fft',
           'scene_retained_cache','scene_parallel_reference','scene_iteration_state','scene_state_diagnostics']
    deps=[Path(__file__),Path('docs/scene-reference-stability-protocol.md')]+[Path('tools')/(n+'.py') for n in names]
    identity=identities(deps)
    if identity['package_source_hash']!=original['identities']['package_source_hash']: raise ValueError('original package changed')
    protocol={'identities':identity,'input_hashes':hashes,'selection':selection,'caps':[1500,3000],
              'lower_cap_reused':True,'upper_initialization':'zero','margin':64,'cell_factor':1,'sum_native_strength':.0003,
              'tolerance':1e-5,'image_tolerance':1e-4,'fit_budget_s':900.,'wall_budget_s':1200.,
              'cache_bytes':4*1024**3,'headroom_bytes':1024**3,'threads':2,'cpu_frame_workers':8,
              'scope':'One full-count reference stability endpoint; no full-family or scientific qualification.','q3_authorized':False}
    directory=open_study(directory,protocol,resume=resume);started=time.monotonic();deadline=started+1200
    failures=[];lower_certificate=None;upper_info=None;changes={};batch=None
    try:
        if torch.cuda.mem_get_info()[0]<protocol['cache_bytes']+protocol['headroom_bytes']:
            raise MemoryError('insufficient declared GPU headroom')
        data=ObservedSceneData(path,range(500),'feature');indices=selection['indices'];ops=data.operators(indices,margin=64)
        images=[data.images[i] for i in indices];variances=[data.variances[i] for i in indices]
        batch=CroppedCellFFTBatch(ops,device='cuda',cache_bytes=protocol['cache_bytes'])
        problem=SceneQuadratic(ops,images,variances,ridge=selection['mean_ridge'],batch=batch)
        reference_batch=ParallelSceneBatch(ops,workers=8)
        reference=SceneQuadratic(ops,images,variances,ridge=selection['mean_ridge'],batch=reference_batch)
        lower_certificate=reference.certificate(lower);write_json(directory/'lower-certificate.json',lower_certificate)
        if not (lower_certificate['feasible'] and np.isfinite(lower_certificate['relative_solution_error_bound']) and lower_certificate['relative_solution_error_bound']<=1e-5):
            raise ValueError('frozen lower image no longer meets independent criterion')
        store=StageStore(directory/'stages',{'protocol':protocol},deadline=deadline)
        def compute():
            trace=[]
            def progress(n,x,certificate):
                trace.append({'iteration':n,**certificate});write_json(directory/'trace-3000.json',{'trace':trace})
            checkpoint=IterationCheckpoint(directory/'iterations/3000.npz',{'protocol':protocol})
            x,info=solve(problem,maxiter=3000,tolerance=1e-5,callback=progress,
                         deadline=min(deadline,time.monotonic()+900),iteration_checkpoint=checkpoint,checkpoint_interval=100)
            info['trace']=trace;info['reference_certificate']=reference.certificate(x)
            return x,info
        upper,upper_info=store.run('budget-3000',compute)
        changes['latent']=float(np.linalg.norm(lower-upper)/max(np.linalg.norm(upper),1.))
        a,b=(ops[0].detector_scene(x) for x in (lower,upper))
        changes['detector']=float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1.))
        write_json(directory/'upper-fit.json',upper_info)
    except Exception as exc:
        failures.append(repr(exc));print(repr(exc),flush=True)
    cache=batch.cache_info() if batch is not None else None
    if batch is not None:batch.clear_cache()
    unchanged=identity==identities(deps) and all(file_hash(Path(p))==digest for p,digest in hashes.items())
    passed=bool(unchanged and not failures and upper_info and upper_info['converged'] and
                upper_info['reference_certificate']['feasible'] and upper_info['reference_certificate']['relative_solution_error_bound']<=1e-5 and
                all(np.isfinite(v) and v<=1e-4 for v in changes.values()) and len(changes)==2)
    result={'status':'valid' if passed else 'incomplete','source_input_unchanged':unchanged,'lower_cap_reused':True,
            'lower_certificate':lower_certificate,'upper_fit':upper_info,'relative_changes':changes,'failures':failures,
            'wall_s':time.monotonic()-started,'cache':cache,'process_peak_rss_bytes':peak_rss_bytes(),
            'scope':protocol['scope'],'q3_authorized':False}
    write_json(directory/'report.json',result);print(result['status'],changes,flush=True);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--resume',action='store_true');a=p.parse_args()
    sys.exit(0 if run(a.input,a.manifest,a.out,resume=a.resume)['status']=='valid' else 1)
