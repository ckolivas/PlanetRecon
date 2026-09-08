"""Bounded coupled-inverse probes on the preserved full-count scene iterate."""
import argparse
from io import BytesIO
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from tools.audit_scene_conditioning import frozen_checkpoint
from tools.audit_scene_selections import validate_case
from tools.scene_study import ObservedSceneData
from tools.scene_quadratic import SceneQuadratic
from tools.scene_parallel_reference import ParallelSceneBatch
from tools.scene_retained_cache import RetainedCellFFTBatch
from tools.scene_periodic_preconditioner import PeriodicScenePreconditioner
from tools.scene_preconditioned_probe import probe
from tools.scene_conditioning import cone_residual, correction_bound
from tools.experiment_stages import atomic_write
from tools.study_io import identities,open_study,file_hash,write_json


def run(path,checkpoint,directory):
    import torch
    from planetrecon.runtime import apply_thread_limits
    from planetrecon.physics_audit import peak_rss_bytes
    apply_thread_limits(2)
    original_path=Path('results/p2-selection-endpoints-reference/protocol.json')
    manifest_path=Path('results/p2-full-selection-manifest/report.json')
    original=json.loads(original_path.read_text())
    selected=validate_case(json.loads(manifest_path.read_text()),1001,4.,'feature')['selections'][-1]
    if file_hash(path)!=original['input_sha256'] or file_hash(manifest_path)!=original['manifest_sha256']:
        raise ValueError('frozen input/manifest mismatch')
    for name,expected in original['identities']['dependencies'].items():
        if file_hash(Path(name))!=expected: raise ValueError('original numerical dependency changed: '+name)
    names=['scene_study','scene_fft','scene_quadratic','scene_parallel_reference',
           'scene_retained_cache','scene_periodic_preconditioner','scene_preconditioned_probe',
           'scene_conditioning','scene_iteration_state','audit_scene_conditioning','audit_scene_selections']
    deps=[Path(__file__),Path('docs/scene-periodic-probe-protocol.md')]+[Path('tools')/(n+'.py') for n in names]
    identity=identities(deps)
    if identity['package_source_hash']!=original['identities']['package_source_hash']:
        raise ValueError('original package source changed')
    state=frozen_checkpoint(checkpoint,original,selected)
    protocol={'identities':identity,'input_sha256':file_hash(path),
              'checkpoint_sha256':file_hash(checkpoint),'original_protocol_sha256':file_hash(original_path),
              'manifest_sha256':file_hash(manifest_path),'initial_iteration':142,
              'modes':['full-diagonal','full-periodic','reduced-diagonal','reduced-periodic'],
              'probe_steps':32,'wall_budget_s':600,'cache_bytes':3*1024**3,
              'headroom_bytes':1024**3,'threads':2,'cpu_frame_workers':8,
              'product_relative_tolerance':1e-10,'original_distance_tolerance':1e-5,
              'scene_updated':False,'q3_authorized':False,
              'scope':'Full/reduced conditioning diagnostic; no scene update or scientific qualification.'}
    directory=open_study(directory,protocol)
    started=time.monotonic();deadline=started+600;rows=[];failures=[];initial=None;batch=None
    try:
        available,total=torch.cuda.mem_get_info()
        if available<protocol['cache_bytes']+protocol['headroom_bytes']:
            raise MemoryError('insufficient declared device headroom')
        torch.cuda.reset_peak_memory_stats()
        data=ObservedSceneData(path,range(500),'feature');ops=data.operators(range(500),margin=64)
        images=[data.images[i] for i in range(500)];variances=[data.variances[i] for i in range(500)]
        batch=RetainedCellFFTBatch(ops,device='cuda',cache_bytes=protocol['cache_bytes'])
        problem=SceneQuadratic(ops,images,variances,ridge=selected['mean_ridge'],batch=batch)
        reference_batch=ParallelSceneBatch(ops,workers=8)
        reference=SceneQuadratic(ops,images,variances,ridge=selected['mean_ridge'],batch=reference_batch)
        x=state['x'];gradient=reference.gradient(x);residual=cone_residual(x,gradient)
        free=(x>0)|(gradient<0)
        error=float(np.linalg.norm(problem.gradient(x)-gradient)/max(np.linalg.norm(gradient),1e-30))
        initial={'raw_certificate':reference.certificate(x,gradient=gradient),'gradient_relative_error':error,
                 'active_fraction':float(np.mean(x==0)),'free_fraction':float(np.mean(free)),
                 'ridge':problem.ridge,'majorizer_max':float(problem.majorizer.max()),
                 'gpu_free_before_bytes':available,'gpu_total_bytes':total}
        write_json(directory/'initial.json',initial)
        if error>1e-10: raise ValueError('independent initial gradient parity failed')
        before=time.monotonic();periodic=PeriodicScenePreconditioner(problem,workers=8)
        write_json(directory/'inverse.json',periodic.info()|{'construction_s':time.monotonic()-before})
        for mode in protocol['modes']:
            reduced=mode.startswith('reduced');mask=free if reduced else np.ones(x.shape,bool)
            inverse=periodic if mode.endswith('periodic') else lambda r:r/problem.majorizer
            trace=[]
            def callback(row,y):
                trace.append(row);write_json(directory/('trace-'+mode+'.json'),{'mode':mode,'trace':trace})
            y,info=probe(problem.normal,residual,inverse,free=mask,steps=32,deadline=deadline,callback=callback)
            stream=BytesIO();np.savez_compressed(stream,correction=y)
            atomic_write(directory/('correction-'+mode+'.npz'),stream.getvalue())
            before=time.monotonic();hy=reference.normal(y);gpu_hy=problem.normal(y)
            product_error=float(np.linalg.norm(hy-gpu_hy)/max(np.linalg.norm(hy),1e-30))
            e=residual-hy;denominator=max(np.linalg.norm(residual),1e-30)
            row={'mode':mode,'probe':info,'product_relative_error':product_error,
                 'independent_full_relative_residual':float(np.linalg.norm(e)/denominator),
                 'independent_system_relative_residual':float(np.linalg.norm(e[mask])/denominator),
                 'bound':correction_bound(x,gradient,y,hy,problem.ridge),
                 'independent_check_s':time.monotonic()-before,'probe_complete':info['reason']!='wall_budget',
                 'cache':batch.cache_info(),'cpu_reference':reference_batch.cache_info(),
                 'process_peak_rss_bytes':peak_rss_bytes(),
                 'torch_peak_allocated_bytes':torch.cuda.max_memory_allocated()}
            rows.append(row);write_json(directory/(mode+'.json'),row)
            print(mode,'steps=',info['iterations'],'system_residual=',row['independent_system_relative_residual'],flush=True)
    except Exception as exc:
        failures.append(repr(exc));print(repr(exc),flush=True)
    finally:
        if batch is not None: batch.clear_cache()
    unchanged=(identity==identities(deps) and file_hash(path)==protocol['input_sha256']
               and file_hash(checkpoint)==protocol['checkpoint_sha256']
               and file_hash(original_path)==protocol['original_protocol_sha256']
               and file_hash(manifest_path)==protocol['manifest_sha256'])
    passed=unchanged and not failures and len(rows)==4 and all(r['probe_complete'] and r['product_relative_error']<=1e-10 for r in rows)
    report={'status':'valid' if passed else 'incomplete','source_input_unchanged':unchanged,
            'initial':initial,'rows':rows,'failures':failures,'wall_s':time.monotonic()-started,
            'scene_updated':False,'q3_authorized':False,'scope':protocol['scope']}
    write_json(directory/'report.json',report)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    sys.exit(0 if run(a.input,a.checkpoint,a.out)['status']=='valid' else 1)
