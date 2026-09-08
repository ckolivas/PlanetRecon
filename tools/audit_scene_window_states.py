"""Bounded early/late-state, window and boundary diagnostics on frozen objectives."""
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
from tools.scene_state_diagnostics import reference_stage,fourier_energy
from tools.scene_study import ObservedSceneData
from tools.scene_quadratic import SceneQuadratic
from tools.scene_parallel_reference import ParallelSceneBatch
from tools.scene_retained_cache import RetainedCellFFTBatch
from tools.scene_periodic_preconditioner import PeriodicScenePreconditioner
from tools.scene_window_preconditioner import WindowAveragedPreconditioner
from tools.scene_preconditioned_probe import probe
from tools.scene_conditioning import cone_residual
from tools.experiment_stages import atomic_write
from tools.study_io import identities,open_study,file_hash,write_json

FROZEN_25='b98d85069d2d16e329491f6597c22dad66892cda74d5d19ff245d188ab49f9bb'


def run(path,directory):
    import torch
    from planetrecon.runtime import apply_thread_limits
    from planetrecon.physics_audit import peak_rss_bytes
    apply_thread_limits(2)
    original_path=Path('results/p2-selection-endpoints-reference/protocol.json')
    manifest_path=Path('results/p2-full-selection-manifest/report.json')
    stage=Path('out/p2-selection-endpoints-reference/stages/5')
    checkpoint=Path('out/p2-selection-endpoints-reference/iterations/100-750.npz')
    original=json.loads(original_path.read_text())
    case=validate_case(json.loads(manifest_path.read_text()),1001,4.,'feature')
    selections={r['fraction']:r for r in case['selections'] if r['fraction'] in (5,100)}
    if file_hash(path)!=original['input_sha256'] or file_hash(manifest_path)!=original['manifest_sha256']:
        raise ValueError('original observations/manifest mismatch')
    for name,digest in original['identities']['dependencies'].items():
        if file_hash(Path(name))!=digest: raise ValueError('original numerical dependency changed: '+name)
    late={5:reference_stage(stage,original,selections[5],FROZEN_25),
          100:frozen_checkpoint(checkpoint,original,selections[100])['x']}
    files=[path,manifest_path,original_path,checkpoint,*[stage/name for name in ('identity.json','budget-750.json','budget-750.npz')]]
    hashes={str(p):file_hash(p) for p in files}
    names=['scene_study','scene_fft','scene_quadratic','scene_parallel_reference','scene_retained_cache',
           'scene_periodic_preconditioner','scene_window_preconditioner','scene_preconditioned_probe',
           'scene_conditioning','scene_iteration_state','scene_state_diagnostics','audit_scene_conditioning','audit_scene_selections']
    deps=[Path(__file__),Path('docs/scene-window-state-protocol.md')]+[Path('tools')/(name+'.py') for name in names]
    identity=identities(deps)
    if identity['package_source_hash']!=original['identities']['package_source_hash']:
        raise ValueError('original package source changed')
    protocol={'identities':identity,'input_hashes':hashes,'fractions':[5,100],
              'states':['zero','late'],'modes':['diagonal','periodic','window'],
              'probe_steps':32,'wall_budget_s':900,'cache_bytes':3*1024**3,'headroom_bytes':1024**3,
              'cpu_frame_workers':8,'threads':2,'product_relative_tolerance':1e-10,
              'initial_gradient_backward_tolerance':1e-10,'original_distance_tolerance':1e-5,
              'scene_updated':False,'scope':'Window/state conditioning diagnostic only; no fitted image or scientific qualification.'}
    directory=open_study(directory,protocol);started=time.monotonic();deadline=started+900
    rows=[];initials=[];spectra=[];failures=[];batch=None
    def check_time():
        if time.monotonic()>=deadline: raise TimeoutError('declared diagnostic budget exhausted')
    try:
        data=ObservedSceneData(path,range(500),'feature')
        for fraction,selection in selections.items():
            check_time();torch.cuda.empty_cache()
            if torch.cuda.mem_get_info()[0]<protocol['cache_bytes']+protocol['headroom_bytes']:
                raise MemoryError('insufficient declared GPU headroom')
            indices=selection['indices'];ops=data.operators(indices,margin=64)
            images=[data.images[i] for i in indices];variances=[data.variances[i] for i in indices]
            batch=RetainedCellFFTBatch(ops,device='cuda',cache_bytes=protocol['cache_bytes'])
            problem=SceneQuadratic(ops,images,variances,ridge=selection['mean_ridge'],batch=batch)
            reference_batch=ParallelSceneBatch(ops,workers=8)
            reference=SceneQuadratic(ops,images,variances,ridge=selection['mean_ridge'],batch=reference_batch)
            check_time();periodic=PeriodicScenePreconditioner(problem,workers=8)
            check_time();window=WindowAveragedPreconditioner(problem,workers=8)
            spectral={'fraction':fraction,'periodic':periodic.info(),'window':window.info(),'frequencies':[]}
            ny,nx=problem.shape
            for ky,kx in [(0,0),(1,0),(0,1),(ny//8,nx//8)]:
                check_time();energy=fourier_energy(reference,ky,kx)
                spectral['frequencies'].append({'ky':ky,'kx':kx,'independent_energy':energy,
                    'periodic_symbol':float(periodic.symbol[ky,kx]),'window_symbol':float(window.symbol[ky,kx])})
            spectra.append(spectral);write_json(directory/f'spectrum-{fraction}.json',spectral)
            for label,x in [('zero',np.zeros(problem.shape)),('late',late[fraction])]:
                check_time();hx=reference.normal(x);g=hx-reference.linear;gpu_g=problem.gradient(x)
                residual=cone_residual(x,g);free=(x>0)|(g<0)
                raw=reference.certificate(x,gradient=g)
                discrepancy=float(np.linalg.norm(gpu_g-g))
                initial={'fraction':fraction,'state':label,'raw_certificate':raw,'free_fraction':float(np.mean(free)),
                         'cpu_cuda_free_mask_disagreement':int(np.count_nonzero(free!=((x>0)|(gpu_g<0)))),
                         'gradient_relative_error':discrepancy/max(float(np.linalg.norm(g)),1e-30),
                         'gradient_backward_error':discrepancy/max(float(np.linalg.norm(hx)+np.linalg.norm(reference.linear)),1e-30)}
                initials.append(initial);write_json(directory/f'initial-{fraction}-{label}.json',initial)
                if initial['gradient_backward_error']>1e-10: raise ValueError('initial gradient backward parity failed')
                if fraction==5 and label=='late' and raw['relative_solution_error_bound']>1e-5:
                    raise ValueError('late25 reference no longer satisfies original criterion')
                for mode,inverse in [('diagonal',lambda r:r/problem.majorizer),('periodic',periodic),('window',window)]:
                    check_time();trace=[];key=f'{fraction}-{label}-{mode}'
                    def callback(row,y):
                        trace.append(row);write_json(directory/('trace-'+key+'.json'),{'trace':trace})
                    y,info=probe(problem.normal,residual,inverse,free=free,steps=32,deadline=deadline,callback=callback)
                    stream=BytesIO();np.savez_compressed(stream,correction=y)
                    atomic_write(directory/('correction-'+key+'.npz'),stream.getvalue())
                    hy=reference.normal(y);gpu_hy=problem.normal(y);error=float(np.linalg.norm(hy-gpu_hy)/max(np.linalg.norm(hy),1e-30))
                    e=residual-hy;den=max(np.linalg.norm(residual),1e-30)
                    row={'fraction':fraction,'state':label,'mode':mode,'probe':info,'product_relative_error':error,
                         'independent_reduced_relative_residual':float(np.linalg.norm(e[free])/den),
                         'independent_full_relative_residual':float(np.linalg.norm(e)/den),
                         'probe_complete':info['reason']!='wall_budget','cache':batch.cache_info(),
                         'process_peak_rss_bytes':peak_rss_bytes()}
                    rows.append(row);write_json(directory/(key+'.json'),row)
                    print(key,'reduced_residual=',row['independent_reduced_relative_residual'],flush=True)
            batch.clear_cache();batch=None
            del problem,reference,reference_batch,ops,periodic,window
    except Exception as exc:
        failures.append(repr(exc));print(repr(exc),flush=True)
    finally:
        if batch is not None: batch.clear_cache()
    unchanged=identity==identities(deps) and all(file_hash(Path(p))==digest for p,digest in hashes.items())
    passed=unchanged and not failures and len(rows)==12 and all(r['probe_complete'] and r['product_relative_error']<=1e-10 for r in rows)
    report={'status':'valid' if passed else 'incomplete','source_input_unchanged':unchanged,'rows':rows,
            'initials':initials,'spectra':spectra,'failures':failures,'wall_s':time.monotonic()-started,
            'scene_updated':False,'q3_authorized':False,'scope':protocol['scope']}
    write_json(directory/'report.json',report);return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    sys.exit(0 if run(a.input,a.out)['status']=='valid' else 1)
