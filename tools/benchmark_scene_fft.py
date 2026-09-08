"""Measured complete solver and kernel parity for shared optical transforms."""
import argparse
from pathlib import Path
import sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from tools.scene_study import ObservedSceneData, TRAIN_SMALL, CellFFTBatch
from tools.scene_quadratic import SceneQuadratic, solve
from tools.study_io import identities, open_study, write_json, file_hash


def run(path,directory,*,cuda=False):
    from planetrecon.runtime import apply_thread_limits
    from planetrecon.physics_audit import peak_rss_bytes
    apply_thread_limits(2)
    dependencies=[Path(__file__),Path('tools/scene_fft.py'),Path('tools/scene_study.py'),Path('tools/scene_quadratic.py')]
    identity=identities(dependencies)
    modes=['reference','cpu-no-cache','cpu-cache']+(['cuda-cache'] if cuda else [])
    protocol={'question':'Do shared transforms preserve full-grid operators, objective and certified solves, and what complete-job cost do they achieve?',
              'identities':identity,'input_sha256':file_hash(path),'modes':modes,
              'indices':list(TRAIN_SMALL),'crop':'feature','threads':2,'repeats':3,
              'ridge_mean':.001,'solver_cap':300,'solver_tolerance':1e-5,
              'operator_relative_tolerance':1e-10,'image_relative_tolerance':1e-5,
              'cache_bytes':256*1024**2,'q3_authorized':False,
              'scope':'One three-frame full-optical-domain pilot, scalar data-derived variance; no full family or production GPU claim.'}
    directory=open_study(directory,protocol)
    started=time.monotonic()
    data=ObservedSceneData(path,TRAIN_SMALL)
    ops=data.operators(TRAIN_SMALL)
    images=[data.images[i] for i in TRAIN_SMALL];var=[data.variances[i] for i in TRAIN_SMALL]
    rng=np.random.default_rng(800)
    x=rng.normal(size=ops[0].scene_shape)
    residuals=[rng.normal(size=o.output_shape) for o in ops]
    weights=[np.full(o.output_shape,1/v/3) for o,v in zip(ops,var)]
    reference_forward=[o.forward(x) for o in ops]
    reference_adjoint=sum(o.adjoint(y) for o,y in zip(ops,residuals))
    reference_normal=sum(o.adjoint(w*o.forward(x)) for o,w in zip(ops,weights))
    rows=[];ref_image=None;ref_problem=None
    def rel(a,b):
        return float(np.linalg.norm(np.asarray(a)-np.asarray(b))/max(np.linalg.norm(b),1e-30))
    for mode in modes:
        t=time.monotonic()
        batch=None if mode=='reference' else CellFFTBatch(ops,device='cuda' if mode.startswith('cuda') else 'cpu',
            cache_bytes=0 if mode=='cpu-no-cache' else protocol['cache_bytes'])
        torch=None
        if mode.startswith('cuda'):
            import torch
            torch.cuda.reset_peak_memory_stats()
        forward=(lambda:[o.forward(x) for o in ops]) if batch is None else lambda:batch.forward(x)
        adjoint=(lambda:sum(o.adjoint(y) for o,y in zip(ops,residuals))) if batch is None else lambda:batch.adjoint(residuals)
        normal=(lambda:sum(o.adjoint(w*o.forward(x)) for o,w in zip(ops,weights))) if batch is None else lambda:batch.normal(x,weights)
        metrics={};errors={}
        for name,fn,ref in [('forward',forward,reference_forward),('adjoint',adjoint,reference_adjoint),('normal',normal,reference_normal)]:
            cold=time.monotonic();result=fn();cold=time.monotonic()-cold
            samples=[]
            for _ in range(protocol['repeats']):
                st=time.monotonic();fn();samples.append(time.monotonic()-st)
            metrics[name]={'first_s':cold,'warm_median_s':float(np.median(samples))}
            errors[name]=rel(result,ref)
        setup=time.monotonic()
        problem=SceneQuadratic(ops,images,var,ridge=.001,batch=batch)
        setup=time.monotonic()-setup
        solution,info=solve(problem,maxiter=300,tolerance=1e-5)
        if mode=='reference':ref_image,ref_problem=solution,problem
        checked=ref_problem.certificate(solution)
        errors['image']=rel(solution,ref_image)
        errors['objective']=abs(problem.objective(solution)-ref_problem.objective(solution))/max(abs(ref_problem.objective(solution)),1.)
        passed=(all(errors[k]<=protocol['operator_relative_tolerance'] for k in ('forward','adjoint','normal','objective'))
                and errors['image']<=protocol['image_relative_tolerance'] and info['converged']
                and checked['relative_solution_error_bound']<=1e-5)
        row={'mode':mode,'passed':passed,'kernels':metrics,'relative_errors':errors,'solve':info,
             'reference_certificate':checked,'setup_s':setup,'mode_wall_s':time.monotonic()-t,
             'process_peak_rss_bytes':peak_rss_bytes(),'cache':None if batch is None else batch.cache_info(),
             'torch_peak_allocated_bytes':None if torch is None else torch.cuda.max_memory_allocated(),
             'gpu_name':None if torch is None else torch.cuda.get_device_name()}
        rows.append(row);write_json(directory/f'{mode}.json',row)
        print(mode,'passed=',passed,'normal=',metrics['normal']['warm_median_s'],'solve=',info['wall_s'],flush=True)
        if batch is not None:batch.clear_cache()
        del batch,problem
    unchanged=identity==identities(dependencies) and protocol['input_sha256']==file_hash(path)
    report={'status':'valid' if unchanged and all(r['passed'] for r in rows) else 'incomplete',
            'source_input_unchanged':unchanged,'rows':rows,'wall_s':time.monotonic()-started,'q3_authorized':False,'scope':protocol['scope']}
    write_json(directory/'report.json',report)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--cuda',action='store_true')
    a=p.parse_args();r=run(a.input,a.out,cuda=a.cuda);sys.exit(0 if r['status']=='valid' else 1)
