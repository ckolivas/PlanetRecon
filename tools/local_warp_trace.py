"""Trace production circular AP measurements and the four composed warp stages.

This diagnostic mirrors the resident matcher and checks its final field against
production. Raw frame samples, the reference and scalar weights stay fixed.
"""
import argparse
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np
from scipy.ndimage import gaussian_filter
import torch
import torch.nn.functional as F

from planetrecon.backends.torch_circular import TorchCircularRegistration, sample, peaks
from planetrecon.backends.torch_accel import TorchBackend
from planetrecon.pipeline.circular_align import CircularMultiscaleRegistration
from planetrecon.pipeline.local_align import LocalRegistration
from planetrecon.io.ser import SERSource
from planetrecon.result import load_snapshot, save_snapshot
from planetrecon.export import export_result, ExportConfig
from planetrecon.runtime import apply_thread_limits
from tools.alignment_noise_experiment import read_png


def smooth_motion(field, sigma):
    """Gaussian regularization of displacements only, never detector pixels.

    The tested residual fields are zero at the detector borders, so periodic
    FFT filtering has no hard displacement step at the opposite edge.
    """
    h,w=field.shape[-2:]
    fy=torch.fft.fftfreq(h,device=field.device,dtype=field.dtype)[:,None]
    fx=torch.fft.fftfreq(w,device=field.device,dtype=field.dtype)[None,:]
    kernel=torch.exp(-2*torch.pi**2*sigma**2*(fx.square()+fy.square()))
    return torch.fft.ifft2(torch.fft.fft2(field)*kernel).real


def trace(engine, proxy, global_shift):
    """Return each accepted stage field plus pre-blending AP measurements."""
    image0 = torch.as_tensor(proxy, device=engine.device, dtype=torch.float64)
    origin = torch.as_tensor(global_shift, device=engine.device, dtype=torch.float64)[:, None, None]
    field = origin.expand(2, *engine.shape).clone()
    stages, records = [], []
    for layer in reversed(engine.layers):
        image = sample(image0, engine.yy+field[1], engine.xx+field[0])
        observed = ((engine.xx+field[0] >= 0)&(engine.xx+field[0] <= engine.shape[1]-1)
                    &(engine.yy+field[1] >= 0)&(engine.yy+field[1] <= engine.shape[0]-1))
        integral = F.pad((~observed).long().cumsum(0).cumsum(1), (1,0,1,0))
        y, x = layer['y'], layer['x']
        margin = layer['h']+layer['m']
        supported = (integral[y+margin+1,x+margin+1]-integral[y-margin,x+margin+1]
                     -integral[y+margin+1,x-margin]+integral[y-margin,x-margin]) == 0
        numerator, denominator = torch.zeros_like(field), torch.zeros_like(image)
        details = []
        for start in range(0, len(y), layer['chunk']):
            section = slice(start, start+layer['chunk'])
            cy, cx = y[section,None,None], x[section,None,None]
            costs = engine._scores(image,layer,section,layer['templates'][section],layer['strength'][section])
            delta, valid, peak = peaks(costs)
            valid &= supported[section]
            patch = sample(image,cy+layer['py']+delta[:,1,None,None],cx+layer['px']+delta[:,0,None,None])
            patch -= (patch*layer['weight']).sum((-1,-2))[:,None,None]
            strength = (patch.square()*layer['weight']).sum((-1,-2)).sqrt()
            reverse, reverse_valid, _ = peaks(engine._scores(engine.reference,layer,section,patch,strength))
            valid &= reverse_valid & (torch.linalg.vector_norm(reverse,dim=1) <= .75)
            confidence = ((peak-.8)/.15).clamp(0.,1.)*valid
            weight = layer['footprint']*confidence[:,None,None]
            indices = ((cy+layer['py'])*engine.shape[1]+cx+layer['px']).flatten()
            denominator.flatten().scatter_add_(0,indices,weight.flatten())
            for axis in range(2):
                numerator[axis].flatten().scatter_add_(0,indices,(weight*delta[:,axis,None,None]).flatten())
            details.append(torch.column_stack((delta,valid,peak,confidence,torch.linalg.vector_norm(reverse,dim=1))))
        residual = numerator/denominator.clamp_min(1e-300)
        residual *= denominator.clamp_max(1.)
        proposed = residual+sample(field,engine.yy+residual[1],engine.xx+residual[0],nearest=True)
        ux, uy = proposed-origin
        ux_y, ux_x = torch.gradient(ux)
        uy_y, uy_x = torch.gradient(uy)
        jacobian = (1+ux_x)*(1+uy_y)-ux_y*uy_x
        accepted = (torch.isfinite(proposed).all() & (jacobian.min() >= .25)
                    & (torch.hypot(ux,uy).max() <= 6.))
        field = torch.where(accepted,proposed,field)
        stages.append(field)
        records.append({'measurements': torch.cat(details).cpu().numpy(),
                        'accepted': bool(accepted), 'minimum_jacobian': float(jacobian.min()),
                        'coverage': denominator})
    return stages, records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=Path('out/saturn-local-warp-trace'))
    p.add_argument('--capture',type=Path,default=Path('2024-09-27-1154_3-CK-R-Sat.ser'))
    p.add_argument('--limit',type=int)
    p.add_argument('--centre',type=Path,help='Second pass: first-pass directory containing the mean field')
    p.add_argument('--smooth-field',type=float,help='Additional second-pass displacement smoothing, in pixels')
    args=p.parse_args()
    if args.limit is not None and args.limit < 1:
        p.error('--limit must be positive')
    if args.smooth_field is not None and (args.centre is None or not np.isfinite(args.smooth_field) or args.smooth_field <= 0):
        p.error('--smooth-field requires --centre and a positive finite sigma')
    args.out.mkdir(parents=True,exist_ok=False)
    apply_thread_limits(8); torch.set_num_threads(8)
    root=Path('out/saturn-controlled-alignment')
    original=load_snapshot(root/'local.npz'); reference=np.load(root/'reference.npy')
    with np.load(root/'diagnostics.npz') as d:
        indices,quality,shifts=(d[k].copy() for k in ('indices','quality','global_shifts'))
    if args.limit:indices,quality,shifts=indices[:args.limit],quality[:args.limit],shifts[:args.limit]
    _,text=read_png(root/'local.png'); mapping=json.loads(text)['mapping']
    matcher=CircularMultiscaleRegistration(reference,sampling_multiplier=5.,wavelength_nm=550.)
    engine=TorchCircularRegistration(matcher); backend=TorchBackend()
    sizes=[2*l['h']+1 for l in reversed(engine.layers)]
    names=[f'through_{s}' for s in sizes]
    mean=None
    if args.centre:
        mean=torch.as_tensor(np.load(args.centre/'mean_residual.npy'),device='cuda')
        with np.load(args.centre/'traces.npz') as d:
            np.testing.assert_array_equal(d['indices'],indices)
            np.testing.assert_array_equal(d['quality'],quality)
        assert tuple(mean.shape)==(2,*reference.shape)
        names=['original_local','mean_only','zero_mean_local']
        if args.smooth_field is not None:names.append('smooth_motion')
        sizes=[sizes[-1]]  # Second pass traces only the production final field.
    sums=np.zeros((len(names),*reference.shape)); weights=np.zeros_like(sums)
    field_sum=torch.zeros((2,*reference.shape),device='cuda',dtype=torch.float64)
    field_square=torch.zeros_like(field_sum); scalar=0.
    points=np.array([[165,194],[175,193],[185,193],[510,189],[520,190],[530,190],[348,150]])
    point_fields=[]; ap_records=[[] for _ in sizes]; guards=[]; errors=[]
    started=time.perf_counter()
    with SERSource(args.capture) as source:
        for n,index in enumerate(indices):
            raw=source.read_raw(int(index)); shift=shifts[n]
            proxy=LocalRegistration.proxy(raw)
            if mean is None:
                stages,records=trace(engine,proxy,shift)
            else:
                stages=[engine.displacement(proxy,shift,lambda:None)]
                records=[]
            if mean is None and n in (0,len(indices)//2,len(indices)-1):
                expected=engine.displacement(proxy,shift,lambda:None)
                error=float((expected-stages[-1]).abs().max()); errors.append(error)
                assert error < 1e-9
            origin=torch.as_tensor(shift,device='cuda')[:,None,None]
            residual=stages[-1]-origin
            score=max(float(quality[n]),1e-12)
            field_sum+=score*residual; field_square+=score*residual.square();scalar+=score
            point_fields.append(torch.stack([s[:,points[:,1],points[:,0]]-origin[:,0,:] for s in stages]).cpu().numpy())
            guards.append([r['accepted'] for r in records])
            if mean is None:
                fields=stages
                for j,r in enumerate(records):ap_records[j].append(r['measurements'])
            else:
                fields=[stages[-1],origin+mean,stages[-1]-mean]
                if args.smooth_field is not None:
                    fields.append(origin+smooth_motion(residual,args.smooth_field))
            for j,field in enumerate(fields):
                signal,weight,*_=backend.backproject(raw,field,'mono')
                sums[j]+=score*signal; weights[j]+=score*weight
            if (n+1)%512 == 0:
                print(f'{n+1}/{len(indices)} frames, {time.perf_counter()-started:.1f}s',flush=True)
    mean_field=field_sum/scalar
    np.save(args.out/'mean_residual.npy',mean_field.cpu().numpy())
    np.save(args.out/'std_residual.npy',(field_square/scalar-mean_field.square()).clamp_min(0).sqrt().cpu().numpy())
    np.savez_compressed(args.out/'traces.npz',indices=indices,quality=quality,points_xy=points,
                        point_fields=point_fields,stage_accepted=guards)
    if mean is None:
        for size,layer,records in zip(sizes,reversed(engine.layers),ap_records):
            np.savez_compressed(args.out/f'ap_{size}.npz',measurements=records,
                centres_xy=torch.column_stack((layer['x'],layer['y'])).cpu().numpy(),
                columns=['dx','dy','valid','peak','confidence','reverse_distance'])
    report={'n_used':len(indices),'pilot':bool(args.limit),'names':names,'sizes':sizes,
            'mean_source':str(args.centre) if args.centre else None,
            'motion_smoothing_sigma_px':args.smooth_field,
            'production_field_max_error_px':max(errors) if errors else None,
            'stage_accepted_fraction':np.mean(guards,axis=0).tolist() if mean is None else None,
            'seconds':time.perf_counter()-started,'variants':{}}
    for j,name in enumerate(names):
        image=np.divide(sums[j],weights[j],out=np.zeros_like(weights[j]),where=weights[j]>0)
        if (j==len(names)-1 and mean is None or j==0 and mean is not None) and not args.limit:
            report['production_image_max_error_adu']=float(np.max(np.abs(image-original.image)))
            assert report['production_image_max_error_adu'] < 1e-8
        provenance=deepcopy(original.provenance)
        provenance['local_warp_trace']={'variant':name,'diagnostic_only':True,'mean_source':report['mean_source'],
            'motion_smoothing_sigma_px':args.smooth_field if name=='smooth_motion' else None}
        result=replace(original,image=image,coverage=weights[j],validity=weights[j]>0,n_used=len(indices),provenance=provenance)
        save_snapshot(args.out/f'{name}.npz',result)
        result=replace(result,image=gaussian_filter(image,1.,radius=3))
        result.provenance['comparison_smoothing']={'sigma_px':1.,'radius_px':3}
        export_result(result,args.out/f'{name}_as_transfer.png',ExportConfig('png16',mapping['black'],mapping['white'],1.))
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
