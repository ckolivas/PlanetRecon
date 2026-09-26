"""Controlled multiscale patch stacks; no brightness normalization or denoising.

Reuses the previous 65px measurements and independently measures 93/131/185px
APs. Run with PYTHONPATH=. and the project CUDA environment. This does not
claim to reproduce AutoStakkert's private AP layout or recombination algorithm.
"""
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
from scipy.ndimage import gaussian_filter, laplace
import torch
import triton
import triton.language as tl

from tools.alignment_noise_experiment import measurements, read_png
from tools.patch_stacking_experiment import top_masks, finalize_patches, register_patches
from planetrecon.pipeline.local_align import LocalRegistration
from planetrecon.pipeline.preprocess_cache import load_cache
from planetrecon.pipeline.preprocess import best_frame_mask
from planetrecon.backends.torch_circular import TorchCircularRegistration, sample as reference_sample, peaks
from planetrecon.backends.cpu import CPUBackend
from planetrecon.io.ser import SERSource
from planetrecon.result import load_snapshot, save_snapshot
from planetrecon.export import ExportConfig, export_result
from planetrecon.reconstruction import ReconstructionConfig
from planetrecon.runtime import apply_thread_limits

SIZES = (65, 93, 131, 185)
NAMES = ('global', 'local2', 'local4')


@triton.jit
def _moments(image, ys, xs, templates, weights, output,
             WIDTH: tl.constexpr, WINDOW: tl.constexpr, TILES: tl.constexpr,
             BLOCK: tl.constexpr):
    point, offset, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    k = tile*BLOCK+tl.arange(0, BLOCK)
    valid = k < WINDOW*WINDOW
    y = tl.load(ys+point)+k//WINDOW-WINDOW//2+offset//7-3
    x = tl.load(xs+point)+k%WINDOW-WINDOW//2+offset%7-3
    value = tl.load(image+y*WIDTH+x, valid, other=0.)
    weight = tl.load(weights+k, valid, other=0.)
    template = tl.load(templates+point*WINDOW*WINDOW+k, valid, other=0.)
    address = output+((point*49+offset)*TILES+tile)*3
    tl.store(address, tl.sum(value*weight, 0))
    tl.store(address+1, tl.sum(value*value*weight, 0))
    tl.store(address+2, tl.sum(value*template*weight, 0))


def tiled_scores(image, ys, xs, templates, strengths, weight):
    """Same float64 weighted NCC, with bounded tiles for large AP windows."""
    p, w = len(ys), weight.shape[0]
    blocks = triton.cdiv(w*w, 1024)
    partial = torch.empty((p,49,blocks,3),dtype=torch.float64,device=image.device)
    _moments[(p,49,blocks)](image,ys,xs,templates,weight,partial,
        WIDTH=image.shape[1],WINDOW=w,TILES=blocks,BLOCK=1024,
        num_warps=4,enable_fp_fusion=False)
    moments = partial.sum(dim=2)
    mean = moments[...,0]
    variance = (moments[...,1]+mean.square()*(weight.sum()-2.)).clamp_min(0.)
    numerator = moments[...,2]-mean*(templates*weight).sum((-1,-2))[:,None]
    costs = numerator/(variance.sqrt()*strengths[:,None]).clamp_min(1e-15)
    return costs.reshape(p,7,7)


@triton.jit
def _sample_pixels(image, ys, xs, output, HEIGHT: tl.constexpr, WIDTH: tl.constexpr,
                   COORDS: tl.constexpr, ELEMENTS: tl.constexpr, BLOCK: tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    valid=i<ELEMENTS
    q=i%COORDS
    y=tl.load(ys+q,valid,other=0.)
    x=tl.load(xs+q,valid,other=0.)
    iy,ix=tl.floor(y).to(tl.int64),tl.floor(x).to(tl.int64)
    fy,fx=y-iy,x-ix
    value=tl.full((BLOCK,),0.,tl.float64)
    for dy in tl.static_range(2):
        for dx in tl.static_range(2):
            sy,sx=iy+dy,ix+dx
            inside=(sy>=0)&(sy<HEIGHT)&(sx>=0)&(sx<WIDTH)
            pixel=tl.load(image+(i//COORDS)*HEIGHT*WIDTH+sy*WIDTH+sx,valid&inside,other=0.)
            wy=fy if dy else 1.-fy
            wx=fx if dx else 1.-fx
            value+=pixel*(wy*wx)
    tl.store(output+i,value,valid)


def sample(image,y,x):
    """One-kernel equivalent of the existing float64 bilinear pull."""
    y,x=torch.broadcast_tensors(y,x)
    y,x=y.contiguous(),x.contiguous()
    image=image.contiguous()
    result=torch.empty((*image.shape[:-2],*y.shape),dtype=image.dtype,device=image.device)
    _sample_pixels[(triton.cdiv(result.numel(),256),)](image,y,x,result,
        HEIGHT=image.shape[-2],WIDTH=image.shape[-1],COORDS=y.numel(),ELEMENTS=result.numel(),
        BLOCK=256,num_warps=4,enable_fp_fusion=False)
    return result


def geometry(reference, size):
    layer = LocalRegistration(reference,window=size,step=size//2,circular=True)
    active = np.flatnonzero(np.asarray(layer.texture_valid)&
                            (layer.strength>=max(layer.strength.max()*.08,1e-12)))
    centres = np.column_stack((layer.ys[active//len(layer.xs)],layer.xs[active%len(layer.xs)]))
    y,x = np.indices((size,size))-size//2
    footprint = .5*(1+np.cos(np.pi*np.clip(2*np.hypot(x,y)/(size//2)-1,0,1)))
    return layer,centres,footprint


def patch_totals(patches, supports, centres, footprint, shape):
    """Unnormalised geometric contributions, suitable for combining scales."""
    total,weight = np.zeros(shape),np.zeros(shape)
    h=footprint.shape[0]//2
    for patch,support,(y,x) in zip(patches,supports,centres):
        sl=np.s_[y-h:y+h+1,x-h:x+h+1]
        total[sl]+=patch*support*footprint
        weight[sl]+=support*footprint
    return total,weight


def blend_totals(total, weight, fallback):
    return (total+np.maximum(1.-weight,0.)*fallback)/np.maximum(weight,1.)


def spread_subset(centres, count):
    """Deterministic spatially spread subset, used only as an AP-count control."""
    centres=np.asarray(centres,dtype=float)
    if count>len(centres):raise ValueError('Subset exceeds available APs')
    chosen=[int(np.argmin(np.sum((centres-centres.mean(axis=0))**2,axis=1)))]
    distance=np.full(len(centres),np.inf)
    while len(chosen)<count:
        distance=np.minimum(distance,np.sum((centres-centres[chosen[-1]])**2,axis=1))
        distance[chosen]=-np.inf
        chosen.append(int(np.argmax(distance)))
    return np.sort(chosen)


def supported_windows(centres, shape, shift, margin):
    c=np.asarray(centres)
    return ((c[:,0]-margin+shift[1]>=0)&(c[:,0]+margin+shift[1]<=shape[0]-1)&
            (c[:,1]-margin+shift[0]>=0)&(c[:,1]+margin+shift[0]<=shape[1]-1))


def prepare(raw):
    return raw,LocalRegistration.proxy(raw),np.stack([
        laplace(gaussian_filter(raw,sigma))**2 for sigma in (2.,4.)])


def prepared_frames(source, indices, workers=8):
    # SERSource has a seek/read cursor: only the calling thread reads it.
    # Workers operate on independent arrays, with a bounded two-frame queue
    # per worker rather than buffering the capture.
    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending=deque()
        iterator=iter(indices)
        def submit():
            index=next(iterator,None)
            if index is None:return
            raw=source.read_raw(int(index)).astype(float)
            pending.append((int(index),executor.submit(prepare,raw)))
        for _ in range(workers*2):submit()
        while pending:
            index,future=pending.popleft()
            prepared=future.result()
            submit()
            yield index,prepared


def measure(args, source, base, reference):
    indices=base['indices'][:args.limit] if args.limit else base['indices']
    engine=TorchCircularRegistration(SimpleNamespace(shape=reference.shape,
        layers=[geometry(reference,w)[0] for w in SIZES[1:]]))
    arrays=[]
    for w,gpu in zip(SIZES[1:],engine.layers):
        centres=geometry(reference,w)[1]
        arrays.append(dict(centres=centres,local_shifts=np.zeros((len(indices),len(centres),2)),
            quality=np.full((2,len(indices),len(centres)),-np.inf),
            reliable=np.zeros((len(indices),len(centres)),bool)))
    started=time.perf_counter()
    for i,(index,(raw,proxy,maps)) in enumerate(prepared_frames(source,indices,args.workers)):
        assert index==indices[i]
        shift=base['global_shifts'][i]
        aligned=sample(torch.as_tensor(proxy,device='cuda'),engine.yy+shift[1],engine.xx+shift[0])
        maps=torch.as_tensor(maps,device='cuda')
        for w,gpu,saved in zip(SIZES[1:],engine.layers,arrays):
            cy,cx=gpu['y'][:,None,None],gpu['x'][:,None,None]
            costs=tiled_scores(aligned,gpu['y'],gpu['x'],gpu['templates'],gpu['strength'],gpu['weight'])
            delta,valid,_=peaks(costs)
            patch=sample(aligned,cy+gpu['py']+delta[:,1,None,None],cx+gpu['px']+delta[:,0,None,None])
            patch-=(patch*gpu['weight']).sum((-1,-2))[:,None,None]
            strength=(patch.square()*gpu['weight']).sum((-1,-2)).sqrt()
            reverse,reverse_valid,_=peaks(tiled_scores(engine.reference,gpu['y'],gpu['x'],patch,strength,gpu['weight']))
            valid &= reverse_valid & (torch.linalg.vector_norm(reverse,dim=1)<=.75)
            c=saved['centres']; margin=w//2+3
            supported=supported_windows(c,raw.shape,shift,margin)
            valid &= torch.as_tensor(supported,device='cuda')
            saved['local_shifts'][i]=shift+torch.where(valid[:,None],delta,0).cpu().numpy()
            saved['reliable'][i]=valid.cpu().numpy()
            # Coordinate arithmetic follows the preceding quality experiment.
            patches=sample(maps,cy+gpu['py']+float(shift[1]),cx+gpu['px']+float(shift[0]))
            saved['quality'][:,i]=(patches*gpu['weight']).sum((-1,-2)).cpu().numpy()
            h=w//2
            quality_supported=supported_windows(c,raw.shape,shift,h)
            saved['quality'][:,i,~quality_supported]=-np.inf
        if (i+1)%512==0 or i+1==len(indices):
            print(json.dumps(dict(stage='measure',frames=i+1,total=len(indices),seconds=round(time.perf_counter()-started,1))),flush=True)
    for w,saved in zip(SIZES[1:],arrays):
        np.savez_compressed(args.out/f'measurements_{w}.npz',indices=indices,**saved)


def stack(args, source, base, reference, config):
    count=int(base['original_selected'].sum())
    resident=[]
    kwargs=dict(device='cuda',dtype=torch.float64)
    for w in SIZES[1:]:
        with np.load(args.out/f'measurements_{w}.npz') as data:
            saved={k:data[k].copy() for k in data.files}
        np.testing.assert_array_equal(saved['indices'],base['indices'])
        c=saved['centres']; p=len(c); h=w//2
        masks=np.stack([np.broadcast_to(base['original_selected'][:,None],(len(base['indices']),p)),
                        top_masks(saved['quality'][0],count),top_masks(saved['quality'][1],count)])
        np.testing.assert_array_equal(masks.sum(axis=1),np.full((3,p),count))
        py,px=torch.meshgrid(torch.arange(-h,h+1,device='cuda'),torch.arange(-h,h+1,device='cuda'),indexing='ij')
        resident.append(dict(size=w,saved=saved,masks=masks,py=py,px=px,
            cy=torch.as_tensor(c[:,0],device='cuda')[:,None,None],cx=torch.as_tensor(c[:,1],device='cuda')[:,None,None],
            signal=torch.zeros((3,p,w,w),**kwargs),weight=torch.zeros((3,p,w,w),**kwargs)))
        np.savez_compressed(args.out/f'selections_{w}.npz',masks=masks,indices=base['indices'])
    ones=torch.ones(reference.shape,**kwargs)
    started=time.perf_counter()
    for i,index in enumerate(base['indices']):
        if not any(item['masks'][:,i].any() for item in resident):continue
        raw=torch.as_tensor(source.read_raw(int(index)).astype(float),**kwargs)
        score=float(base['global_weights'][i]) if config.quality_weighting else 1.
        for item in resident:
            if not item['masks'][:,i].any():continue
            shift=torch.as_tensor(item['saved']['local_shifts'][i],**kwargs)
            y=item['cy']+item['py']+shift[:,1,None,None]
            x=item['cx']+item['px']+shift[:,0,None,None]
            signal,support=sample(raw,y,x),sample(ones,y,x)
            factors=torch.as_tensor(item['masks'][:,i],**kwargs)[:,:,None,None]*score
            item['signal']+=factors*signal[None]
            item['weight']+=factors*support[None]
        if (i+1)%1024==0:
            print(json.dumps(dict(stage='stack',frames=i+1,total=len(base['indices']),seconds=round(time.perf_counter()-started,1))),flush=True)
    for item in resident:
        total,weight=item['signal'].cpu().numpy(),item['weight'].cpu().numpy()
        np.savez_compressed(args.out/f'patch_stacks_{item["size"]}.npz',
            means=finalize_patches(total,weight),weights=weight,centres=item['saved']['centres'],
            footprint=geometry(reference,item['size'])[2])


def combine(args, reference, original, base, mapping):
    products={}; counts={}; cpu=CPUBackend(threads=8)
    totals={(name,registered):[np.zeros_like(reference),np.zeros_like(reference)]
            for name in NAMES for registered in (False,True)}
    report=dict(sizes=list(SIZES),ap_counts={},frame_count_per_ap=int(base['original_selected'].sum()),
        screened_pool=len(base['indices']),reference_sha256=hashlib.sha256(reference.tobytes()).hexdigest(),
        source_measurements=str(args.single),quality_sigmas=[2,4],normalization='none',variants={},scales={})
    sparse_totals={k:[np.zeros_like(reference),np.zeros_like(reference)] for k in totals}
    sparse_counts=dict(zip(SIZES,[22,15,10,8]))
    report['sparse_ap_counts']=sparse_counts
    report['sparse_layout']='spatially spread 55-AP subset; not the exact AS coordinates'
    masks_by_name={name:base['original_selected'].copy() for name in NAMES}
    sparse_masks={name:base['original_selected'].copy() for name in NAMES}
    for w in SIZES:
        path=args.single/'patch_stacks.npz' if w==65 else args.out/f'patch_stacks_{w}.npz'
        with np.load(path) as data:
            means,weights,c,foot=(data[k].copy() for k in ('means','weights','centres','footprint'))
        if w==65:
            with np.load(args.single/'selections.npz') as data:masks=data['masks'][:3].copy()
            reliability=base['reliable']
        else:
            with np.load(args.out/f'selections_{w}.npz') as data:masks=data['masks'].copy()
            with np.load(args.out/f'measurements_{w}.npz') as data:reliability=data['reliable'].copy()
        subset=spread_subset(c,sparse_counts[w])
        report['ap_counts'][str(w)]=len(c)
        report['scales'][str(w)]={'reliable_match_fraction':float(reliability.mean()),'selection':{}}
        for j,name in enumerate(NAMES):
            valid=(weights[j]>0).astype(float)
            masks_by_name[name]|=masks[j].any(axis=1)
            sparse_masks[name]|=masks[j,:,subset].any(axis=0)
            adjusted,support,offsets=register_patches(means[j],valid,c,reference,cpu)
            report['scales'][str(w)]['selection'][name]=dict(
                overlap_with_original_percent=np.percentile((masks[j]&base['original_selected'][:,None]).sum(axis=0)/masks[j].sum(axis=0)*100,[0,50,100]).tolist(),
                post_stack_shifts_xy=offsets)
            for registered,patches,supports in [(False,means[j],valid),(True,adjusted,support)]:
                suffix='_registered' if registered else ''
                total,weight=patch_totals(patches,supports,c,foot,reference.shape)
                products[f'scale{w}_{name}{suffix}']=blend_totals(total,weight,original.image)
                counts[f'scale{w}_{name}{suffix}']=int((masks[j].any(axis=1)|base['original_selected']).sum())
                totals[name,registered][0]+=total
                totals[name,registered][1]+=weight
                sparse_total,sparse_weight=patch_totals(patches[subset],supports[subset],c[subset],foot,reference.shape)
                sparse_totals[name,registered][0]+=sparse_total
                sparse_totals[name,registered][1]+=sparse_weight
                if w==65:
                    prior=load_snapshot(args.single/f'patch_{name}{suffix}.npz')
                    error=float(abs(products[f'scale65_{name}{suffix}']-prior.image).max())
                    report.setdefault('scale65_control_max_error_adu',{})[name+suffix]=error
                    np.testing.assert_allclose(products[f'scale65_{name}{suffix}'],prior.image,rtol=0,atol=1e-12)
    for (name,registered),(total,weight) in totals.items():
        suffix='_registered' if registered else ''
        products[f'multiscale_{name}{suffix}']=blend_totals(total,weight,original.image)
        counts[f'multiscale_{name}{suffix}']=int(masks_by_name[name].sum())
    for (name,registered),(total,weight) in sparse_totals.items():
        suffix='_registered' if registered else ''
        products[f'multiscale55_{name}{suffix}']=blend_totals(total,weight,original.image)
        counts[f'multiscale55_{name}{suffix}']=int(sparse_masks[name].sum())
    for name,image in products.items():
        assert np.isfinite(image).all()
        used=counts[name]
        provenance=deepcopy(original.provenance)
        provenance['experiment']=dict(kind='independent_multiscale_patch_stacking',variant=name,
            sizes=list(SIZES),n_per_ap=report['frame_count_per_ap'],normalization='none',
            coverage_units='binary observed support; per-AP weights in patch_stacks_SIZE.npz',
            note='Experimental mechanisms; baseline config retained as context only')
        result=replace(original,image=image,n_used=used,
            n_rejected=original.n_used+original.n_rejected-used,
            coverage=original.validity.astype(float),layer_coverage={},provenance=provenance,backend='gpu')
        save_snapshot(args.out/f'{name}.npz',result)
        export_result(result,args.out/f'{name}.png',ExportConfig('png16',mapping['black'],mapping['white']),overwrite=True)
        report['variants'][name]=dict(raw=measurements(image))
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(stage='complete',outputs=list(products))),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture',type=Path,required=True)
    parser.add_argument('--single',type=Path,default=Path('out/saturn-patch-stacking'))
    parser.add_argument('--baseline',type=Path,default=Path('out/saturn-controlled-alignment'))
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--stage',choices=['all','measure','stack','combine'],default='all')
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--limit',type=int,help='Measurement-only pilot; never used for the final stack')
    args=parser.parse_args()
    if not 1<=args.workers<=16:raise ValueError('Use 1 to 16 preparation workers')
    if args.limit and args.stage!='measure':raise ValueError('Partial captures are measurement-only pilots')
    args.out.mkdir(parents=True,exist_ok=True)
    apply_thread_limits(8); torch.set_num_threads(1)
    if args.stage!='combine' and not torch.cuda.is_available():raise RuntimeError('CUDA required')
    # Reuse the exact previous fallback, including its summation order.
    original=load_snapshot(args.single/'global_control.npz')
    config=ReconstructionConfig.from_dict(original.provenance['config'])
    reference=np.load(args.baseline/'reference.npy')
    _,metadata=read_png(args.baseline/'local.png'); mapping=json.loads(metadata)['mapping']
    with np.load(args.single/'measurements.npz') as data:base={k:data[k].copy() for k in data.files}
    with SERSource(args.capture) as source:
        if source.color_mode()!='mono':raise ValueError('This experiment requires mono input')
        selection,status=load_cache(source,config)
        if selection is None:raise ValueError(status)
        np.testing.assert_array_equal(base['indices'],np.flatnonzero(selection.accepted))
        np.testing.assert_array_equal(base['global_weights'],selection.measurements[base['indices'],0])
        np.testing.assert_array_equal(base['original_selected'],best_frame_mask(selection,config.stack_percent,config.frame_selection_mode)[base['indices']])
        if args.stage in ('all','measure'):
            if any((args.out/f'measurements_{w}.npz').exists() for w in SIZES[1:]):raise FileExistsError('Measurements already exist')
            measure(args,source,base,reference)
        if args.stage in ('all','stack'):stack(args,source,base,reference,config)
        if args.stage in ('all','stack','combine'):combine(args,reference,original,base,mapping)


if __name__=='__main__':main()
