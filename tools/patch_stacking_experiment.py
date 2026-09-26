"""Controlled per-AP selection and stack/recombine experiment for mono Saturn.

No brightness normalization or output denoising. Keeps the existing reference,
screened input pool, global quality weights and 5,738-frame target. This is an
experimental patch implementation, not an implementation of private AS internals.
"""
import argparse
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
from scipy.ndimage import gaussian_filter, laplace, shift as ndshift

from tools.alignment_noise_experiment import read_png, measurements
from planetrecon.io.ser import SERSource
from planetrecon.pipeline.local_align import LocalRegistration
from planetrecon.pipeline.preprocess import best_frame_mask
from planetrecon.pipeline.preprocess_cache import load_cache
from planetrecon.reconstruction import ReconstructionConfig
from planetrecon.result import load_snapshot, save_snapshot
from planetrecon.export import ExportConfig, export_result
from planetrecon.runtime import apply_thread_limits


def top_masks(quality, count):
    """Columns rank APs independently; ties retain capture order."""
    quality = np.asarray(quality)
    if quality.ndim != 2 or not 0 < count <= quality.shape[0]:
        raise ValueError('Require frame-by-AP qualities and a valid frame count')
    if np.any(np.isfinite(quality).sum(axis=0) < count):
        raise ValueError('Insufficient supported observations for an AP')
    order = np.argsort(-np.where(np.isfinite(quality), quality, -np.inf), axis=0, kind='stable')[:count]
    mask = np.zeros(quality.shape, dtype=bool)
    np.put_along_axis(mask, order, True, axis=0)
    return mask


def combine_patches(patches, supports, centres, footprint, fallback):
    h = footprint.shape[0]//2
    total = np.zeros_like(fallback)
    weight = np.zeros_like(fallback)
    for patch, support, (y, x) in zip(patches, supports, centres):
        section = np.s_[y-h:y+h+1, x-h:x+h+1]
        total[section] += patch*footprint*support
        weight[section] += footprint*support
    # Fade into the globally aligned fallback where AP support dwindles.
    # Renormalizing a lone vanishing footprint would otherwise create a hard seam.
    total += np.maximum(1.-weight, 0.)*fallback
    return total/np.maximum(weight, 1.), weight


def finalize_patches(total, weight):
    return np.divide(total, weight, out=np.zeros_like(total), where=weight>0)


def register_patches(patches, valid, centres, reference, backend):
    """Pull completed patch means toward the unchanged reference coordinates."""
    adjusted, adjusted_support, offsets = [], [], []
    h = patches.shape[-1]//2
    for patch, support0, (y, x) in zip(patches, valid, centres):
        refpatch = reference[y-h:y+h+1, x-h:x+h+1]
        dx, dy = backend.phase_correlation(refpatch, patch)
        if not np.isfinite([dx, dy]).all() or max(abs(dx), abs(dy)) > 3:
            dx = dy = 0.
        signal = ndshift(patch*support0, (-dy, -dx), order=1, mode='grid-constant', prefilter=False)
        support = ndshift(support0, (-dy, -dx), order=1, mode='grid-constant', prefilter=False)
        adjusted.append(np.divide(signal, support, out=np.zeros_like(signal), where=support > 0))
        adjusted_support.append(support)
        offsets.append([dx, dy])
    return np.array(adjusted), np.array(adjusted_support), offsets


def geometry(reference):
    layer = LocalRegistration(reference, window=65, step=32, circular=True)
    active = np.flatnonzero(np.asarray(layer.texture_valid) &
                            (layer.strength >= max(layer.strength.max()*.08, 1e-12)))
    centres = np.column_stack((layer.ys[active//len(layer.xs)], layer.xs[active % len(layer.xs)]))
    y, x = np.indices((65, 65))-32
    footprint = .5*(1+np.cos(np.pi*np.clip(2*np.hypot(x,y)/32-1,0,1)))
    return layer, centres, footprint


def measure(args, source, selection, config, reference, layer, centres):
    import torch
    from planetrecon.backends.torch_accel import TorchBackend
    from planetrecon.backends.torch_circular import TorchCircularRegistration, peaks, sample
    engine = TorchCircularRegistration(SimpleNamespace(shape=reference.shape, layers=[layer]))
    gpu_layer = engine.layers[0]
    assert len(gpu_layer['y']) == len(centres)
    backend = TorchBackend()
    prepared = backend.prepare_reference(reference)
    indices = np.flatnonzero(selection.accepted)
    n, p = len(indices), len(centres)
    global_shifts = np.zeros((n,2))
    local_shifts = np.zeros((n,p,2))
    quality = np.full((2,n,p), -np.inf)
    reliable = np.zeros((n,p), bool)
    cy, cx = gpu_layer['y'][:,None,None], gpu_layer['x'][:,None,None]
    py, px = gpu_layer['py'], gpu_layer['px']
    started = time.perf_counter()
    for i, index in enumerate(indices):
        raw = source.read_raw(int(index)).astype(float)
        shift = backend.phase_correlation(prepared, raw)
        global_shifts[i] = shift
        if not np.isfinite(shift).all() or max(abs(v) for v in shift) > config.max_shift_px:
            raise ValueError(f'Unsupported global shift in screened frame {index}')
        proxy = torch.as_tensor(LocalRegistration.proxy(raw), device='cuda:0')
        aligned = sample(proxy, engine.yy+shift[1], engine.xx+shift[0])
        deltas = []
        validity = []
        supported = ((centres[:,0]-35+shift[1] >= 0) & (centres[:,0]+35+shift[1] < raw.shape[0]) &
                     (centres[:,1]-35+shift[0] >= 0) & (centres[:,1]+35+shift[0] < raw.shape[1]))
        for start in range(0,p,gpu_layer['chunk']):
            section = slice(start,start+gpu_layer['chunk'])
            costs = engine._scores(aligned, gpu_layer, section,
                                  gpu_layer['templates'][section], gpu_layer['strength'][section])
            delta, valid, _ = peaks(costs)
            patch = sample(aligned, cy[section]+py+delta[:,1,None,None],
                           cx[section]+px+delta[:,0,None,None])
            patch -= (patch*gpu_layer['weight']).sum((-1,-2))[:,None,None]
            strength = (patch.square()*gpu_layer['weight']).sum((-1,-2)).sqrt()
            reverse, reverse_valid, _ = peaks(engine._scores(engine.reference,gpu_layer,section,patch,strength))
            valid &= reverse_valid & (torch.linalg.vector_norm(reverse,dim=1)<=.75)
            valid &= torch.as_tensor(supported[section],device='cuda:0')
            deltas.append(torch.where(valid[:,None],delta,0).cpu().numpy())
            validity.append(valid.cpu().numpy())
        local_shifts[i] = np.asarray(shift)+np.concatenate(deltas)
        reliable[i] = np.concatenate(validity)
        # Smooth only for quality measurement. No ranked pixels enter stacking.
        maps = np.stack([laplace(gaussian_filter(raw,sigma))**2 for sigma in (2.,4.)])
        patches = sample(torch.as_tensor(maps,device='cuda:0'),cy+py+shift[1],cx+px+shift[0])
        quality[:,i] = (patches*gpu_layer['weight']).sum((-1,-2)).cpu().numpy()
        quality[:,i,~supported] = -np.inf
        if (i+1)%1024 == 0:
            print(json.dumps(dict(stage='measure',used=i+1,total=n,seconds=round(time.perf_counter()-started,1))),flush=True)
    np.savez_compressed(args.out/'measurements.npz',indices=indices,global_shifts=global_shifts,
        local_shifts=local_shifts,quality=quality,reliable=reliable,
        centres=centres,global_weights=selection.measurements[indices,0],
        original_selected=best_frame_mask(selection,config.stack_percent,config.frame_selection_mode)[indices])


def stack(args, source, config, reference, centres, footprint, original, mapping):
    import torch
    from planetrecon.backends.torch_circular import sample
    from planetrecon.backends.cpu import CPUBackend
    with np.load(args.out/'measurements.npz') as data:
        saved = {k:data[k].copy() for k in data.files}
    indices, shifts, gshifts = (saved[k] for k in ('indices','local_shifts','global_shifts'))
    chosen = saved['original_selected']
    count = int(chosen.sum())
    n, p = len(indices), len(centres)
    names = ['patch_global','patch_local2','patch_local4','patch_global4']
    masks = np.stack([np.broadcast_to(chosen[:,None],(n,p)),top_masks(saved['quality'][0],count),
                      top_masks(saved['quality'][1],count),
                      np.broadcast_to(top_masks(saved['quality'][1].mean(axis=1)[:,None],count),(n,p))])
    names += ['patch_global_sequential', 'patch_local4_sequential']
    masks = np.concatenate([masks, masks[[0, 2]]], axis=0)
    v = len(names)
    np.testing.assert_array_equal(masks.sum(axis=1), np.full((v,p),count))
    np.savez_compressed(args.out/'selections.npz',names=names,indices=indices,masks=masks)
    kwargs = dict(dtype=torch.float64,device='cuda:0')
    sums, weights = [torch.zeros((v,p,65,65),**kwargs) for _ in range(2)]
    dense_sum,dense_weight,global_sum,global_weight = [torch.zeros(reference.shape,**kwargs) for _ in range(4)]
    cy = torch.as_tensor(centres[:,0],device='cuda:0')[:,None,None]
    cx = torch.as_tensor(centres[:,1],device='cuda:0')[:,None,None]
    py,px = torch.meshgrid(torch.arange(-32,33,device='cuda:0'),torch.arange(-32,33,device='cuda:0'),indexing='ij')
    yy,xx = torch.meshgrid(torch.arange(reference.shape[0],device='cuda:0'),torch.arange(reference.shape[1],device='cuda:0'),indexing='ij')
    ones = torch.ones(reference.shape,**kwargs)
    foot = torch.as_tensor(footprint,**kwargs)
    flat = ((cy+py)*reference.shape[1]+cx+px).flatten()
    denominator = torch.zeros(reference.shape,**kwargs)
    denominator.flatten().scatter_add_(0,flat,foot.expand(p,-1,-1).flatten())
    actual = np.zeros((v,p),int)
    started = time.perf_counter()
    for i,index in enumerate(indices):
        if not masks[:,i].any():
            continue
        raw = torch.as_tensor(source.read_raw(int(index)).astype(float),**kwargs)
        displacement = torch.as_tensor(shifts[i],**kwargs)
        patches = sample(raw,cy+py+displacement[:,1,None,None],cx+px+displacement[:,0,None,None])
        supports = sample(ones,cy+py+displacement[:,1,None,None],cx+px+displacement[:,0,None,None])
        score = float(saved['global_weights'][i]) if config.quality_weighting else 1.
        factors = torch.as_tensor(masks[:,i],**kwargs)[:,:,None,None]*score
        sums[:4] += factors[:4]*patches[None]
        weights[:4] += factors[:4]*supports[None]
        global_shift = torch.as_tensor(gshifts[i],**kwargs)
        if masks[4:,i].any():
            # Control for two separate geometric resamplings, preserving
            # original detector brightness and observed support at both steps.
            aligned = sample(raw,yy+global_shift[1],xx+global_shift[0])
            observed_global = sample(ones,yy+global_shift[1],xx+global_shift[0])
            residual = displacement-global_shift
            sequential = sample(aligned,cy+py+residual[:,1,None,None],cx+px+residual[:,0,None,None])
            sequential_support = sample(observed_global,cy+py+residual[:,1,None,None],cx+px+residual[:,0,None,None])
            sums[4:] += factors[4:]*sequential[None]
            weights[4:] += factors[4:]*sequential_support[None]
        actual += masks[:,i]
        if chosen[i]:
            global_shift = torch.as_tensor(gshifts[i],**kwargs)
            global_sum += score*sample(raw,yy+global_shift[1],xx+global_shift[0])
            global_weight += score*sample(ones,yy+global_shift[1],xx+global_shift[0])
            field = []
            for axis in range(2):
                numerator = torch.zeros(reference.shape,**kwargs)
                residual = displacement[:,axis]-global_shift[axis]
                numerator.flatten().scatter_add_(0,flat,(residual[:,None,None]*foot).flatten())
                field.append(global_shift[axis]+numerator/denominator.clamp_min(1.))
            dense_sum += score*sample(raw,yy+field[1],xx+field[0])
            dense_weight += score*sample(ones,yy+field[1],xx+field[0])
        if (i+1)%1024 == 0:
            print(json.dumps(dict(stage='stack',visited=i+1,total=n,seconds=round(time.perf_counter()-started,1))),flush=True)
    np.testing.assert_array_equal(actual,np.full((v,p),count))
    fallback = (global_sum/global_weight.clamp_min(1e-300)).cpu().numpy()
    np.testing.assert_allclose(fallback,original.image,rtol=0,atol=1e-8)
    sums,weights = sums.cpu().numpy(),weights.cpu().numpy()
    means = finalize_patches(sums,weights)
    products = {'dense_global': (dense_sum/dense_weight.clamp_min(1e-300)).cpu().numpy(),
                'global_control':fallback}
    observed = (global_weight > 0).cpu().numpy()
    report = dict(frame_count_per_ap=count,screened_pool=n,ap_count=p,ap_diameter=65,
        reference='fixed production 64-frame template',quality_sigmas=[2,4],
        global4_ranking='mean of global-aligned AP quality scores; all APs require support',
        baseline_maximum_error_adu=float(abs(fallback-original.image).max()),
        registration_reliable_fraction=float(saved['reliable'].mean()),variants={})
    cpu = CPUBackend(threads=8)
    for j,name in enumerate(names):
        valid = (weights[j]>0).astype(float)
        products[name],coverage = combine_patches(means[j],valid,centres,footprint,fallback)
        report['variants'][name] = dict(
            selected_overlap_with_global_percent=np.percentile(
                (masks[j]&chosen[:,None]).sum(axis=0)/count*100,[0,50,100]).tolist(),
            effective_frames_per_ap=np.percentile([
                saved['global_weights'][masks[j,:,k]].sum()**2/
                np.square(saved['global_weights'][masks[j,:,k]]).sum() for k in range(p)],[0,50,100]).tolist(),
            reference_peak_coverage_fraction=float(np.mean(coverage[reference>reference.max()*.15]>1e-12)))
        # High-SNR registration of completed patch stacks to their reference
        # patches. This tests a distinct recombination stage, not a known AS rule.
        adjusted,adjusted_support,offsets = register_patches(means[j], valid, centres, reference, cpu)
        products[name+'_registered'],_ = combine_patches(adjusted,adjusted_support,centres,footprint,fallback)
        report['variants'][name]['post_stack_shifts_xy'] = offsets
    np.savez_compressed(args.out/'patch_stacks.npz',means=means,weights=weights,centres=centres,footprint=footprint)
    for name,image in products.items():
        assert np.isfinite(image).all()
        base_name = name.removesuffix('_registered')
        union = (masks[names.index(base_name)].any(axis=1) | chosen) if base_name in names else chosen
        unique_count = int(union.sum())
        provenance=deepcopy(original.provenance)
        provenance['experiment']=dict(kind='patch_selection_and_recombination',variant=name,
            ap_diameter=65,n_per_ap=count,unique_contributing_frames=unique_count,normalization='none',
            coverage_units='binary observed support; per-AP accumulation weights in patch_stacks.npz',
            note='Experimental implementation; original config retained only as baseline context')
        product=replace(original,image=image,coverage=observed.astype(float),validity=observed,
            backend='gpu',layer_coverage={},n_used=unique_count,n_rejected=source.n_frames()-unique_count,
            provenance=provenance)
        save_snapshot(args.out/f'{name}.npz',product)
        export_result(product,args.out/f'{name}.png',ExportConfig('png16',mapping['black'],mapping['white']),
                      overwrite=True)  # --stage stack intentionally regenerates these experiment products.
        report['variants'].setdefault(name,{})['raw_metrics']=measurements(image)
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(complete=True,outputs=list(products))),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture',type=Path,required=True)
    parser.add_argument('--experiment',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--stage',choices=['measure','stack','all'],default='all')
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    apply_thread_limits(8)
    import torch
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    torch.set_num_threads(8)
    original=load_snapshot(args.experiment/'global_subpixel.npz')
    config=ReconstructionConfig.from_dict(original.provenance['config'])
    reference=np.load(args.experiment/'reference.npy')
    _,text=read_png(args.experiment/'local.png')
    mapping=json.loads(text)['mapping']
    layer,centres,footprint=geometry(reference)
    with SERSource(args.capture) as source:
        assert source.color_mode()=='mono'
        if args.stage in ('measure','all'):
            if (args.out/'measurements.npz').exists(): raise FileExistsError('Measurements already exist; use --stage stack')
            selection,status=load_cache(source,config)
            if selection is None: raise ValueError(status)
            measure(args,source,selection,config,reference,layer,centres)
        if args.stage in ('stack','all'):
            stack(args,source,config,reference,centres,footprint,original,mapping)


if __name__=='__main__':main()
