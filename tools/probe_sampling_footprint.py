"""Development-only positive sampling footprints, with exact shifts and noise.

Images are sampled from analytic continuous scenes, never created by warping the
reference with the interpolation being tested. No sharpening or reference fit.
"""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scipy.ndimage import gaussian_filter, uniform_filter
from planetrecon.detector import cfa_labels, extract_green_proxy


def pull(image, shift, radius=1., squared=False):
    h, w = image.shape[:2]
    y, x = np.indices((h, w), dtype=float)
    y += shift[1]; x += shift[0]
    iy, ix = np.floor(y).astype(int), np.floor(x).astype(int)
    result = np.zeros_like(image, dtype=float)
    for dy in (0, 1):
        for dx in (0, 1):
            yy, xx = iy+dy, ix+dx
            weight = np.maximum(1-np.abs(y-yy)/radius, 0)*np.maximum(1-np.abs(x-xx)/radius, 0)
            weight *= (yy >= 0)&(yy < h)&(xx >= 0)&(xx < w)
            if squared: weight **= 2
            if image.ndim == 3: weight = weight[..., None]
            result += image[np.clip(yy, 0, h-1), np.clip(xx, 0, w-1)]*weight
    return result


def scene(x, y, textured):
    base = np.broadcast_to([240., 300., 200.], (*x.shape, 3)).copy()
    if textured:
        for c in range(3):
            base[..., c] += 28*np.cos(2*np.pi*(x/(8+c)+y/(15-c))+.7*c)
            base[..., c] += 19*np.sin(2*np.pi*(x/(19-c)-y/(11+c))-.3*c)
    return base


def controls():
    rows = []
    h, w = 80, 96
    y, x = np.indices((h, w), dtype=float)
    roi = np.s_[6:-6, 6:-6]
    for mode in ('mono', 'RGGB', 'GRBG', 'GBRG', 'BGGR'):
        masks = np.ones((h, w)) if mode == 'mono' else (cfa_labels(h, w, mode)[..., None] == np.array(list('RGB'))).astype(float)
        for case in ('texture_noise', 'flat_noise', 'constant_colour'):
            for seed in range(4):
                rng = np.random.default_rng(820+seed)
                shifts = rng.uniform(-2, 2, (128, 2)); shifts[0] = 0
                target = scene(x, y, case == 'texture_noise')
                if mode == 'mono': target = target[..., 1]
                arrays = {r: [np.zeros_like(target) for _ in range(3)] for r in (1., .75)}
                for shift in shifts:
                    raw = scene(x-shift[0], y-shift[1], case == 'texture_noise')
                    raw = raw[..., 1] if mode == 'mono' else np.sum(raw*masks, axis=2)
                    if case != 'constant_colour': raw += rng.normal(0, 4., raw.shape)
                    planes = raw if mode == 'mono' else raw[..., None]*masks
                    for radius, (total, support, variance) in arrays.items():
                        total += pull(planes, shift, radius)
                        support += pull(masks, shift, radius)
                        variance += pull(masks, shift, radius, squared=True)
                metrics = {}
                for radius, (total, support, variance) in arrays.items():
                    assert (support[roi] > 0).all()
                    result = np.divide(total, support, out=np.zeros_like(total), where=support > 0)
                    error = (result-target)[roi]
                    if case == 'constant_colour': assert np.max(np.abs(error)) < 1e-10
                    metrics[str(radius)] = dict(rmse=float(np.sqrt(np.mean(error**2))),
                        maximum_error=float(np.max(np.abs(error))), bias=float(error.mean()),
                        minimum_support=float(support[roi].min()),
                        predicted_noise_rmse=float(4*np.sqrt(np.mean((variance/support**2)[roi]))))
                row = dict(mode=mode, case=case, seed=seed, metrics=metrics)
                if case != 'constant_colour': row['rmse_ratio'] = metrics['0.75']['rmse']/metrics['1.0']['rmse']
                rows.append(row)
            if case != 'constant_colour':
                print(mode, case, 'narrow/standard RMSE', np.mean([r['rmse_ratio'] for r in rows[-4:]]), flush=True)
    return rows


def structure_gate(a,b):
 def band(x):
  x=gaussian_filter(x,1.5)
  return x-gaussian_filter(x,8)
 a,b=band(a),band(b)
 ma,mb=uniform_filter(a,33),uniform_filter(b,33)
 va=np.maximum(uniform_filter(a*a,33)-ma*ma,0)
 vb=np.maximum(uniform_filter(b*b,33)-mb*mb,0)
 cov=uniform_filter(a*b,33)-ma*mb
 corr=cov/np.sqrt(np.maximum(va*vb,1e-20))
 return gaussian_filter((corr>.8).astype(float),3.)

def gated_target(x,y,case):
 flat=scene(x,y,False)
 if case=='mixed':return flat+(scene(x,y,True)-flat)*np.clip((x-72)/16,0,1)[...,None]
 return scene(x,y,case=='texture_noise')


def gated_controls():
    rows=[]
    h,w=96,192;y,x=np.indices((h,w),dtype=float);roi=np.s_[8:-8,8:-8];flat_roi=np.s_[32:-32,8:24]
    for mode in ('mono','RGGB','GRBG','GBRG','BGGR'):
     masks=np.ones((h,w)) if mode=='mono' else (cfa_labels(h,w,mode)[...,None]==np.array(list('RGB'))).astype(float)
     for case in ('texture_noise','mixed','flat_noise','constant_colour'):
      for seed in range(4):
       rng=np.random.default_rng(820+seed); truth=gated_target(x,y,case)
       if mode=='mono':truth=truth[...,1]
       def observation(shift):
        raw=gated_target(x-shift[0],y-shift[1],case)
        raw=raw[...,1] if mode=='mono' else np.sum(raw*masks,axis=2)
        if case!='constant_colour':raw+=rng.normal(0,4,raw.shape)
        return raw
       templates=[]
       for half in range(2):
        total=np.zeros((h,w)); support=np.zeros((h,w))
        for shift in rng.uniform(-2,2,(32,2)):
         raw=observation(shift);proxy=raw if mode=='mono' else extract_green_proxy(raw,mode)
         total+=pull(proxy,shift);support+=pull(np.ones((h,w)),shift)
        templates.append(total/support)
       gate=structure_gate(*templates)
       if case in ('flat_noise','constant_colour'):assert np.max(gate[roi])==0, (mode,case,seed,np.max(gate[roi]))
       g=gate if mode=='mono' else gate[...,None]
       totals=[np.zeros_like(truth),np.zeros_like(truth)];supports=[np.zeros_like(truth),np.zeros_like(truth)]
       shifts=rng.uniform(-2,2,(128,2));shifts[0]=0
       for shift in shifts:
        raw=observation(shift);planes=raw if mode=='mono' else raw[...,None]*masks
        wide,narrow=pull(planes,shift),pull(planes,shift,.75)/.75**2
        sw,sn=pull(masks,shift),pull(masks,shift,.75)/.75**2
        totals[0]+=wide;supports[0]+=sw
        totals[1]+=(1-g)*wide+g*narrow;supports[1]+=(1-g)*sw+g*sn
       metrics=[]
       for total,support in zip(totals,supports):
        assert (support[roi]>0).all()
        result=np.divide(total,support,out=np.zeros_like(total),where=support>0)
        error=result-truth
        metrics.append(dict(rmse=float(np.sqrt(np.mean(error[roi]**2))),flat_rmse=float(np.sqrt(np.mean(error[flat_roi]**2))),max_error=float(np.max(np.abs(error[roi]))),min_support=float(support[roi].min())))
       if case=='constant_colour':assert metrics[1]['max_error']<1e-10
       ratio=metrics[1]['rmse']/metrics[0]['rmse'] if case!='constant_colour' else 1.
       rows.append(dict(mode=mode,case=case,seed=seed,standard=metrics[0],gated=metrics[1],rmse_ratio=ratio,mean_gate=float(gate[roi].mean()),flat_gate_max=float(gate[flat_roi].max())))
      print(mode,case,'RMSE ratio',np.mean([r['rmse_ratio'] for r in rows[-4:]]),'gate',np.mean([r['mean_gate'] for r in rows[-4:]]),flush=True)
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--gated', action='store_true', help='Use agreement between independent template halves to select the footprint')
    args = parser.parse_args()
    rows = gated_controls() if args.gated else controls()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2, allow_nan=False)+'\n')
