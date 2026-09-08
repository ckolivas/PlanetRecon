"""Known-noise experiment, not application code. No scene truth used by weights.

Interior noise propagation: independent detector variances are convolved with
squared impulse responses of Gaussian(1) -> Laplacian. Bayer measured-green
impulses also contribute 1/4 to each axial neighbour before that filter.
All evaluations exclude boundaries, where green interpolation differs.
"""
from pathlib import Path
import sys, json
import numpy as np
from scipy.ndimage import gaussian_filter, laplace, uniform_filter, convolve
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from planetrecon.detector import extract_green_proxy, cfa_labels
ROI = np.s_[36:-36, 36:-36]

def proxy(raw, mode):
    return raw if mode == 'mono' else extract_green_proxy(raw, mode)

def filtered(image):
    return laplace(gaussian_filter(image, 1.))

def energy(image):
    return uniform_filter(filtered(image)**2, 33)

def noise_energy(variance, mode):
    impulse = np.zeros((15, 15)); impulse[7, 7] = 1.
    mask = np.ones(variance.shape)
    if mode != 'mono':
        impulse[6, 7] = impulse[8, 7] = impulse[7, 6] = impulse[7, 8] = .25
        mask = (cfa_labels(*variance.shape, mode) == 'G')
    return uniform_filter(convolve(variance*mask, filtered(impulse)**2, mode='constant'), 33)

def estimate_sigma(raw, mode):
    kernel = np.outer([1., -2., 1.], [1., -2., 1.])
    planes = [raw] if mode == 'mono' else [raw[y::2, x::2] for y in range(2) for x in range(2) if cfa_labels(2, 2, mode)[y, x] == 'G']
    responses = np.concatenate([convolve(p, kernel)[3:-3, 3:-3].ravel() for p in planes])
    return float(np.median(np.abs(responses-np.median(responses)))/(.6744897501960817*6))

def local_weights(frame, template, frame_noise=None, template_noise=None):
    q = energy(frame); qr = energy(template)
    global_weight = q.mean()  # Identical scalar in both arms.
    if frame_noise is not None:
        q = np.maximum(q-frame_noise, 0)
        qr = np.maximum(qr-template_noise, 0)
    relative = (q/max(q.mean(), 1e-12))/(qr/max(qr.mean(), 1e-12)+1e-12)
    a = gaussian_filter(frame, 1.5); b = gaussian_filter(template, 1.5)
    ma, mb = uniform_filter(a, 33), uniform_filter(b, 33)
    cov = uniform_filter(a*b, 33)-ma*mb
    va = np.maximum(uniform_filter(a*a, 33)-ma*ma, 0)
    vb = np.maximum(uniform_filter(b*b, 33)-mb*mb, 0)
    corr = cov/np.sqrt(np.maximum(va*vb, 1e-20))
    return global_weight*gaussian_filter(np.where(corr > .5, np.clip(relative, .25, 4.), 1.), 3.)

def propagation_controls():
    rows = []
    for mode in ('mono', 'RGGB', 'GRBG', 'GBRG', 'BGGR'):
        rng = np.random.default_rng(941)
        # Varying variance also checks propagation of heteroscedastic noise.
        variance = np.broadcast_to(np.linspace(4, 144, 160), (128, 160))
        predicted = noise_energy(variance, mode)
        measured = np.mean([energy(proxy(rng.normal(size=variance.shape)*np.sqrt(variance), mode)) for _ in range(128)], axis=0)
        error = float(np.mean(measured[ROI])/np.mean(predicted[ROI])-1)
        assert abs(error) < .025, (mode, error)
        rows.append(dict(mode=mode, measured_mean=float(measured[ROI].mean()),
                         predicted_mean=float(predicted[ROI].mean()), relative_error=error))
    return rows

def stacking_controls():
    rows = []
    for mode in ('mono', 'RGGB'):
      for case in ('spatial_blur', 'uniform_blur', 'flat_noise'):
       for sigma in (4., 12.):
        for seed in range(4):
            rng = np.random.default_rng(seed+321)
            truth = 300+50*gaussian_filter(rng.normal(size=(128,160)), .8)
            if case == 'flat_noise': truth.fill(300.)
            xx = np.indices(truth.shape)[1]; blend=(1+np.tanh((xx-80)/8))/2
            clean=[]
            for i in range(48):
                if case == 'spatial_blur':
                    left,right=(.5,2.) if i%2 else (2.,.5)
                    clean.append((1-blend)*gaussian_filter(truth,left)+blend*gaussian_filter(truth,right))
                else: clean.append(gaussian_filter(truth,.5))
            clean=np.asarray(clean)
            template=np.mean([proxy(clean[i]+rng.normal(0,sigma,truth.shape),mode) for i in range(16)],axis=0)
            rawdata=np.asarray([f+rng.normal(0,sigma,truth.shape) for f in clean])
            data=np.asarray([proxy(f,mode) for f in rawdata])
            target=proxy(truth,mode)  # Same sampling operator; no colour claim.
            noise=noise_energy(np.full(truth.shape,sigma*sigma),mode)
            global_weights=np.array([energy(f).mean() for f in data])
            baseline=np.average(data,axis=0,weights=global_weights)
            metrics={'estimated_sigma_ratio':float(np.mean([estimate_sigma(f,mode) for f in rawdata])/sigma)}
            for label, subtract in [('raw',False),('corrected',True)]:
                w=np.asarray([local_weights(f,template,noise if subtract else None,noise/16 if subtract else None) for f in data])
                result=np.sum(data*w,axis=0)/np.sum(w,axis=0)
                metrics[label+'_rmse']=float(np.sqrt(np.mean((result[ROI]-target[ROI])**2)))
                metrics[label+'_bias']=float((result-target)[ROI].mean())
            metrics['global_rmse']=float(np.sqrt(np.mean((baseline[ROI]-target[ROI])**2)))
            metrics['noise_energy_fraction']=float(noise[ROI].mean()/np.mean([energy(f)[ROI].mean() for f in data]))
            metrics['corrected_over_raw']=metrics['corrected_rmse']/metrics['raw_rmse']
            rows.append(dict(mode=mode,case=case,sigma=sigma,seed=seed,**metrics))
        group=rows[-4:]
        print(mode,case,sigma,'corrected/raw',np.mean([r['corrected_over_raw'] for r in group]),'noise fraction',np.mean([r['noise_energy_fraction'] for r in group]),flush=True)
    return rows

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    propagation=propagation_controls(); print('propagation',propagation,flush=True)
    rows=stacking_controls()
    args.out.write_text(json.dumps(dict(propagation=propagation,stacking=rows),indent=2)+'\n')
