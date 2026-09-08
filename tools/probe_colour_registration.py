"""Known-shift low-contrast Bayer controls; no reconstruction/sharpening change."""
from pathlib import Path
import sys,json,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.ndimage import gaussian_filter, fourier_shift
from planetrecon.detector import cfa_labels,extract_green_proxy,bilinear_demosaic
from planetrecon.pipeline.align import phase_correlation_shift
from planetrecon.pipeline.local_align import LocalRegistration


def luminance(raw,pattern):
    return bilinear_demosaic(raw,pattern) @ np.array([.25,.5,.25])


def controls():
    rows=[]
    shape=(128,160)
    cases={'balanced':([150.,150.,150.],[1.,1.,1.]),
           'jupiter_colour_ratio':([120.,150.,80.],[1.,1.,1.]),
           'weak_red_blue':([12.,150.,8.],[1.,1.,1.]),
           'unequal_channel_blur':([120.,150.,80.],[1.8,1.,2.5])}
    for pattern in ('RGGB','GRBG','GBRG','BGGR'):
      masks=cfa_labels(*shape,pattern)[...,None]==np.array(list('RGB'))
      for case,(means,sigmas) in cases.items():
       for seed in range(6):
        rng=np.random.default_rng(749+seed)
        texture=gaussian_filter(rng.normal(size=shape),1.,mode='wrap');texture/=texture.std()
        clean=np.stack([mean+.025*mean*gaussian_filter(texture,sigma,mode='wrap') for mean,sigma in zip(means,sigmas)],axis=2)
        # Known independent shot-like plus read variance, not fitted to the scene.
        variance=.23*clean+4.
        template_rgb=clean+rng.normal(size=clean.shape)*np.sqrt(variance/64)
        template_raw=np.sum(template_rgb*masks,axis=2)
        references=[extract_green_proxy(template_raw,pattern),luminance(template_raw,pattern)]
        spectra=[np.fft.fftn(clean[...,c]) for c in range(3)]
        errors=[[],[]]
        for shift in rng.uniform(-1.5,1.5,(16,2)):
            moved=np.stack([np.fft.ifftn(fourier_shift(s,(shift[1],shift[0]))).real for s in spectra],axis=2)
            raw=np.sum((moved+rng.normal(size=clean.shape)*np.sqrt(np.maximum(.23*moved+4,0)))*masks,axis=2)
            proxies=[extract_green_proxy(raw,pattern),luminance(raw,pattern)]
            for j in range(2):
                estimated=phase_correlation_shift(references[j],proxies[j])
                errors[j].append(np.asarray(estimated)-shift)
        rows.append(dict(pattern=pattern,case=case,seed=seed,green_rms_px=float(np.sqrt(np.mean(np.sum(np.asarray(errors[0])**2,axis=1)))),luminance_rms_px=float(np.sqrt(np.mean(np.sum(np.asarray(errors[1])**2,axis=1)))),green_bias_xy=np.mean(errors[0],axis=0).tolist(),luminance_bias_xy=np.mean(errors[1],axis=0).tolist()))
       group=rows[-6:]
       print(pattern,case,'luminance/green registration RMS',np.sqrt(sum(r['luminance_rms_px']**2 for r in group)/sum(r['green_rms_px']**2 for r in group)),flush=True)
      constant=np.sum(np.broadcast_to([120.,150.,80.],(*shape,3))*masks,axis=2)
      for proxy in (extract_green_proxy,luminance):
        plane=proxy(constant,pattern)
        assert np.max(plane)-np.min(plane)<1e-12
        assert phase_correlation_shift(plane,plane)==(0.,0.)
        fields=LocalRegistration(plane).displacement(plane,(0.,0.))
        assert max(abs(d).max() for d in fields)==0
      # Unrelated noise must not establish local motion against a flat template.
      flat=luminance(constant,pattern)
      noise=luminance(constant+np.random.default_rng(31).normal(0,6,shape),pattern)
      assert max(abs(d).max() for d in LocalRegistration(flat).displacement(noise,(0.,0.)))==0
    return rows

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();rows=controls();args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n')
