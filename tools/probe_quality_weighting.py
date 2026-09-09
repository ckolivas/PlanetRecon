"""Known-blur controls of cached scalar quality weights and their squares."""
from pathlib import Path
import sys,json,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.ndimage import gaussian_filter
from planetrecon.detector import cfa_labels
from planetrecon.pipeline.preprocess import measure_frame,sigma_selection

def controls():
 rows=[];y,x=np.indices((128,160));disk=(x-80)**2+(y-64)**2<46**2
 roi=(x-80)**2+(y-64)**2<38**2
 for colour in ('mono','RGGB','GRBG','GBRG','BGGR'):
  colour_scale=np.ones(disk.shape) if colour=='mono' else sum(v*(cfa_labels(*disk.shape,colour)==c) for v,c in zip((.8,1.,.6),'RGB'))
  for case in ('variable_blur','uniform_blur','flat_interior'):
   for seed in range(4):
    rng=np.random.default_rng(303+seed)
    texture=gaussian_filter(rng.normal(size=disk.shape),.8);texture/=texture.std()
    surface=disk*(140+(5*texture if case!='flat_interior' else 0))
    target=10+colour_scale*gaussian_filter(surface,.8)
    data=[];measurements=[]
    sigmas=rng.uniform(.8,2.2,96) if case=='variable_blur' else np.full(96,.8)
    for sigma in sigmas:
     clean=10+colour_scale*gaussian_filter(surface,sigma)
     frame=np.clip(np.rint(clean+rng.normal(size=clean.shape)*np.sqrt(.23*clean+4)),0,255)
     data.append(frame);measurements.append(measure_frame(frame,colour))
    values=np.array([r[:4] for r in measurements]);valid=np.array([r[4]=='ok' for r in measurements])
    accepted,_,_=sigma_selection(values,valid);assert accepted.sum()>=64
    frames=np.asarray(data)[accepted];q=values[accepted,0];assert np.isfinite(q).all() and (q>0).all()
    metrics={}
    for power in (1,2):
     weights=q**power;result=np.average(frames,axis=0,weights=weights)
     metrics[str(power)]=dict(rmse=float(np.sqrt(np.mean((result-target)[roi]**2))),bias=float(np.mean((result-target)[roi])),nominal_effective_count=float(weights.sum()**2/(weights*weights).sum()))
    rows.append(dict(colour=colour,case=case,seed=seed,frames=int(accepted.sum()),metrics=metrics,rmse_ratio=metrics['2']['rmse']/metrics['1']['rmse']))
   print(colour,case,'squared/linear RMS',np.mean([r['rmse_ratio'] for r in rows[-4:]]),flush=True)
 return rows

if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
 args=parser.parse_args();rows=controls();args.out.parent.mkdir(parents=True,exist_ok=True)
 args.out.write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n')
