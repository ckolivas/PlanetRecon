"""Run the fixed local/upper-50-percent crop protocol in docs/local-quality-development.md.

Requires the private Jupiter capture and matching preprocessing cache. Refuses to
overwrite a completed pilot. The scientific gate never changes GUI settings.
"""
from pathlib import Path
import sys, json, hashlib, time
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
import numpy as np
from scipy.ndimage import shift as move, gaussian_filter
from PySide6.QtGui import QImage
import tifffile
from planetrecon.io.source import open_source
from planetrecon.reconstruction import ReconstructionConfig
from planetrecon.pipeline.preprocess_cache import load_cache
from planetrecon.pipeline.preprocess import best_frame_mask
from planetrecon.pipeline.baseline import _colour_registration_plane
from planetrecon.pipeline.local_align import LocalRegistration, build_template
from planetrecon.backends.torch_accel import TorchBackend
from tools.cfa_local_transport_probe import LocalQuadraticAccumulator
from tools.cfa_local_transport_probe import completed
from tools.compare_stack_reference import read_reference
from planetrecon.provenance import source_hash
out=root/'out/cfa-local-transport'
out.mkdir(parents=True, exist_ok=True)
if (out/'pilot.json').exists():
 raise FileExistsError('Completed pilot exists; do not repeat the study')
start=time.monotonic();version=source_hash();backend=TorchBackend()
region=(slice(164,260),slice(280,376))
with open_source(root/'2022-09-10-0649_2-GB-L-Jup_ZWO ASI224MC.ser') as source:
 cfg=ReconstructionConfig(local_alignment=True,stack_percent=50,frame_selection_mode='quality_range')
 cache, report=load_cache(source,cfg);assert cache is not None, report
 selected=np.flatnonzero(best_frame_mask(cache,50,'quality_range'))
 q=cache.measurements[:,0]; anchor=int(selected[np.argmax(q[selected])])
 indices=selected[np.linspace(0,len(selected)-1,64).round().astype(int)]
 if anchor not in indices:indices[np.argmin(abs(indices-anchor))]=anchor;indices.sort()
 train=selected[np.argsort(-q[selected],kind='stable')[:64]]
 color=source.color_mode();shape=source.frame_shape()
 plane=lambda i:_colour_registration_plane(source.read_raw(int(i)),color)
 template=build_template(plane(anchor),train,plane,backend.phase_correlation,cfg.max_shift_px)
 matcher=LocalRegistration(template,use_cuda=True)
 model=LocalQuadraticAccumulator(shape,color,region=region)
 map_digest=hashlib.sha256(); raw_digest=hashlib.sha256()
 for n,i in enumerate(indices):
  raw=source.read_raw(int(i)); raw_digest.update(raw.tobytes())
  proxy=_colour_registration_plane(raw,color)
  global_shift=backend.phase_correlation(template,proxy)
  assert max(abs(v) for v in global_shift)<=cfg.max_shift_px
  local=matcher.displacement(proxy,global_shift)
  map_digest.update(np.asarray(local).tobytes())
  model.add(raw,local,float(q[i]))
  if (n+1)%8==0:print('frames',n+1,'elapsed',time.monotonic()-start,flush=True)
 after,variance,degree,before=model.finish()
 before.n_used=len(indices)
 full=completed(np.divide(model.sums,model.weights,out=np.zeros_like(model.sums),where=model.weights>0),model.weights,color)
 reference=read_reference(root/'2022-09-10-0649_2-GB-L-Jup_ZWO ASI224MC_lapl6_ap59.png')
 dx,dy=backend.phase_correlation(reference[...,1],full.image[...,1])
 target=move(reference,(dy,dx,0),order=1,mode='constant',prefilter=False)[region]
 mask=before.validity.all(axis=-1);mask[:8]=False;mask[-8:]=False;mask[:,:8]=False;mask[:,-8:]=False
 metrics={};previews=[]
 for name,image in [('ordinary',before.image),('quadratic',after)]:
  aligned=image.copy();fits=[]
  for c in range(3):
   a,b=np.linalg.lstsq(np.column_stack([image[...,c][mask],np.ones(mask.sum())]),target[...,c][mask],rcond=None)[0]
   aligned[...,c]=image[...,c]*a+b;fits.append([a,b])
  hp=aligned-gaussian_filter(aligned,(3,3,0));refhp=target-gaussian_filter(target,(3,3,0))
  metrics[name]={'relative_rmse':float(np.linalg.norm((aligned-target)[mask])/np.linalg.norm(target[mask])),
   'highpass_correlation_sigma3':float(np.corrcoef(hp[mask].ravel(),refhp[mask].ravel())[0,1]),'gain_offset':fits}
  tifffile.imwrite(out/(name+'.tif'),image.astype('float32'),photometric='rgb')
  previews.append(aligned)
 white=min(65535.,max(float(v.max()) for v in [*previews,target])*1.43)
 for name,image in zip(['ordinary','quadratic','reference'],[*previews,target]):
  pixels=np.ascontiguousarray(np.uint8(np.clip(image/white,0,1)*255))
  preview=QImage(pixels.data,pixels.shape[1],pixels.shape[0],pixels.strides[0],QImage.Format.Format_RGB888)
  assert preview.save(str(out/(name+'.png')))
 np.savez_compressed(out/'comparison.npz',ordinary=before.image,quadratic=after,variance=variance,degree=degree,reference=target,indices=indices,qualities=q[indices],template=template,mask=mask)
 assert source_hash()==version
 payload={'scope':'Fixed 64-frame central-crop development comparison; conventional reference is not truth',
  'source_hash':version,'probe_sha256':hashlib.sha256((root/'tools/cfa_local_transport_probe.py').read_bytes()).hexdigest(),
  'selected_capture_frames':len(selected),'sample_frames':len(indices),'indices':indices.tolist(),'anchor':anchor,
  'template_indices':train.tolist(),'raw_sha256':raw_digest.hexdigest(),'maps_sha256':map_digest.hexdigest(),
  'selection':'upper 50% capture quality range then cached screening','weight':'original linear cached Emil quality',
  'output_region_yx':[164,260,280,376],'registration_xy':[dx,dy],'common_pixels':int(mask.sum()),
  'degree_counts':{str(k):int((degree==k).sum()) for k in [-1,0,1,2]},'metrics':metrics,
  'passes_both_metrics':metrics['quadratic']['relative_rmse']<metrics['ordinary']['relative_rmse'] and metrics['quadratic']['highpass_correlation_sigma3']>metrics['ordinary']['highpass_correlation_sigma3'],
  'elapsed_s':time.monotonic()-start}
 (out/'pilot.json').write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload),flush=True)
