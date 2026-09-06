"""Private real-capture CPU qualification with a predeclared bounded assessment.

Each CLI invocation is a fresh process; RSS is its lifetime high-water mark.
No capture pixels, observer metadata or filenames are published by the report.
"""
from dataclasses import replace
import hashlib
import json
import platform
from pathlib import Path
import time

import numpy as np
from scipy.ndimage import shift

from planetrecon.detector import cfa_labels, is_bayer
from planetrecon.io.source import FrameSource, open_source
from planetrecon.pipeline.align import phase_correlation_shift
from planetrecon.pipeline.baseline import stack_source, _alignment_plane
from planetrecon.provenance import source_hash
from planetrecon.reconstruction import ReconstructionConfig
from planetrecon.result import save_snapshot
from planetrecon.runtime import apply_thread_limits


class IndexedSource(FrameSource):
    def __init__(self, source, indices):
        self.source, self.indices = source, np.asarray(indices, dtype=np.int64)
    def metadata(self):
        return replace(self.source.metadata(), n_frames=self.n_frames())
    def n_frames(self):
        return len(self.indices)
    def frame_shape(self):
        return self.source.frame_shape()
    def color_mode(self):
        return self.source.color_mode()
    def read_raw(self, index):
        return self.source.read_raw(int(self.indices[index]))


def capture_sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def split_half_metric(a,b):
    def plane(r):
        image=np.where(r.validity,r.image,0.)
        return image[...,1] if image.ndim==3 else image
    dx,dy=phase_correlation_shift(plane(a),plane(b))
    offsets=(-dy,-dx,0) if b.image.ndim==3 else (-dy,-dx)
    other=shift(b.image,offsets,order=1,prefilter=False,mode='grid-constant')
    valid=a.validity & (shift(b.validity.astype(float),offsets,order=1,prefilter=False,mode='grid-constant') > 1-1e-9)
    valid &= np.isfinite(a.image) & np.isfinite(other)
    if not valid.any():
        return {'status':'invalid','reason':'no common coverage'}
    denom=float(np.sum(.5*(a.image[valid]**2+other[valid]**2)))
    return {'status':'diagnostic','relative_rms':float(np.sqrt(np.sum((a.image[valid]-other[valid])**2)/max(denom,1e-20))),
        'common_samples':int(valid.sum()),'shift_xy_px':[dx,dy]}


def raw_residual(source,result,indices,config):
    """In-sample code-value residual, with detector-fixed Bayer sampling."""
    color=source.color_mode()
    ref=source.read_raw(result.provenance['reference_index'])
    refplane=_alignment_plane(ref,color)
    labels=cfa_labels(*ref.shape[:2],color) if is_bayer(color) else None
    error=energy=0.
    count=frames=0
    for index in indices:
        raw=source.read_raw(int(index)).astype(float)
        dx,dy=phase_correlation_shift(refplane,_alignment_plane(raw,color))
        if abs(dx)>config.max_shift_px or abs(dy)>config.max_shift_px:
            continue
        offsets=(dy,dx,0) if result.image.ndim==3 else (dy,dx)
        pred=shift(result.image,offsets,order=1,prefilter=False,mode='grid-constant')
        valid=shift(result.validity.astype(float),offsets,order=1,prefilter=False,mode='grid-constant') > 1-1e-9
        if labels is not None:
            mapped=np.zeros(raw.shape); mask=np.zeros(raw.shape,dtype=bool)
            for c,name in enumerate('RGB'):
                sites=labels==name
                mapped[sites]=pred[...,c][sites]; mask[sites]=valid[...,c][sites]
            pred,valid=mapped,mask
        valid &= np.isfinite(raw) & np.isfinite(pred)
        error+=float(np.sum((raw[valid]-pred[valid])**2));energy+=float(np.sum(raw[valid]**2))
        count+=int(valid.sum());frames+=1
    return {'status':'diagnostic' if count else 'invalid','relative_rms':float(np.sqrt(error/max(energy,1e-20))) if count else None,
        'samples':count,'frames':frames,'role':'in-sample unwhitened residual; not independent prediction or noise calibration'}


def run(path, out, capture_id, *, threads=2, sample_frames=128, max_frames=None, device="cpu", color_override=None, uniform=False):
    if sample_frames < 4 or (max_frames is not None and max_frames < 4):
        raise ValueError('at least four frames required')
    apply_thread_limits(threads)
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    config=ReconstructionConfig(device=device,threads=threads,batch_frames=32,bayer_override=color_override)
    protocol={'capture_id':capture_id,'distribution_permission':False,'permission_to_process':'user supplied local capture',
        'config':config.to_dict(),'sample_frames':sample_frames,'max_frames':max_frames,
        'sampling':'uniform capture subsample' if uniform else 'processed prefix; uniform assessment indices',
        'assessment':'baseline only; no geometry or atmospheric inference; no sharpening',
        'source_hash':source_hash(),'input_sha256':capture_sha256(path)}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    with open_source(path,bayer_override=color_override) as original:
        count=original.n_frames() if max_frames is None else min(original.n_frames(),max_frames)
        if count<4:raise ValueError('at least four capture frames required')
        selected=np.linspace(0,original.n_frames()-1,count,dtype=np.int64) if uniform else np.arange(count)
        source=original if count==original.n_frames() else IndexedSource(original,selected)
        start=time.monotonic();first=None;last_print=0
        def event(result,info):
            nonlocal first,last_print
            if result.n_used and first is None:first=time.monotonic()-start
            if info['n_processed']-last_print >= 1000:
                last_print=info['n_processed'];print(f'{capture_id}: {last_print}/{count}',flush=True)
        result=stack_source(source,config,on_event=event)
        wall=time.monotonic()-start
        save_snapshot(out/'result.npz',result)
        indices=np.linspace(0,count-1,min(sample_frames,count),dtype=np.int64)
        halves=[stack_source(IndexedSource(source,idx),config) for idx in (indices[::2],indices[1::2])]
        split=split_half_metric(*halves)
        residual=raw_residual(source,result,indices,config)
        try:
            import resource
            rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if platform.system()=='Darwin' else 1024)
        except ImportError:
            rss=None
        report={'status':'diagnostic','capture_id':capture_id,'input_sha256':protocol['input_sha256'],
            'source_hash':protocol['source_hash'],'hardware':{'system':platform.system(),'machine':platform.machine(),'processor':platform.processor()},
            'backend':result.backend,'precision':result.precision,'threads':threads,'requested_device':device,
            'source_color':original.metadata().extras.get('color_id').value if 'color_id' in original.metadata().extras else original.color_mode(),
            'color_override':color_override,'sampling':protocol['sampling'],
            'device_report':result.provenance.get('device_report'),
            'source_frames':original.n_frames(),'processed_frames':count,'used_frames':result.n_used,
            'rejected_frames':result.n_rejected,'frame_shape':source.frame_shape(),'color':source.color_mode(),
            'bit_depth':source.metadata().bit_depth,'wall_s':wall,'frames_per_s':count/wall,
            'first_preview_s':first,'peak_rss_bytes':rss,'rss_scope':'whole benchmark process including assessment',
            'split_half':split,'raw_residual':residual,'distribution_permission':False,
            'geometry_qualified':False,'gpu_qualified':False,'full_capture':count==original.n_frames()}
        (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print(json.dumps(report),flush=True)
        return report


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--path',required=True,type=Path)
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--id',required=True)
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--sample-frames',type=int,default=128)
    parser.add_argument('--max-frames',type=int)
    parser.add_argument('--device',choices=['cpu','gpu','auto'],default='cpu')
    parser.add_argument('--color',choices=['mono','RGGB','GRBG','GBRG','BGGR'])
    parser.add_argument('--uniform',action='store_true',help='sample over the full capture when max-frames is set')
    args=parser.parse_args()
    run(args.path,args.out,args.id,threads=args.threads,sample_frames=args.sample_frames,max_frames=args.max_frames,device=args.device,color_override=args.color,uniform=args.uniform)
