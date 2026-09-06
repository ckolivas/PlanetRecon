"""Exact CPU translation-baseline accumulator checkpoints, distinct from images."""
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from planetrecon.pipeline.provenance import capture_provenance

SCHEMA='planetrecon-baseline-state-1'
ARRAYS=('accum','weight','reference','demosaic_accum','demosaic_weight')


def identity(source,config,calibration):
    path=Path(source.metadata().path)
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return {'input_sha256':digest.hexdigest(),'config':config.to_dict(),
            'capture':capture_provenance(source,config,calibration)}


def validate_destination(path,source,config):
    path=Path(path)
    for value in (source.metadata().path,config.bias_path,config.dark_path,config.flat_path):
        if value is None:continue
        other=Path(value)
        if path.resolve()==other.resolve() or (path.exists() and other.exists() and path.samefile(other)):
            raise ValueError('state checkpoint cannot replace a capture or calibration input')


def save(path,identity,state):
    path=Path(path)
    metadata={'schema':SCHEMA,'identity':identity,
        **{k:state[k] for k in ('n_used','n_rejected','reference_index','next_index')},
        'arrays':[name for name in ARRAYS if state[name] is not None]}
    fd,tmp=tempfile.mkstemp(prefix=f'.{path.name}-',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as stream:
            np.savez_compressed(stream,metadata=json.dumps(metadata,sort_keys=True),
                **{name:state[name] for name in metadata['arrays']})
        os.replace(tmp,path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def load(path,expected,shape,n_frames,bayer):
    with np.load(path,allow_pickle=False) as data:
        meta=json.loads(str(data['metadata']))
        if meta.get('schema')!=SCHEMA or meta.get('identity')!=expected:
            raise ValueError('state checkpoint identity/configuration mismatch')
        names=meta.get('arrays',[])
        required={'accum','weight'} | ({'demosaic_accum','demosaic_weight'} if bayer else set())
        if not required.issubset(names) or set(names)-set(ARRAYS):
            raise ValueError('state checkpoint arrays are incomplete or unknown')
        state={name:(data[name].copy() if name in names else None) for name in ARRAYS}
    for key in ('n_used','n_rejected','next_index'):
        value=meta.get(key)
        if type(value) is not int or not 0<=value<=n_frames:
            raise ValueError('invalid state checkpoint counts')
        state[key]=value
    if state['next_index']!=state['n_used']+state['n_rejected']:
        raise ValueError('state checkpoint counts disagree')
    ref=meta.get('reference_index')
    if ref is not None and (type(ref) is not int or not 0<=ref<n_frames):
        raise ValueError('invalid reference index in state checkpoint')
    if state['n_used'] and (ref is None or state['reference'] is None):
        raise ValueError('state checkpoint reference missing')
    state['reference_index']=ref
    for name,arr in ((name,state[name]) for name in ARRAYS):
        if arr is None:continue
        expected_shape=shape[:2] if name=='reference' else shape
        if arr.shape!=expected_shape or arr.dtype!=np.float64 or not np.isfinite(arr).all():
            raise ValueError(f'invalid state checkpoint array {name}')
        if 'weight' in name and (arr<0).any():raise ValueError('negative checkpoint weights')
    return state
