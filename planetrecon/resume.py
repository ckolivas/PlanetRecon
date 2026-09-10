"""Baseline accumulator checkpoints, with geometry and execution provenance."""
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from planetrecon.pipeline.provenance import capture_provenance

SCHEMA='planetrecon-baseline-state-2'
GEOMETRY_SCHEMA='planetrecon-geometry-state-1'
ARRAYS=('accum','weight','reference','demosaic_accum','demosaic_weight')
GEOMETRY_ARRAYS=ARRAYS+('globe_weight','ring_weight')


def identity(source,config,calibration,should_cancel=None):
    path=Path(source.metadata().path)
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        while True:
            if should_cancel is not None and should_cancel():
                raise InterruptedError('checkpoint input verification cancelled')
            block=stream.read(1024*1024)
            if not block:break
            digest.update(block)
    settings = config.to_dict()
    capture = capture_provenance(source,config,calibration)
    # Preserve identities of unchanged linear-weight checkpoints made before
    # this optional setting existed. Squared weights remain identity-bound.
    if not config.squared_quality_weights:
        settings.pop('squared_quality_weights')
        capture['config'].pop('squared_quality_weights')
    # The established 65-pixel matcher is unchanged. Preserve its checkpoints;
    # custom sizes remain bound to the identity so accumulators cannot be mixed.
    if config.local_patch_size == 65:
        settings.pop('local_patch_size')
        capture['config'].pop('local_patch_size')
    return {'input_sha256':digest.hexdigest(),'config':settings,'capture':capture}


def validate_destination(path,source,config):
    path=Path(path)
    for value in (source.metadata().path,config.bias_path,config.dark_path,config.flat_path):
        if value is None:continue
        other=Path(value)
        if path.resolve()==other.resolve() or (path.exists() and other.exists() and path.samefile(other)):
            raise ValueError('state checkpoint cannot replace a capture or calibration input')


def save(path,identity,state,*,geometry=False,should_cancel=None):
    path=Path(path)
    names=GEOMETRY_ARRAYS if geometry else ARRAYS
    metadata={'schema':GEOMETRY_SCHEMA if geometry else SCHEMA,'identity':identity,
        **{k:state[k] for k in ('n_used','n_rejected','reference_index','next_index')},
        'arrays':[name for name in names if state[name] is not None],
        'execution_history':state.get('execution_history',['cpu']),
        'warnings':state.get('warnings',[])}
    if should_cancel is not None and should_cancel():
        raise InterruptedError('checkpoint cancelled before publication')
    fd,tmp=tempfile.mkstemp(prefix=f'.{path.name}-',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as stream:
            from planetrecon.io.checkpoint_write import save_compressed
            save_compressed(stream,should_cancel=should_cancel,metadata=json.dumps(metadata,sort_keys=True),
                **{name:state[name] for name in metadata['arrays']})
        if should_cancel is not None and should_cancel():
            raise InterruptedError('checkpoint cancelled before publication')
        os.replace(tmp,path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def load(path,expected,shape,n_frames,bayer,*,geometry=False):
    allowed_arrays=GEOMETRY_ARRAYS if geometry else ARRAYS
    with np.load(path,allow_pickle=False) as data:
        meta=json.loads(str(data['metadata']))
        schema=meta.get('schema')
        schemas=(GEOMETRY_SCHEMA,) if geometry else (SCHEMA,'planetrecon-baseline-state-1')
        if schema not in schemas or meta.get('identity')!=expected:
            raise ValueError('state checkpoint identity/configuration mismatch')
        if schema=='planetrecon-baseline-state-1' and expected['config']['device']!='cpu':
            raise ValueError('legacy state checkpoints require CPU execution')
        names=meta.get('arrays',[])
        required={'accum','weight'} | ({'demosaic_accum','demosaic_weight'} if bayer else set())
        if geometry:required|={'globe_weight','ring_weight'}
        if not required.issubset(names) or set(names)-set(allowed_arrays):
            raise ValueError('state checkpoint arrays are incomplete or unknown')
        state={name:(data[name].copy() if name in names else None) for name in allowed_arrays}
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
    if state['n_used'] and (ref is None or (not geometry and state['reference'] is None)):
        raise ValueError('state checkpoint reference missing')
    state['reference_index']=ref
    history=meta.get('execution_history',['cpu'] if schema=='planetrecon-baseline-state-1' else None)
    warnings=meta.get('warnings',[])
    if not isinstance(history,list) or not history or any(value not in ('cpu','cuda') for value in history):
        raise ValueError('invalid checkpoint execution history')
    if not isinstance(warnings,list) or any(not isinstance(value,str) for value in warnings):
        raise ValueError('invalid checkpoint warnings')
    state['execution_history']=history
    state['warnings']=warnings
    for name,arr in ((name,state[name]) for name in allowed_arrays):
        if arr is None:continue
        expected_shape=shape[:2] if name in ('reference','globe_weight','ring_weight') else shape
        if arr.shape!=expected_shape or arr.dtype!=np.float64 or not np.isfinite(arr).all():
            raise ValueError(f'invalid state checkpoint array {name}')
        if 'weight' in name and (arr<0).any():raise ValueError('negative checkpoint weights')
    return state
