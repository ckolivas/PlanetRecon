"""Baseline accumulator checkpoints, with geometry and execution provenance."""
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from planetrecon.pipeline.provenance import capture_provenance

SCHEMA='planetrecon-baseline-state-2'
LOCAL_SCHEMA='planetrecon-local-cfa-application-state-1'
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
    if not config.local_cfa_interpolation:
        settings.pop('local_cfa_interpolation')
        capture['config'].pop('local_cfa_interpolation')
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
    local = 'local_cfa' in state
    if local:
        if path.suffix != '.npz':
            raise ValueError('local state checkpoint must be an NPZ')
        if path.exists():
            try:
                with np.load(path, allow_pickle=False) as data:
                    previous = json.loads(str(data['metadata']))
            except (OSError, ValueError, KeyError) as exc:
                raise ValueError('checkpoint destination is not a recognized local state') from exc
            if previous.get('schema') != LOCAL_SCHEMA or previous.get('identity') != identity:
                raise ValueError('checkpoint destination belongs to another input or policy')
        from planetrecon.pipeline.cfa_application import MOMENTS
        names += tuple('local_'+name for name in MOMENTS)
    metadata={'schema':GEOMETRY_SCHEMA if geometry else SCHEMA,'identity':identity,
        **{k:state[k] for k in ('n_used','n_rejected','reference_index','next_index')},
        'arrays':[name for name in names if state[name] is not None],
        'execution_history':state.get('execution_history',['cpu']),
        'warnings':state.get('warnings',[])}
    if local:
        from planetrecon.pipeline.cfa_checkpoint import array_digest, canonical
        metadata.update(schema=LOCAL_SCHEMA, local_cfa=state['local_cfa'])
        metadata['array_sha256'] = {name: array_digest(state[name]) for name in metadata['arrays']}
        metadata['metadata_sha256'] = hashlib.sha256(canonical(metadata).encode()).hexdigest()
    if should_cancel is not None and should_cancel():
        raise InterruptedError('checkpoint cancelled before publication')
    fd,tmp=tempfile.mkstemp(prefix=f'.{path.name}-',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as stream:
            from planetrecon.io.checkpoint_write import save_compressed
            save_compressed(stream, should_cancel=should_cancel,
                metadata=json.dumps(metadata,sort_keys=True),
                **{name:state[name] for name in metadata['arrays']})
        if should_cancel is not None and should_cancel():
            raise InterruptedError('checkpoint cancelled before publication')
        os.replace(tmp,path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def load(path,expected,shape,n_frames,bayer,*,geometry=False,local_model=None):
    allowed_arrays=GEOMETRY_ARRAYS if geometry else ARRAYS
    if local_model is not None:
        import zipfile
        from planetrecon.pipeline.cfa_application import MOMENTS
        from planetrecon.pipeline.cfa_checkpoint import array_digest, canonical
        allowed_arrays += tuple('local_'+name for name in MOMENTS)
        with zipfile.ZipFile(path) as archive:
            for name in allowed_arrays:
                with archive.open(name+'.npy') as stream:
                    version = np.lib.format.read_magic(stream)
                    if version not in ((1, 0), (2, 0)):
                        raise ValueError('unsupported local checkpoint array format')
                    reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
                    dimensions, fortran, dtype = reader(stream)
                    target_shape = (getattr(local_model, name[6:]).shape if name.startswith('local_')
                                    else shape[:2] if name == 'reference' else shape)
                    if dimensions != target_shape or fortran or dtype != np.dtype('float64'):
                        raise ValueError('invalid local checkpoint array header '+name)
    with np.load(path,allow_pickle=False) as data:
        meta=json.loads(str(data['metadata']))
        schema=meta.get('schema')
        schemas=(GEOMETRY_SCHEMA,) if geometry else (SCHEMA,'planetrecon-baseline-state-1')
        if local_model is not None:
            schemas = (LOCAL_SCHEMA,)
            checksum = meta.pop('metadata_sha256', None)
            if checksum != hashlib.sha256(canonical(meta).encode()).hexdigest():
                raise ValueError('local checkpoint metadata checksum mismatch')
        if schema not in schemas or meta.get('identity')!=expected:
            raise ValueError('state checkpoint identity/configuration mismatch')
        if schema=='planetrecon-baseline-state-1' and expected['config']['device']!='cpu':
            raise ValueError('legacy state checkpoints require CPU execution')
        names=meta.get('arrays',[])
        required={'accum','weight'} | ({'demosaic_accum','demosaic_weight'} if bayer else set())
        if local_model is not None:
            required = set(allowed_arrays)
        if geometry:required|={'globe_weight','ring_weight'}
        if not required.issubset(names) or set(names)-set(allowed_arrays):
            raise ValueError('state checkpoint arrays are incomplete or unknown')
        if local_model is not None and set(data.files) != {'metadata', *allowed_arrays}:
            raise ValueError('local checkpoint arrays are missing or unknown')
        state={name:(data[name] if name in names else None) for name in allowed_arrays}
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
        expected_shape=(getattr(local_model, name[6:]).shape if name.startswith('local_') else
                        shape[:2] if name in ('reference','globe_weight','ring_weight') else shape)
        if arr.shape!=expected_shape or arr.dtype!=np.float64 or not np.isfinite(arr).all():
            raise ValueError(f'invalid state checkpoint array {name}')
        if 'weight' in name and (arr<0).any():raise ValueError('negative checkpoint weights')
    if local_model is not None:
        for name in allowed_arrays:
            if meta.get('array_sha256', {}).get(name) != array_digest(state[name]):
                raise ValueError('local checkpoint array checksum mismatch '+name)
            if state['n_used'] == 0 and name != 'reference' and state[name].any():
                raise ValueError('empty local checkpoint contains contributions')
        if (state['local_variance_sum'] < 0).any() or not isinstance(meta.get('local_cfa'), dict):
            raise ValueError('invalid local checkpoint state')
        state['local_cfa'] = meta['local_cfa']
    return state
