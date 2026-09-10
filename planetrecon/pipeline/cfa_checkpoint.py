"""Exact accumulation checkpoints for fixed raw frames and dense local maps.

This wraps the separate interpolation probe. It does not estimate alignment or
re-estimate maps on CUDA failure. Every supplied frame/map must match the
manifest before any of its contributions are published.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import sys
import zipfile

import scipy

import numpy as np

from planetrecon.provenance import source_hash
from planetrecon.pipeline.cfa_interpolation import LocalQuadraticAccumulator

SCHEMA = 'planetrecon-local-quadratic-state-1'
ARRAYS = ('gram', 'noise', 'rhs', 'sums', 'weights', 'variance_sum')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def array_digest(value):
    value = np.ascontiguousarray(value, dtype='<f8')
    digest = hashlib.sha256(canonical(list(value.shape)).encode())
    digest.update(memoryview(value).cast('B'))
    return digest.hexdigest()


def map_digest(shift, shape):
    return array_digest(np.stack([np.broadcast_to(v, shape) for v in shift]))


def make_manifest(source_path, source_sha256, shape, color, indices, qualities,
                  raw_digests, map_digests, *, region=None, chunk_rows=64,
                  max_array_bytes=2*1024**3, max_neighbour_entries=65536,
                  device='cpu', gpu_chunk_rows=512, max_gpu_array_bytes=128*1024**2):
    """Bind externally frozen calibrated raw inputs and maps to an execution policy."""
    shape = tuple(shape)
    if len(shape) != 2 or any(type(v) is not int or v < 2 for v in shape):
        raise ValueError('positive detector shape required')
    indices = list(indices)
    if (not indices or any(type(v) is not int or v < 0 for v in indices)
            or indices != sorted(set(indices))):
        raise ValueError('nonempty strictly increasing source indices required')
    qualities = list(qualities)
    if len(qualities) != len(indices) or not np.isfinite(qualities).all() or not (np.asarray(qualities) > 0).all():
        raise ValueError('positive finite quality per selected frame required')
    def validate_digests(values):
        if any(not isinstance(v, str) or len(v) != 64 or any(c not in '0123456789abcdef' for c in v) for v in values):
            raise ValueError('SHA256 digests required')
    raw_digests, map_digests = list(raw_digests), list(map_digests)
    if len(raw_digests) != len(indices) or len(map_digests) != len(indices):
        raise ValueError('one raw and map digest per selected frame required')
    validate_digests([source_sha256, *raw_digests, *map_digests])
    region = (slice(0, shape[0]), slice(0, shape[1])) if region is None else region
    # The accumulator validates execution options when the run is constructed.
    # Manifest serialization rejects nonfinite settings.
    options = {'shape': list(shape), 'color': color, 'region': [[v.start, v.stop, v.step] for v in region],
               'chunk_rows': chunk_rows, 'max_array_bytes': max_array_bytes,
               'max_neighbour_entries': max_neighbour_entries}
    value = {'source_path': str(Path(source_path).resolve()), 'source_sha256': source_sha256,
             'indices': indices, 'qualities': [float(v) for v in qualities],
             'raw_digests': raw_digests, 'map_digests': map_digests, 'options': options,
             'operator': {'name': 'reference-coordinate-local-quadratic',
                          'package_sha256': source_hash(),
                          'probe_sha256': hashlib.sha256(Path(__file__).with_name('cfa_interpolation.py').read_bytes()).hexdigest(),
                          'radius': 4, 'rank_relative_threshold': 1e-10,
                          'fallback': 'quadratic then affine under ordinary iid-noise variance cap then ordinary completion'},
             'execution': {'accumulation': 'cpu-float64', 'frame_order': 'manifest order',
                           'numpy': np.__version__, 'scipy': scipy.__version__, 'byteorder': sys.byteorder,
                           'resume_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}
    if device not in ('cpu', 'gpu'):
        raise ValueError('local moment device must be cpu or gpu')
    if device == 'gpu':
        value['options'].update(device=device, gpu_chunk_rows=gpu_chunk_rows, max_gpu_array_bytes=max_gpu_array_bytes)
        value['execution'].update(accumulation='cuda-float64-with-whole-frame-cpu-retry',
                                  gpu_sha256=hashlib.sha256(Path(__file__).with_name('cfa_cuda.py').read_bytes()).hexdigest())
    return json.loads(canonical(value))


def validate_gpu_state(state, history, count):
    """Validate the complete-frame CUDA-to-CPU execution contract."""
    gpu_state = state
    keys = {'gpu_enabled', 'fallback_reason', 'fallback_frame', 'device_identity'}
    if not isinstance(gpu_state, dict) or set(gpu_state) != keys:
        raise ValueError('missing or invalid GPU state')
    if (not isinstance(history, list) or len(history) != count
            or any(v not in ('cpu', 'cuda') for v in history)
            or type(gpu_state['gpu_enabled']) is not bool):
        raise ValueError('checkpoint execution history mismatch')
    fallback = gpu_state['fallback_frame']
    if gpu_state['gpu_enabled']:
        if history != ['cuda']*count or fallback is not None or gpu_state['fallback_reason'] is not None:
            raise ValueError('invalid active GPU history')
    elif (type(fallback) is not int or not 0 <= fallback <= count
            or history != ['cuda']*fallback+['cpu']*(count-fallback)
            or not isinstance(gpu_state['fallback_reason'], str) or not gpu_state['fallback_reason']):
        raise ValueError('invalid CPU retry history')
    identity = gpu_state['device_identity']
    if identity is not None:
        if (not isinstance(identity, dict) or set(identity) != {'name', 'capability', 'multiprocessors', 'torch', 'cuda'}
                or not isinstance(identity['name'], str) or not isinstance(identity['torch'], str)
                or not isinstance(identity['cuda'], str)
                or type(identity['multiprocessors']) is not int or identity['multiprocessors'] < 1
                or not isinstance(identity['capability'], list) or len(identity['capability']) != 2
                or any(type(v) is not int or v < 0 for v in identity['capability'])):
            raise ValueError('invalid CUDA device identity')
    if 'cuda' in history and identity is None:
        raise ValueError('CUDA history lacks a device identity')
    return deepcopy(gpu_state)


class FixedLocalRun:
    def __init__(self, manifest, *, should_cancel=None):
        self._manifest = json.loads(canonical(manifest))
        options = deepcopy(self._manifest['options'])
        options['region'] = tuple(slice(*v) for v in options['region'])
        rebuilt = make_manifest(self._manifest['source_path'], self._manifest['source_sha256'],
                                tuple(options['shape']), options['color'], self._manifest['indices'],
                                self._manifest['qualities'], self._manifest['raw_digests'],
                                self._manifest['map_digests'], **{k: v for k, v in options.items() if k not in ('shape', 'color')})
        if rebuilt != self._manifest:
            raise ValueError('interpolation operator or execution manifest changed')
        device = options.pop('device', 'cpu')
        if device == 'gpu':
            from planetrecon.pipeline.cfa_cuda import CudaLocalAccumulator
            self.model = CudaLocalAccumulator(**options, should_cancel=should_cancel)
        else:
            self.model = LocalQuadraticAccumulator(**options, should_cancel=should_cancel)

    @property
    def manifest(self):
        return deepcopy(self._manifest)

    def add(self, source_index, raw, shift):
        self.model._check_cancel()
        cursor = self.model.n_used
        if cursor >= len(self._manifest['indices']) or source_index != self._manifest['indices'][cursor]:
            raise ValueError('frame does not match the next selected source index')
        if array_digest(raw) != self._manifest['raw_digests'][cursor]:
            raise ValueError('raw frame differs from the frozen input')
        if map_digest(shift, self.model.shape) != self._manifest['map_digests'][cursor]:
            raise ValueError('local map differs from the frozen alignment')
        self.model._check_cancel()
        self.model.add(raw, shift, self._manifest['qualities'][cursor])

    def finish(self):
        if self.model.n_used != len(self._manifest['indices']):
            raise ValueError('cannot publish final output before all selected frames complete')
        return self.model.finish()

    def _destination(self, path):
        path = Path(path)
        source = Path(self._manifest['source_path'])
        if path.suffix != '.npz' or path.resolve() == source.resolve() or (path.exists() and source.exists() and path.samefile(source)):
            raise ValueError('checkpoint must be an NPZ distinct from the source')
        if path.exists():
            try:
                with np.load(path, allow_pickle=False) as data:
                    metadata = json.loads(str(data['metadata']))
            except (OSError, ValueError, KeyError) as exc:
                raise ValueError('checkpoint destination is not a recognized local state') from exc
            if metadata.get('schema') != SCHEMA or metadata.get('manifest') != self._manifest:
                raise ValueError('checkpoint destination belongs to another input or policy')
        return path

    def save(self, path):
        self.model._check_cancel()
        path = self._destination(path)
        hashes = {}
        for name in ARRAYS:
            self.model._check_cancel()
            hashes[name] = array_digest(getattr(self.model, name))
        metadata = {'schema': SCHEMA, 'manifest': self._manifest, 'n_used': self.model.n_used,
                    'execution_history': getattr(self.model, 'execution_history', ['cpu'] if self.model.n_used else []), 'array_sha256': hashes}
        if self._manifest['options'].get('device') == 'gpu':
            metadata['gpu_state'] = {key: getattr(self.model, key) for key in (
                'gpu_enabled', 'fallback_reason', 'fallback_frame', 'device_identity')}
        metadata['metadata_sha256'] = hashlib.sha256(canonical(metadata).encode()).hexdigest()
        fd, temporary = tempfile.mkstemp(prefix=f'.{path.name}-', suffix='.tmp', dir=path.parent)
        try:
            with os.fdopen(fd, 'wb') as stream:
                np.savez_compressed(stream, metadata=canonical(metadata),
                                    **{name: getattr(self.model, name) for name in ARRAYS})
            self.model._check_cancel()
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)

    @classmethod
    def load(cls, path, expected_manifest, *, should_cancel=None):
        run = cls(expected_manifest, should_cancel=should_cancel)
        run.model._check_cancel()
        staged = {}
        # Read only NPY headers before decompressing array payloads. Reject a
        # corrupt shape/object dtype before it can bypass the model's array plan.
        with zipfile.ZipFile(path) as archive:
            for name in ARRAYS:
                run.model._check_cancel()
                with archive.open(name+'.npy') as stream:
                    version = np.lib.format.read_magic(stream)
                    if version not in ((1, 0), (2, 0)):
                        raise ValueError('unsupported checkpoint array format')
                    read_header = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                                   else np.lib.format.read_array_header_2_0)
                    shape, fortran, dtype = read_header(stream)
                    template = getattr(run.model, name)
                    if shape != template.shape or dtype != np.dtype('float64') or fortran:
                        raise ValueError(f'invalid checkpoint array header {name}')
        with np.load(path, allow_pickle=False) as data:
            if set(data.files) != {'metadata', *ARRAYS}:
                raise ValueError('checkpoint arrays are missing or unknown')
            metadata = json.loads(str(data['metadata']))
            checksum = metadata.pop('metadata_sha256', None)
            if checksum != hashlib.sha256(canonical(metadata).encode()).hexdigest():
                raise ValueError('checkpoint metadata checksum mismatch')
            if metadata.get('schema') != SCHEMA or metadata.get('manifest') != run._manifest:
                raise ValueError('checkpoint source, selection, alignment or policy mismatch')
            count = metadata.get('n_used')
            if type(count) is not int or not 0 <= count <= len(run._manifest['indices']):
                raise ValueError('invalid completed-frame count')
            gpu_state = None
            history = metadata.get('execution_history')
            if run._manifest['options'].get('device') == 'gpu':
                gpu_state = validate_gpu_state(metadata.get('gpu_state'), history, count)
            elif history != (['cpu'] if count else []):
                raise ValueError('checkpoint execution history mismatch')
            for name in ARRAYS:
                run.model._check_cancel()
                array = data[name]
                template = getattr(run.model, name)
                if array.shape != template.shape or array.dtype != np.float64 or not np.isfinite(array).all():
                    raise ValueError(f'invalid checkpoint array {name}')
                if metadata.get('array_sha256', {}).get(name) != array_digest(array):
                    raise ValueError(f'checkpoint checksum mismatch for {name}')
                if name in ('weights', 'variance_sum') and (array < 0).any():
                    raise ValueError('negative checkpoint support')
                if count == 0 and array.any():
                    raise ValueError('empty checkpoint has contributions')
                # NPZ reads already produce independent arrays; keep them staged
                # without allocating a second copy of the full checkpoint.
                staged[name] = array
        run.model._check_cancel()
        for name, array in staged.items():
            setattr(run.model, name, array)
        run.model.n_used = count
        if gpu_state is not None:
            for key, value in gpu_state.items():
                setattr(run.model, key, value)
            run.model.execution_history = history.copy()
            run.model.expected_device_identity = gpu_state['device_identity']
        return run
