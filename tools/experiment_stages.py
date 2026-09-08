"""Durable numerical stages; exact identities, no pickle, no implicit retries.

A stage stores a single image and diagnostic dictionary (including array-valued
diagnostics). Callers bind all inputs, protocol and implementation dependencies.
Different cases/initializations use different directories. One writer per store.
"""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def atomic_write(path, data):
    path = Path(path)
    fd, name = tempfile.mkstemp(prefix='.'+path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        if os.name == 'posix':
            fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
    finally:
        Path(name).unlink(missing_ok=True)


class StageStore:
    def __init__(self, directory, identity, *, deadline=None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.identity = json.loads(canonical(identity))
        self.deadline = deadline
        path = self.directory/'identity.json'
        if path.exists():
            if json.loads(path.read_text()) != self.identity:
                raise ValueError('stage identity mismatch: use a new experiment directory')
        else:
            if any(self.directory.iterdir()):
                raise ValueError('stage store lacks identity')
            atomic_write(path, canonical(self.identity).encode())

    def run(self, name, compute):
        if not name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.' for c in name):
            raise ValueError('invalid stage name')
        manifest = self.directory/(name+'.json')
        payload = self.directory/(name+'.npz')
        if manifest.exists():
            record = json.loads(manifest.read_text())
            blob = payload.read_bytes()
            if record['sha256'] != hashlib.sha256(blob).hexdigest():
                raise ValueError(f'corrupt stage: {name}')
            with np.load(payload, allow_pickle=False) as data:
                info = json.loads(str(data['metadata']))
                info.update({k[5:]: data[k].copy() for k in data.files if k.startswith('info_')})
                return data['image'].copy(), info
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise TimeoutError('experiment wall budget exhausted before '+name)
        started = time.monotonic()
        try:
            image, info = compute()
            from planetrecon.evaluate import _to_jsonable
            metadata = {k: _to_jsonable(v) for k, v in info.items() if not isinstance(v, np.ndarray)}
            arrays = {'info_'+k: v for k, v in info.items() if isinstance(v, np.ndarray)}
            from io import BytesIO
            stream = BytesIO()
            np.savez_compressed(stream, image=image, metadata=canonical(metadata), **arrays)
            blob = stream.getvalue()
            atomic_write(payload, blob)
            from planetrecon.physics_audit import peak_rss_bytes
            record = {'sha256': hashlib.sha256(blob).hexdigest(),
                      'wall_s': time.monotonic()-started,
                      'process_peak_rss_bytes': peak_rss_bytes(),
                      'actual_iterations': metadata.get('n_iter'),
                      'solver_converged': metadata.get('converged'),
                      'status': 'completed'}
            # The manifest is the commit point; an orphan NPZ is never reused.
            atomic_write(manifest, canonical(record).encode())
            return image, info
        except BaseException as exc:
            failure = {'stage': name, 'status': 'incomplete', 'error': repr(exc),
                       'wall_s': time.monotonic()-started}
            atomic_write(self.directory/(name+f'.failure-{time.time_ns()}.json'), canonical(failure).encode())
            raise
