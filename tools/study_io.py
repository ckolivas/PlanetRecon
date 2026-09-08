"""Explicit runtime/code identities and atomic study records."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
from tools.experiment_stages import atomic_write

ROOT=Path(__file__).resolve().parents[1]


def file_hash(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def write_json(path,value):
    atomic_write(Path(path),(json.dumps(value,indent=2,allow_nan=False)+'\n').encode())


def identities(paths):
    from planetrecon.provenance import source_hash
    dependencies=set(map(Path,paths))|{Path(__file__),ROOT/'tools/experiment_stages.py',
        ROOT/'tools/input_compatibility.py',ROOT/'results/r10-full-grid-inputs/compatibility.json'}
    versions={}
    for name in ('numpy','scipy','h5py','torch'):
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            if name!='torch':raise
            versions[name]=None
    return {'package_source_hash':source_hash(),
            'dependencies':{str(p.resolve().relative_to(ROOT)):file_hash(p) for p in sorted(dependencies)},
            'runtime':{'python':sys.version,'platform':platform.platform(),
                       **versions}}


def open_study(directory,protocol,*,resume=False):
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=resume)
    path=directory/'protocol.json'
    if resume:
        if not path.exists() or json.loads(path.read_text())!=protocol:
            raise ValueError('study input/protocol/runtime/code identity mismatch')
    else:
        write_json(path,protocol)
    return directory
