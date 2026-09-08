"""Read-only frozen-stage and independent Fourier-energy diagnostics."""
import json
from pathlib import Path
import numpy as np
from tools.study_io import file_hash


def reference_stage(directory, original, selection, expected_hash, *, shape=(1024,912)):
    directory=Path(directory)
    if json.loads((directory/'identity.json').read_text())!={'protocol':original,'selection':selection}:
        raise ValueError('stage objective identity mismatch')
    payload=directory/'budget-750.npz'
    manifest=json.loads((directory/'budget-750.json').read_text())
    if manifest.get('status')!='completed' or manifest.get('sha256')!=expected_hash or file_hash(payload)!=expected_hash:
        raise ValueError('frozen stage checksum mismatch')
    with np.load(payload,allow_pickle=False) as z:
        x=z['image'].copy();info=json.loads(str(z['metadata']))
    expected={'solver':'extended_scene_projected_acceleration_v4','maxiter':750,'n_iter':310,
              'converged':True,'scaling':'diagonal'}
    if any(info.get(k)!=v for k,v in expected.items()): raise ValueError('unexpected reference stage')
    if x.shape!=shape or x.dtype!=np.float64 or not np.isfinite(x).all() or np.any(x<0):
        raise ValueError('invalid frozen stage scene')
    return x


def fourier_energy(problem, ky, kx):
    """Exact diagonal of F* H F using real/imaginary unit Fourier modes.

For scalar scenes only. Evaluate forward weighted energies independently, not
the preconditioner or cancellation of objective constants. No latent truth.
"""
    if len(problem.shape)!=2: raise ValueError('scalar scene required for this diagnostic')
    ny,nx=problem.shape
    if int(ky)!=ky or int(kx)!=kx or not 0<=ky<ny or not 0<=kx<=nx//2:
        raise ValueError('valid integer Fourier frequency required')
    yy,xx=np.indices(problem.shape)
    angle=2*np.pi*(ky*yy/ny+kx*xx/nx)
    components=[np.cos(angle)/np.sqrt(ny*nx)]
    if ky or kx: components.append(np.sin(angle)/np.sqrt(ny*nx))
    value=0.
    for x in components:
        predictions=problem.batch.forward(x) if problem.batch is not None else (op.forward(x) for op in problem.operators)
        value+=sum(float(np.sum(w*y*y)) for w,y in zip(problem.weights,predictions))
        value+=problem.ridge*float(np.sum(x*x))
        value+=problem.smoothness*sum(float(np.sum(np.diff(x,axis=a)**2)) for a in (0,1))
    return value
