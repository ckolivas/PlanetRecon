"""Bounded full/reduced PCG diagnostics with a callable positive inverse."""
import time
import numpy as np


def probe(normal, rhs, precondition, *, free=None, steps=32, deadline=None, callback=None):
    rhs=np.asarray(rhs,dtype=float)
    free=np.ones(rhs.shape,dtype=bool) if free is None else np.asarray(free,dtype=bool)
    if free.shape != rhs.shape or not np.isfinite(rhs).all() or np.any(rhs[~free] != 0):
        raise ValueError('finite residual supported on the free set required')
    if int(steps) != steps or steps < 1: raise ValueError('positive integer product cap required')
    def apply(r):
        z=np.asarray(precondition(r),dtype=float)
        if z.shape != r.shape or not np.isfinite(z).all(): raise ValueError('invalid inverse product')
        return np.where(free,z,0.)
    x=np.zeros_like(rhs); r=rhs.copy(); initial=float(np.linalg.norm(r))
    trace=[]; started=time.monotonic(); reason='product_budget'
    if initial == 0:
        return x,{'reason':'zero_residual','iterations':0,'trace':trace,'wall_s':0.}
    z=apply(r); p=z.copy(); rz=float(np.vdot(r,z))
    for n in range(1,int(steps)+1):
        if deadline is not None and time.monotonic() >= deadline:
            reason='wall_budget';break
        if not np.isfinite(rz) or rz <= 0: raise ValueError('inverse is not positive on residual')
        raw=np.asarray(normal(p),dtype=float)
        if raw.shape != rhs.shape or not np.isfinite(raw).all(): raise ValueError('invalid Hessian product')
        hp=np.where(free,raw,0.)
        curvature=float(np.vdot(p,hp))
        if not np.isfinite(curvature) or curvature <= 0: raise ValueError('nonpositive Hessian curvature')
        alpha=rz/curvature;x+=alpha*p;r-=alpha*hp
        ratio=float(np.linalg.norm(r)/initial)
        row={'iteration':n,'hessian_products':n,'elapsed_s':time.monotonic()-started,
             'recursive_relative_residual':ratio,
             'direction_rayleigh':curvature/float(np.vdot(p,p))}
        trace.append(row)
        if callback is not None: callback(dict(row),x)
        if ratio <= 1e-12:
            reason='linear_residual_small';break
        z=apply(r); next_rz=float(np.vdot(r,z));p=z+(next_rz/rz)*p;rz=next_rz
    return x,{'reason':reason,'iterations':len(trace),'trace':trace,'wall_s':time.monotonic()-started}
