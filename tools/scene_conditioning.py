"""Frozen-iterate Hessian probes and independently checkable quadratic gap bounds.

These diagnostics do not update the scene or change the production certificate.
"""
import time
import numpy as np


def cone_residual(x, gradient):
    x, gradient = np.asarray(x, dtype=float), np.asarray(gradient, dtype=float)
    if x.shape != gradient.shape or not np.isfinite(x).all() or not np.isfinite(gradient).all() or np.any(x < 0):
        raise ValueError('finite feasible scene and matching gradient required')
    return np.where(x > 0, gradient, np.minimum(gradient, 0.))


def correction_bound(x, gradient, correction, h_correction, ridge):
    """Bound gap/distance using any correction and an independently computed H*y.

Caller must supply the actual objective gradient/Hessian product and valid ridge
lower bound. The algebra is exact-arithmetic; the guard covers scalar reduction
roundoff, not a formal interval enclosure of FFT/operator error.
"""
    r = cone_residual(x, gradient)
    y, hy = np.asarray(correction, dtype=float), np.asarray(h_correction, dtype=float)
    if y.shape != r.shape or hy.shape != r.shape or not np.isfinite(y).all() or not np.isfinite(hy).all():
        raise ValueError('finite matching correction and Hessian product required')
    if not np.isfinite(ridge) or ridge <= 0: raise ValueError('positive ridge lower bound required')
    e = r-hy
    rr, ry, yhy, ee = (float(np.vdot(a, b).real) for a, b in ((r, r), (r, y), (y, hy), (e, e)))
    raw_energy = rr/ridge
    corrected_energy = 2*ry-yhy+ee/ridge
    eps = np.finfo(float).eps
    gamma = (r.size+4)*eps/(1-(r.size+4)*eps)
    scale = 2*float(np.sum(np.abs(r*y)))+float(np.sum(np.abs(y*hy)))+ee/ridge
    # Also guard the component-wise subtraction in e and its squared norm.
    delta = eps*(float(np.linalg.norm(r))+float(np.linalg.norm(hy)))
    guard = gamma*scale+(2*float(np.linalg.norm(e))*delta+delta**2)/ridge
    if corrected_energy < -guard: raise ValueError('negative energy bound: inconsistent products or numerical failure')
    energy = min(raw_energy*(1+gamma), max(corrected_energy+guard, 0.))
    return {'certificate': 'quadratic_normal_cone_correction_v1_diagnostic',
            'objective_gap_upper_bound': .5*energy,
            'absolute_solution_error_bound': float(np.sqrt(energy/ridge)),
            'relative_solution_error_bound': float(np.sqrt(energy/ridge)/max(np.linalg.norm(x), 1.)),
            'raw_objective_gap_upper_bound': .5*raw_energy,
            'correction_linear_residual_norm': float(np.sqrt(ee)),
            'scalar_roundoff_guard': guard, 'feasible': True}


def probe_cg(normal, rhs, diagonal, *, steps=32, deadline=None, callback=None):
    """Probe H*y=rhs; bounded PCG with trace, never a constrained scene solve."""
    rhs, diagonal = np.asarray(rhs, dtype=float), np.asarray(diagonal, dtype=float)
    if rhs.shape != diagonal.shape or not np.isfinite(rhs).all() or not np.isfinite(diagonal).all() or np.any(diagonal <= 0):
        raise ValueError('finite rhs and positive matching diagonal required')
    if int(steps) != steps or steps < 1: raise ValueError('positive probe length required')
    start = time.monotonic(); y = np.zeros_like(rhs); r = rhs.copy()
    z = r/diagonal; p = z.copy(); rz = float(np.vdot(r, z).real)
    initial = float(np.linalg.norm(rhs)); rows = []; reason = 'probe_budget'
    for n in range(1, int(steps)+1):
        if np.linalg.norm(r) <= 1e-12*max(initial, 1e-30):
            reason = 'linear_residual_small'; break
        if deadline is not None and time.monotonic() >= deadline:
            reason = 'wall_budget'; break
        hp = np.asarray(normal(p), dtype=float)
        if hp.shape != rhs.shape or not np.isfinite(hp).all(): raise ValueError('invalid Hessian product')
        curvature = float(np.vdot(p, hp).real)
        if curvature <= 0 or not np.isfinite(curvature): raise ValueError('nonpositive Hessian curvature')
        alpha = rz/curvature
        y += alpha*p; r -= alpha*hp
        row = {'iteration': n, 'hessian_evaluations': n, 'elapsed_s': time.monotonic()-start,
               'recursive_relative_residual': float(np.linalg.norm(r)/max(initial, 1e-30)),
               'direction_rayleigh': curvature/float(np.vdot(p, p).real),
               'scaled_direction_rayleigh': curvature/float(np.vdot(p, diagonal*p).real)}
        rows.append(row)
        if callback is not None: callback(dict(row), y)
        z = r/diagonal; next_rz = float(np.vdot(r, z).real)
        p = z+(next_rz/rz)*p; rz = next_rz
    return y, {'reason': reason, 'iterations': len(rows), 'trace': rows,
               'wall_s': time.monotonic()-start, 'scene_updated': False}
