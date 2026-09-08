"""Experimental gradient-projection/CG phases for the nonnegative scene quadratic.

Projected gradient steps identify a face before the reduced Newton solve. This
candidate is audit-only; the unchanged fresh global-ridge certificate decides
success. See docs/scene-gradient-projection.md for scope and algorithm sources.
"""
import time
import numpy as np
from tools.scene_conditioning import cone_residual
from tools.scene_projected_newton import direction


class _Budget(Exception):
    pass


def solve(problem, *, max_products=750, tolerance=1e-5, x0=None, inner_steps=32,
          projection_steps=8, deadline=None, callback=None, preconditioner=None,
          inner_callback=None):
    for value in (max_products, inner_steps, projection_steps):
        if not np.isfinite(value) or int(value) != value or value < 1:
            raise ValueError('positive integer budgets required')
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('positive finite tolerance required')
    x = np.zeros(problem.shape) if x0 is None else np.array(x0,dtype=float,copy=True)
    if x.shape != problem.shape or not np.isfinite(x).all() or np.any(x < 0):
        raise ValueError('finite feasible initialization required')
    diagonal = np.asarray(problem.majorizer)
    if diagonal.shape != x.shape or not np.isfinite(diagonal).all() or np.any(diagonal <= 0):
        raise ValueError('finite positive step majorizer required')
    started = time.monotonic(); products = 0; trace = []; inner_trace = []; cycle = 0
    gradient = problem.gradient(x); reason = 'product_budget'

    def normal(p):
        nonlocal products
        if deadline is not None and time.monotonic() >= deadline: raise _Budget('wall_budget')
        if products >= max_products: raise _Budget('product_budget')
        hp = np.asarray(problem.normal(p)); products += 1
        if hp.shape != x.shape or not np.isfinite(hp).all(): raise ValueError('invalid Hessian product')
        return hp

    def certified():
        cert = problem.certificate(x,gradient=gradient)
        return cert['feasible'] and cert['relative_solution_error_bound'] <= tolerance

    def search(d, alpha, hd=None):
        for trial in range(12):
            scale = alpha*2.**-trial
            raw = x+scale*d; candidate = np.maximum(raw,0.); step = candidate-x
            slope = float(np.vdot(gradient,step))
            if slope >= 0: continue
            # Reuse an existing direction product only when the actual step is
            # exactly that scaled direction. Subtraction/clipping can alter it.
            hs = scale*hd if hd is not None and np.array_equal(step,scale*d) else normal(step)
            change = slope+.5*float(np.vdot(step,hs))
            if np.isfinite(change) and change <= 1e-4*slope:
                return candidate,hs,change,scale
        return None

    def accept(result, kind, begin_products, inner=0, rejected=None):
        nonlocal x,gradient,inner_trace
        candidate,hs,change,scale = result
        old_active = x == 0
        x = candidate; gradient = gradient+hs
        active_changes = int(np.count_nonzero(old_active != (x == 0)))
        row = {'iteration':len(trace)+1,'cycle':cycle,'hessian_products':products,
               'step_products':products-begin_products,'step_kind':kind,'step_scale':scale,
               'quadratic_objective_change':change,'active_set_changes':active_changes,
               'active_fraction':float(np.mean(x == 0)),'inner_products':inner,
               'inner_trace':inner_trace if kind=='face_newton' else [],
               'relative_solution_error_bound':problem.certificate(x,gradient=gradient)['relative_solution_error_bound'],
               'elapsed_s':time.monotonic()-started}
        if rejected is not None: row['rejected_inner_trace'] = rejected
        trace.append(row)
        if callback is not None: callback(dict(row),x.copy())
        inner_trace = []
        return active_changes

    def safeguard():
        candidate = np.maximum(x-gradient/diagonal,0.)
        step = candidate-x; slope = float(np.vdot(gradient,step))
        if slope >= 0: return None
        hs = normal(step); change = slope+.5*float(np.vdot(step,hs))
        if not np.isfinite(change) or change > 1e-4*slope:
            raise ValueError('projected majorizer safeguard failed descent')
        return candidate,hs,change,1.

    try:
        while not certified():
            cycle += 1
            if cycle > 1:
                # Every cycle refresh is counted, avoiding accumulated recursive
                # gradient error when an inner solve is very accurate.
                gradient = normal(x)-problem.linear
                if certified(): break
            best_decrease = 0.
            for _ in range(projection_steps):
                begin = products
                d = -cone_residual(x,gradient)/diagonal
                slope = float(np.vdot(gradient,d))
                if slope >= 0: raise _Budget('stalled')
                hd = normal(d); curvature = float(np.vdot(d,hd))
                if not np.isfinite(curvature) or curvature <= 0:
                    raise ValueError('invalid projected-gradient curvature')
                result = search(d,-slope/curvature,hd)
                if result is None: result = safeguard()
                if result is None: raise _Budget('stalled')
                changes = accept(result,'gradient_projection',begin)
                decrease = -result[2]
                if certified(): break
                if changes == 0 or decrease <= .1*best_decrease: break
                best_decrease = max(best_decrease,decrease)
            if certified(): break
            # Search the face found by projection. Bound coordinates can be
            # released by the next projection phase, not by this face solve.
            free = x > 0
            if not np.any(free): continue
            begin = products; inner_trace = []
            def progress(row):
                row = row | {'cycle':cycle,'hessian_products':products,'elapsed_s':time.monotonic()-started}
                inner_trace.append(row)
                if inner_callback is not None: inner_callback(dict(row))
            d,inner = direction(normal,gradient,free,diagonal,steps=inner_steps,
                                precondition=preconditioner,callback=progress)
            result = search(d,1.)
            if result is not None:
                accept(result,'face_newton',begin,inner)
            else:
                # Retain failed inner work in the following accepted record.
                rejected = list(inner_trace)
                result = safeguard()
                if result is None: raise _Budget('stalled')
                accept(result,'projected_majorizer',begin,inner,rejected)
    except _Budget as exc:
        reason = str(exc)
    certificate = problem.certificate(x)
    converged = certificate['feasible'] and certificate['relative_solution_error_bound'] <= tolerance
    if converged: reason = 'certified'
    elif certified(): reason = 'fresh_gradient_disagrees'
    return x, {'solver':'extended_scene_gradient_projection_cg_v1',
               'status':'valid' if converged else 'incomplete','converged':bool(converged),
               'reason':reason,'n_iter':len(trace),'cycles':cycle,'hessian_products':products,
               'max_products':max_products,'inner_steps':inner_steps,'projection_steps':projection_steps,
               'preconditioning':'diagonal_majorizer' if preconditioner is None else 'supplied_positive_inverse',
               'initial_final_gradient_evaluations':2,'tolerance':tolerance,'objective':problem.objective(x),
               'trace':trace,'unaccepted_inner_trace':inner_trace,'exact_iteration_resume':False,
               'wall_s':time.monotonic()-started,**certificate}
