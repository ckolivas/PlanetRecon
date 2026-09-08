"""Experimental feasible Newton-CG updates for the frozen scene quadratic.

The existing global-ridge certificate decides convergence. No approximate inner
solve, line-search flag or assumed active set can replace that certificate.
"""
import time
import numpy as np


class _Budget(Exception):
    pass


def direction(normal, gradient, free, diagonal, *, steps=32, relative_tolerance=.1):
    """Truncated PCG on a principal Hessian; excluded components stay zero."""
    r = np.where(free, -gradient, 0.)
    d = np.zeros_like(r); z = r/diagonal; p = z.copy()
    rz = float(np.vdot(r, z)); initial = float(np.linalg.norm(r))
    for n in range(steps):
        if np.linalg.norm(r) <= relative_tolerance*initial:
            return d, n
        hp = np.where(free, normal(p), 0.)
        php = float(np.vdot(p, hp))
        if not np.isfinite(php) or php <= 0:
            raise ValueError('nonpositive or invalid reduced Hessian curvature')
        alpha = rz/php; d += alpha*p; r -= alpha*hp
        z = r/diagonal; next_rz = float(np.vdot(r, z))
        p = z+(next_rz/rz)*p; rz = next_rz
    return d, steps


def solve(problem, *, max_products=750, tolerance=1e-5, x0=None,
          inner_steps=32, deadline=None, callback=None):
    if int(max_products) != max_products or max_products < 1 or int(inner_steps) != inner_steps or inner_steps < 1:
        raise ValueError('positive integer product and inner budgets required')
    if not np.isfinite(tolerance) or tolerance <= 0: raise ValueError('positive finite tolerance required')
    x = np.zeros(problem.shape) if x0 is None else np.array(x0, dtype=float, copy=True)
    if x.shape != problem.shape or not np.isfinite(x).all() or np.any(x < 0):
        raise ValueError('finite feasible initialization required')
    diagonal = np.asarray(problem.majorizer)
    if diagonal.shape != x.shape or not np.isfinite(diagonal).all() or np.any(diagonal <= 0):
        raise ValueError('finite positive step majorizer required')
    started = time.monotonic(); products = 0; trace = []; reason = 'product_budget'
    # Initialization and final independent recomputation are recorded separately
    # from the bounded inner/line-search Hessian products.
    gradient = problem.gradient(x)

    def normal(p):
        nonlocal products
        if deadline is not None and time.monotonic() >= deadline: raise _Budget('wall_budget')
        if products >= max_products: raise _Budget('product_budget')
        hp = np.asarray(problem.normal(p))
        products += 1
        if hp.shape != x.shape or not np.isfinite(hp).all(): raise ValueError('invalid Hessian product')
        return hp

    try:
        while True:
            # Refresh before a new update, so deadline termination always leaves
            # the last accepted scene paired with its completed trace record.
            if trace and len(trace) % 10 == 0 and products < max_products:
                gradient = normal(x)-problem.linear
            cert = problem.certificate(x, gradient=gradient)
            if cert['feasible'] and cert['relative_solution_error_bound'] <= tolerance:
                reason = 'certified'; break
            if deadline is not None and time.monotonic() >= deadline: raise _Budget('wall_budget')
            if products >= max_products: raise _Budget('product_budget')
            # Wrong active guesses are released whenever their gradient violates
            # the lower-bound KKT condition. Positive coordinates remain free.
            free = (x > 0) | (gradient < 0)
            d, inner = direction(normal, gradient, free, diagonal, steps=inner_steps)
            accepted = False; evaluations = 0
            for trial in range(12):
                candidate = np.maximum(x+(2.**-trial)*d, 0.)
                step = candidate-x; slope = float(np.vdot(gradient, step))
                if slope >= 0: continue
                hs = normal(step); evaluations += 1
                change = slope+.5*float(np.vdot(step, hs))
                if change <= 1e-4*slope:
                    accepted = True; kind = 'newton'; break
            if not accepted:
                # diag(M) >= H gives a feasible global descent safeguard.
                candidate = np.maximum(x-gradient/diagonal, 0.)
                step = candidate-x; slope = float(np.vdot(gradient, step))
                if slope >= 0: reason = 'stalled'; break
                hs = normal(step); evaluations += 1
                change = slope+.5*float(np.vdot(step, hs))
                if change > 1e-4*slope: raise ValueError('projected majorizer safeguard failed descent')
                kind = 'projected_majorizer'
            x = candidate; gradient = gradient+hs
            cert = problem.certificate(x, gradient=gradient)
            row = {'iteration': len(trace)+1, 'hessian_products': products,
                   'inner_products': inner, 'line_search_products': evaluations,
                   'step_kind': kind, 'quadratic_objective_change': change,
                   'active_fraction': float(np.mean(x == 0)),
                   'relative_solution_error_bound': cert['relative_solution_error_bound'],
                   'elapsed_s': time.monotonic()-started}
            trace.append(row)
            if callback is not None: callback(dict(row), x.copy())
    except _Budget as exc:
        reason = str(exc)
    certificate = problem.certificate(x)
    converged = certificate['feasible'] and certificate['relative_solution_error_bound'] <= tolerance
    if converged: reason = 'certified'
    elif reason == 'certified': reason = 'fresh_gradient_disagrees'
    return x, {'solver': 'extended_scene_projected_newton_cg_v1',
               'status': 'valid' if converged else 'incomplete', 'converged': bool(converged),
               'reason': reason, 'n_iter': len(trace), 'hessian_products': products,
               'max_products': max_products, 'inner_steps': inner_steps,
               'initial_final_gradient_evaluations': 2, 'tolerance': tolerance,
               'objective': problem.objective(x), 'trace': trace,
               'exact_iteration_resume': False, 'wall_s': time.monotonic()-started,
               **certificate}
