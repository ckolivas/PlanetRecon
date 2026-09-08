"""One experimental bounded quasi-Newton alternative; certificate decides success.

SciPy's function/gradient stopping flags never confer numerical qualification.
Internal L-BFGS history is not serialized: no exact iteration-resume claim.
"""
import time
import numpy as np
from scipy.optimize import minimize, Bounds
from tools.scene_residual import objective_gradient


class _Stop(Exception):
    pass


def solve(problem, *, maxiter=750, tolerance=1e-5, x0=None, deadline=None, callback=None):
    if int(maxiter) != maxiter or maxiter < 1 or not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('positive iteration budget and tolerance required')
    x = np.zeros(problem.shape) if x0 is None else np.array(x0, dtype=float, copy=True)
    if x.shape != problem.shape or not np.isfinite(x).all():
        raise ValueError('invalid initialization')
    x = np.maximum(x, 0.)
    scale = np.sqrt(problem.majorizer)
    start = time.monotonic()
    value, gradient = objective_gradient(problem, x)
    state = {'x': x, 'value': value, 'gradient': gradient, 'n': 0}
    cache = {}; evaluations = 1; message = None
    reason = 'optimizer_stop'

    def fun(z):
        nonlocal evaluations, cache
        if deadline is not None and time.monotonic() >= deadline:
            raise _Stop('wall_budget')
        candidate = z.reshape(problem.shape)/scale
        f, g = objective_gradient(problem, candidate)
        evaluations += 1
        cache = {'z': z.copy(), 'x': candidate, 'value': f, 'gradient': g}
        return f, (g/scale).ravel()

    def accepted(z):
        nonlocal state
        if not cache or not np.array_equal(z, cache['z']):
            fun(z)
        state = {k: cache[k] for k in ('x', 'value', 'gradient')} | {'n': state['n']+1}
        cert = problem.certificate(state['x'], gradient=state['gradient'])
        if callback is not None:
            callback(state['n'], state['x'].copy(), dict(cert))
        if cert['feasible'] and cert['relative_solution_error_bound'] <= tolerance:
            raise _Stop('certified')

    initial = problem.certificate(x, gradient=gradient)
    try:
        if initial['feasible'] and initial['relative_solution_error_bound'] <= tolerance:
            reason = 'certified'
        else:
            result = minimize(fun, (x*scale).ravel(), jac=True, method='L-BFGS-B',
                              bounds=Bounds(0., np.inf), callback=accepted,
                              options={'maxiter': int(maxiter), 'maxcor': 8, 'maxls': 20,
                                       'maxfun': 20*(int(maxiter)+1), 'ftol': 0., 'gtol': 0.})
            message = str(result.message)
            reason = 'iteration_budget' if state['n'] >= maxiter else 'optimizer_stop'
    except _Stop as exc:
        reason = str(exc)
    x = state['x']
    # Re-evaluate through the authoritative normal-gradient implementation.
    certificate = problem.certificate(x)
    converged = certificate['feasible'] and certificate['relative_solution_error_bound'] <= tolerance
    if converged:
        reason = 'certified'
    elif reason == 'certified':
        reason = 'independent_gradient_disagrees'
    return x, {'solver': 'extended_scene_scaled_lbfgsb_v1', 'converged': bool(converged),
               'status': 'valid' if converged else 'incomplete', 'reason': reason,
               'n_iter': state['n'], 'maxiter': int(maxiter), 'tolerance': tolerance,
               'function_gradient_evaluations': evaluations, 'objective': problem.objective(x),
               'optimizer_message': message, 'exact_iteration_resume': False,
               'wall_s': time.monotonic()-start, **certificate}
