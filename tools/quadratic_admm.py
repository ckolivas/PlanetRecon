"""Experimental Fourier ADMM for the unchanged positivity/support quadratic.

The returned feasible image has a computable distance bound to the unique
optimum. This avoids thousands of nested Dykstra projections per outer solve.
The production estimator is deliberately not replaced by this audit candidate.
"""
import numpy as np

from planetrecon import constants as C
from planetrecon.estimators import constraint_diagnostics, wiener_num_den


def certificate(image, multiplier, num, den, mask):
    """Feasible primal/dual pair and strong-convexity distance bound.

    On spectral subspace S: f(x)=x'Ax/2-b'x+c, y<=0 is dual feasible.
    f(x)-g(y) = r'A_S^-1 r/2 - x'y, r=P_S(Ax-b+y).
    Thus ||x-x*|| <= sqrt(2 gap / lambda_min(A_S)).
    """
    feasible = image + max(0., -float(image.min()))
    dual = np.minimum(multiplier, 0.)
    residual_f = mask*(den*np.fft.fft2(feasible)-num+np.fft.fft2(dual))
    gap = float(.5*np.sum(abs(residual_f[mask])**2/den[mask])/image.size - np.sum(feasible*dual))
    norm = max(float(np.linalg.norm(feasible)), 1e-12)
    bound = float(np.sqrt(max(0., 2*gap)/den[mask].min())/norm)
    stationarity = float(np.linalg.norm(residual_f)/np.sqrt(image.size)/(den[mask].max()*norm))
    return feasible, {'primal_dual_gap': gap, 'relative_solution_error_bound': bound,
                      'dual_stationarity_residual': stationarity,
                      'dc_feasibility_shift': max(0., -float(image.min()))}


def solve(otfs, images, sigma2, lam_f, support, maxiter=10000, tol=C.E2A_FISTA_TOL,
          x0=None, *, error_bound_tol=1e-4):
    if maxiter < 1 or not np.isfinite(tol) or tol <= 0 or not 0 < error_bound_tol < 1:
        raise ValueError('positive finite budgets and tolerances required')
    mask = np.asarray(support) > .5
    if not mask[0, 0]:
        raise ValueError('ADMM audit requires DC in the spectral support')
    num, den = wiener_num_den(otfs, images, sigma2, lam_f)
    # Fractional registration on even grids need not preserve Hermitian
    # transfer coefficients at Nyquist. Restrict the quadratic to real images
    # before inverting its diagonal; taking .real after division is not the
    # same solve. This is also the real normal operator used by production
    # FISTA, whose gradient is the real part of the inverse FFT.
    iy = (-np.arange(num.shape[0])) % num.shape[0]
    ix = (-np.arange(num.shape[1])) % num.shape[1]
    reflected = np.ix_(iy, ix)
    num = .5*(num+num[reflected].conj())
    den = .5*(den+den[reflected])
    rho = float(np.sqrt(den[mask].min()*den[mask].max()))
    lipschitz = float(den[mask].max())
    x = np.fft.ifft2(mask*num/den).real if x0 is None else np.asarray(x0, dtype=float).copy()
    z = np.maximum(x, 0.)
    u = np.zeros_like(z)
    rel_delta = float('inf')
    for iteration in range(1, maxiter+1):
        previous = z
        x = np.fft.ifft2(mask*(num+rho*np.fft.fft2(z-u))/(den+rho)).real
        z = np.maximum(x+u, 0.)
        u += x-z
        primal = float(np.linalg.norm(x-z))
        dual = float(rho*np.linalg.norm(z-previous))
        norm = max(float(np.linalg.norm(x)), 1e-12)
        rel_delta = float(np.linalg.norm(z-previous)/norm)
        if iteration % 10 == 0 or iteration == maxiter:
            feasible, info = certificate(x, rho*u, num, den, mask)
            constraints = constraint_diagnostics(feasible, support)
            if (info['relative_solution_error_bound'] <= error_bound_tol
                    and info['dual_stationarity_residual'] <= tol and constraints['feasible']):
                break
        if iteration % 25 == 0:
            old_rho = rho
            if primal > 10*dual/lipschitz:
                rho *= 2
            elif dual/lipschitz > 10*primal:
                rho /= 2
            u *= old_rho/rho
    converged = bool(info['relative_solution_error_bound'] <= error_bound_tol
                     and info['dual_stationarity_residual'] <= tol and constraints['feasible'])
    return feasible, {**info, **constraints, 'converged': converged, 'n_iter': iteration,
                      'maxiter': maxiter, 'rel_delta': rel_delta, 'rho': rho,
                      'termination_reason': 'converged' if converged else 'iteration_limit',
                      'relative_solution_error_bound_tolerance': error_bound_tol,
                      'stationarity_tolerance': tol, 'solver': 'experimental_quadratic_admm_v2'}
