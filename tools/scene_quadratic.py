"""Matrix-free nonnegative extended-scene quadratic reference (experimental).

Objective: mean_k .5 ||A_k x-y_k||^2_{1/variance_k}
         + .5 ridge ||x-prior||^2 + .5 smoothness ||D x||^2.
D contains nonperiodic horizontal/vertical adjacent-cell differences, per colour.
The strictly positive ridge supplies a proved strong-convexity lower bound; no
Fourier diagonal inverse or legacy spectral-support certificate is reused.
"""
import time
import numpy as np


def laplacian_cells(x):
    result = np.zeros_like(x)
    for axis in (0, 1):
        lo, hi = [slice(None)]*x.ndim, [slice(None)]*x.ndim
        lo[axis], hi[axis] = slice(None, -1), slice(1, None)
        lo, hi = tuple(lo), tuple(hi)
        diff = x[lo]-x[hi]
        result[lo] += diff
        result[hi] -= diff
    return result


class SceneQuadratic:
    def __init__(self, operators, images, variances, *, ridge, smoothness=0., prior=None):
        self.operators = tuple(operators)
        if not self.operators or len(images) != len(self.operators) or len(variances) != len(self.operators):
            raise ValueError('one image and variance per operator required')
        self.shape = self.operators[0].scene_shape
        if any(op.scene_shape != self.shape for op in self.operators):
            raise ValueError('inconsistent scene domains')
        if not np.isfinite([ridge, smoothness]).all() or ridge <= 0 or smoothness < 0:
            raise ValueError('positive ridge and nonnegative smoothness required')
        self.ridge, self.smoothness = float(ridge), float(smoothness)
        self.prior = np.zeros(self.shape) if prior is None else np.array(prior, dtype=float, copy=True)
        if self.prior.shape != self.shape or not np.isfinite(self.prior).all():
            raise ValueError('invalid scene prior')
        self.images, self.weights = [], []
        for op, image, variance in zip(self.operators, images, variances):
            image = np.asarray(image, dtype=float)
            variance = np.broadcast_to(np.asarray(variance, dtype=float), op.output_shape)
            mask = np.broadcast_to(op.valid_mask[..., None] if len(op.output_shape) == 3 else op.valid_mask, op.output_shape)
            if image.shape != op.output_shape or not np.isfinite(image[mask]).all() or not np.isfinite(variance[mask]).all() or np.any(variance[mask] <= 0):
                raise ValueError('finite observations and positive variances required at valid pixels')
            self.images.append(np.where(mask, image, 0.))
            self.weights.append(np.divide(1., variance, out=np.zeros(op.output_shape), where=mask)/len(self.operators))
        self.linear = self.ridge*self.prior.copy()
        # For nonnegative A, ||A||_2^2 <= ||A||_infinity ||A||_1.
        # Apply to sqrt(W)A for spatial variance and average-frame normalization.
        self.lipschitz = self.ridge+8*self.smoothness
        ones = np.ones(self.shape)
        for op, image, weight in zip(self.operators, self.images, self.weights):
            self.linear += op.adjoint(weight*image)
            root = np.sqrt(weight)
            row_sum = root*op.forward(ones)
            col_sum = op.adjoint(root)
            self.lipschitz += max(0., float(row_sum.max()))*max(0., float(col_sum.max()))

    def normal(self, x):
        out = self.ridge*x+self.smoothness*laplacian_cells(x)
        for op, weight in zip(self.operators, self.weights):
            out += op.adjoint(weight*op.forward(x))
        return out

    def gradient(self, x):
        return self.normal(x)-self.linear

    def objective(self, x):
        value = self.ridge*np.sum((x-self.prior)**2)
        value += self.smoothness*sum(np.sum(np.diff(x, axis=a)**2) for a in (0, 1))
        for op, image, weight in zip(self.operators, self.images, self.weights):
            value += np.sum(weight*(op.forward(x)-image)**2)
        return float(.5*value)

    def certificate(self, x, gradient=None):
        g = self.gradient(x) if gradient is None else gradient
        # g + a feasible normal-cone vector; exact zeros are active constraints.
        r = np.where(x > 0, g, np.minimum(g, 0.))
        stationarity = float(np.linalg.norm(r))
        bound = stationarity/self.ridge
        feasible = bool(np.isfinite(x).all() and np.min(x) >= 0.)
        return {'feasible': feasible, 'kkt_residual_norm': stationarity,
                'absolute_solution_error_bound': bound,
                'relative_solution_error_bound': bound/max(float(np.linalg.norm(x)), 1.),
                'objective_gap_upper_bound': stationarity**2/(2*self.ridge),
                'strong_convexity_lower_bound': self.ridge,
                'gradient_lipschitz_upper_bound': self.lipschitz}


def solve(problem, *, maxiter=2000, tolerance=1e-5, x0=None, callback=None, deadline=None):
    if int(maxiter) != maxiter or maxiter < 1 or not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('positive iteration budget and tolerance required')
    x = np.zeros(problem.shape) if x0 is None else np.array(x0, dtype=float, copy=True)
    if x.shape != problem.shape or not np.isfinite(x).all():
        raise ValueError('invalid initialization')
    x = np.maximum(x, 0.)
    y = x.copy()
    L, mu = problem.lipschitz, problem.ridge
    beta = (np.sqrt(L)-np.sqrt(mu))/(np.sqrt(L)+np.sqrt(mu))
    start = time.monotonic()
    reason, converged = 'iteration_budget', False
    certificate = problem.certificate(x)
    n = 0
    if certificate['relative_solution_error_bound'] <= tolerance:
        reason, converged = 'certified', True
    else:
        for n in range(1, int(maxiter)+1):
            if deadline is not None and time.monotonic() >= deadline:
                reason = 'wall_budget'
                n -= 1
                break
            new = np.maximum(y-problem.gradient(y)/L, 0.)
            # Gradient restart prevents momentum repeatedly crossing an active face.
            restart = np.vdot(y-new, new-x).real > 0
            y = new if restart else new+beta*(new-x)
            x = new
            if n % 10 == 0 or n == maxiter:
                certificate = problem.certificate(x)
                if callback is not None:
                    callback(n, x.copy(), dict(certificate))
                if certificate['feasible'] and certificate['relative_solution_error_bound'] <= tolerance:
                    reason, converged = 'certified', True
                    break
    certificate = problem.certificate(x)
    info = {'solver': 'extended_scene_projected_acceleration_v1', 'converged': converged,
            'status': 'valid' if converged else 'incomplete', 'reason': reason,
            'n_iter': n, 'maxiter': int(maxiter), 'tolerance': tolerance,
            'objective': problem.objective(x), 'wall_s': time.monotonic()-start, **certificate}
    return x, info
