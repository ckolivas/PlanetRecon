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
    def __init__(self, operators, images, variances, *, ridge, smoothness=0., prior=None, batch=None):
        self.operators = tuple(operators)
        if not self.operators or len(images) != len(self.operators) or len(variances) != len(self.operators):
            raise ValueError('one image and variance per operator required')
        self.batch = batch
        if batch is not None and (len(batch.operators) != len(self.operators) or any(a is not b for a,b in zip(batch.operators,self.operators))):
            raise ValueError("batch must bind exactly the supplied operators")
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
        ones = np.ones(self.shape)
        self.majorizer = np.full(self.shape, self.ridge)
        degree = np.zeros(self.shape)
        for axis in (0, 1):
            lo, hi = [slice(None)]*len(self.shape), [slice(None)]*len(self.shape)
            lo[axis], hi[axis] = slice(None, -1), slice(1, None)
            degree[tuple(lo)] += 1
            degree[tuple(hi)] += 1
        self.majorizer += 2*self.smoothness*degree
        if self.batch is not None:
            self.linear += self.batch.adjoint([w*y for w,y in zip(self.weights,self.images)])
            self.majorizer += self.batch.normal(ones, self.weights)
        else:
            for op, image, weight in zip(self.operators, self.images, self.weights):
                self.linear += op.adjoint(weight*image)
                self.majorizer += op.adjoint(weight*op.forward(ones))
        # For nonnegative data H, diag(H 1)-H is a graph Laplacian.
        # For D'D smoothness, 2 diag(degree)-D'D is signless PSD.
        # Hence this diagonal majorizes the complete Hessian. Tiny inflation
        # covers roundoff in zero-valued FFT tails; it is not a fitted parameter.
        self.majorizer = np.maximum(self.majorizer, self.ridge)
        self.majorizer += 1e-12*max(float(self.majorizer.max()), self.ridge)
        self.lipschitz = float(self.majorizer.max())

    def normal(self, x):
        out = self.ridge*x+self.smoothness*laplacian_cells(x)
        if self.batch is not None:
            out += self.batch.normal(x, self.weights)
        else:
            for op, weight in zip(self.operators, self.weights):
                out += op.adjoint(weight*op.forward(x))
        return out

    def gradient(self, x):
        return self.normal(x)-self.linear

    def objective(self, x):
        value = self.ridge*np.sum((x-self.prior)**2)
        value += self.smoothness*sum(np.sum(np.diff(x, axis=a)**2) for a in (0, 1))
        predictions = self.batch.forward(x) if self.batch is not None else (op.forward(x) for op in self.operators)
        for predicted, image, weight in zip(predictions, self.images, self.weights):
            value += np.sum(weight*(predicted-image)**2)
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


def solve(problem, *, maxiter=2000, tolerance=1e-5, x0=None, callback=None, deadline=None,
          scaling="diagonal", iteration_checkpoint=None, checkpoint_interval=50):
    if int(maxiter) != maxiter or maxiter < 1 or not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('positive iteration budget and tolerance required')
    if scaling not in ("diagonal", "global"):
        raise ValueError("scaling must be diagonal or global")
    if int(checkpoint_interval)!=checkpoint_interval or checkpoint_interval<1:
        raise ValueError('positive checkpoint interval required')
    solver_name='extended_scene_projected_acceleration_v4'
    checkpoint=None;restored=None
    if iteration_checkpoint is not None:
        from tools.scene_iteration_state import IterationCheckpoint
        # The caller binds input, operator, protocol and runtime identities.
        # Bind the solver contract here as well: larger-budget controls must not
        # accidentally continue the smaller-budget fit instead of starting at zero.
        checkpoint=IterationCheckpoint(iteration_checkpoint.path,{
            'context':iteration_checkpoint.identity,'solver':solver_name,
            'shape':list(problem.shape),'maxiter':int(maxiter),'tolerance':tolerance,'scaling':scaling})
        restored=checkpoint.load()
    if restored is not None:
        if x0 is not None or restored['x'].shape!=problem.shape or restored['iteration']>maxiter:
            raise ValueError('incompatible initialization or iteration checkpoint')
        x,y,n0,past=restored['x'],restored['y'],restored['iteration'],restored['elapsed_s']
    else:
        x = np.zeros(problem.shape) if x0 is None else np.array(x0, dtype=float, copy=True)
        if x.shape != problem.shape or not np.isfinite(x).all():
            raise ValueError('invalid initialization')
        x = np.maximum(x, 0.)
        y=x.copy();n0=0;past=0.
    state=(x,y,n0)
    diagonal = problem.majorizer if scaling == "diagonal" else np.full(problem.shape, problem.lipschitz)
    mu = problem.ridge/float(diagonal.max())
    beta = (1-np.sqrt(mu))/(1+np.sqrt(mu))
    start = time.monotonic()
    reason='iteration_budget'
    def save_state():
        if checkpoint is not None:
            sx,sy,sn=state
            checkpoint.save(sn,sx,sy,past+time.monotonic()-start)
    certificate=problem.certificate(x)
    try:
        if certificate['relative_solution_error_bound'] > tolerance:
            for iteration in range(n0+1,int(maxiter)+1):
                if deadline is not None and time.monotonic()>=deadline:
                    reason='wall_budget';break
                new=np.maximum(y-problem.gradient(y)/diagonal,0.)
                restart=np.vdot(diagonal*(y-new),new-x).real>0
                new_y=new if restart else new+beta*(new-x)
                # Commit one coherent iterate/momentum/iteration tuple. A cancel
                # during the next gradient cannot publish a half-updated state.
                state=(new,new_y,iteration)
                x,y,n=state
                if n%checkpoint_interval==0:save_state()
                if n%10==0 or n==maxiter:
                    certificate=problem.certificate(x)
                    if callback is not None:callback(n,x.copy(),dict(certificate))
                    if certificate['feasible'] and certificate['relative_solution_error_bound']<=tolerance:
                        break
    except BaseException:
        save_state()
        raise
    x,y,n=state
    certificate=problem.certificate(x)
    converged=certificate['feasible'] and certificate['relative_solution_error_bound']<=tolerance
    if converged:reason='certified'
    save_state()
    info = {'solver':solver_name,'converged':bool(converged),
            'status':'valid' if converged else 'incomplete','reason':reason,
            'scaling':scaling,'majorizer_min':float(diagonal.min()),'majorizer_max':float(diagonal.max()),
            'n_iter':n,'resume_from_iteration':n0,'maxiter':int(maxiter),'tolerance':tolerance,
            'objective':problem.objective(x),'wall_s':time.monotonic()-start,
            'cumulative_solver_wall_s':past+time.monotonic()-start,**certificate}
    return x,info
