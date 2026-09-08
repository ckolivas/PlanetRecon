"""Stable fused residual objective/gradient on the existing exact FFT operator."""
import numpy as np
from tools.scene_study import expand_cells, reduce_cells
from tools.scene_quadratic import laplacian_cells


def weighted_residual(batch, scene, images, weights):
    """Return sum .5*w*(A*x-y)^2 and its adjoint gradient, using one frame pass."""
    batch._check_image(scene)
    if len(images) != len(batch.operators) or len(weights) != len(images):
        raise ValueError('one image and weight per operator required')
    sf = batch._fft(batch._array(scene))
    acc = batch._zeros(tuple(sf.shape), complex=True)
    value = batch._array(0.)
    for i, (op, image, weight) in enumerate(zip(batch.operators, images, weights)):
        if np.shape(image) != op.output_shape or np.shape(weight) != op.output_shape:
            raise ValueError('wrong image or weight shape')
        h = batch._spectrum(i)
        if len(batch.shape) == 3:
            h = h[..., None]
        residual = batch._detector(batch._inverse(sf*h), op)-batch._array(image)
        weighted = residual*batch._array(weight)
        value += .5*(weighted*residual).sum()
        acc += h.conj()*batch._fft(batch._embed(weighted, op))
    gradient = batch._numpy(batch._inverse(acc)[:batch.shape[0], :batch.shape[1]])
    return float(batch._numpy(value)), gradient


def objective_gradient(problem, x):
    """Same objective as SceneQuadratic, avoiding subtraction of large constants."""
    if problem.batch is None:
        value = 0.
        gradient = np.zeros_like(x)
        for op, image, weight in zip(problem.operators, problem.images, problem.weights):
            residual = op.forward(x)-image
            value += .5*float(np.sum(weight*residual**2))
            gradient += op.adjoint(weight*residual)
    else:
        batch = problem.batch
        if hasattr(batch, 'native'):
            value, native_gradient = weighted_residual(batch.native, expand_cells(x, batch.factor),
                                                       problem.images, problem.weights)
            gradient = reduce_cells(native_gradient, batch.factor)
        else:
            value, gradient = weighted_residual(batch, x, problem.images, problem.weights)
    delta = x-problem.prior
    value += .5*problem.ridge*float(np.sum(delta**2))
    value += .5*problem.smoothness*sum(float(np.sum(np.diff(x, axis=a)**2)) for a in (0, 1))
    gradient += problem.ridge*delta+problem.smoothness*laplacian_cells(x)
    return value, gradient
