"""Bounded W01 operator and constraint audit against dense independent oracles.

Run with python -m planetrecon.constraint_audit --out NEW_DIRECTORY. This is
small-grid numerical qualification, not a scientific-family or Q3 certificate.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import LinearConstraint, minimize

from planetrecon import constants as C
from planetrecon.estimators import e1, e2a, e2a0
from planetrecon.operators import (
    fourier_shift_image, fourier_shift_adjoint, padded_detector_crop,
    padded_detector_crop_adjoint, relative_l2, scalar_noise_mismatch,
)
from planetrecon.optics import otf_from_centered_psf
from planetrecon.provenance import source_hash


def convolution_matrix(shape, kernel, *, circular=False):
    """Direct spatial convolution matrix, independent of production FFT code."""
    y, x = np.indices(shape).reshape(2, -1)
    center = np.array(kernel.shape) // 2 if circular else (np.array(kernel.shape) - 1) // 2
    ky = y[:, None] - y[None, :] + center[0]
    kx = x[:, None] - x[None, :] + center[1]
    if circular:
        if kernel.shape != tuple(shape):
            raise ValueError('circular oracle requires a full-sized centered kernel')
        return kernel[ky % shape[0], kx % shape[1]]
    valid = (ky >= 0) & (ky < kernel.shape[0]) & (kx >= 0) & (kx < kernel.shape[1])
    return np.where(valid, kernel[np.clip(ky, 0, kernel.shape[0]-1),
                                  np.clip(kx, 0, kernel.shape[1]-1)], 0.)


def shift_matrix(n, shift):
    """Real part of the direct Fourier series, including even-grid Nyquist."""
    y, x = np.indices((n, n)).reshape(2, -1)
    freq = np.fft.fftfreq(n)
    fy, fx = np.meshgrid(freq, freq, indexing='ij')
    fy, fx = fy.ravel(), fx.ravel()
    modes = np.exp(2j*np.pi*(y[:, None]*fy + x[:, None]*fx))
    phase = np.exp(-2j*np.pi*(shift[1]*fy + shift[0]*fx))
    return ((modes*phase) @ modes.conj().T).real / n**2


def detector_matrix(shape, kernel, factor, origin, size, flux, shift):
    conv = convolution_matrix(shape, kernel)
    rows = []
    for y in range(origin[1], origin[1]+size):
        for x in range(origin[0], origin[0]+size):
            indices = [yy*shape[1]+xx for yy in range(y*factor, (y+1)*factor)
                       for xx in range(x*factor, (x+1)*factor)]
            rows.append(conv[indices].sum(axis=0)*flux)
    return shift_matrix(size, shift) @ np.array(rows)


def spectral_basis(support):
    """Real orthonormal allowed-frequency basis from sampled sines/cosines."""
    n = support.shape[0]
    y, x = np.indices((n, n)).reshape(2, -1)
    fy, fx = np.meshgrid(np.fft.fftfreq(n), np.fft.fftfreq(n), indexing='ij')
    phase = 2*np.pi*(y[:, None]*fy[support > .5] + x[:, None]*fx[support > .5])
    u, singular, _ = np.linalg.svd(np.concatenate((np.cos(phase), np.sin(phase)), axis=1), full_matrices=False)
    return u[:, singular > 1e-10]


def quadratic_oracle(kernels, images, variances, regularizer, support):
    """Solve in a real spectral basis with SLSQP and linear positivity bounds.

    The data objective and derivatives use dense spatial convolution, with no
    production normal equations, FFT convolutions or Dykstra projections.
    """
    basis = spectral_basis(support)
    matrices = np.array([convolution_matrix(support.shape, p, circular=True) for p in kernels])
    hessian = sum(a.T @ a / v for a, v in zip(matrices, variances))
    hessian += regularizer*np.eye(support.size)
    rhs = sum(a.T @ im.ravel() / v for a, im, v in zip(matrices, images, variances))
    reduced = basis.T @ hessian @ basis
    b = basis.T @ rhs
    fit = minimize(lambda z: .5*z @ reduced @ z - b @ z, np.zeros(b.size),
                   jac=lambda z: reduced @ z - b, method='SLSQP',
                   constraints=LinearConstraint(basis, 0., np.inf),
                   options={'ftol': 1e-12, 'maxiter': 2000})
    result = (basis @ fit.x).reshape(support.shape)
    info = {'success': bool(fit.success), 'message': str(fit.message), 'n_iter': int(fit.nit),
            'positivity_violation': float(max(0., -result.min()))}

    def objective(image):
        v = image.ravel()
        # Include the constant data term to normalize gaps by a positive objective.
        return float(.5*sum(np.sum((a @ v-im.ravel())**2)/var
                            for a, im, var in zip(matrices, images, variances))
                     + .5*regularizer*np.sum(v*v))

    return result, info, objective


def operator_cases():
    rows = []
    rng = np.random.default_rng(1901)
    for shape in ((12, 16), (16, 12)):
        for kernel_shape in ((3, 5), (4, 4)):
            for factor, size, origin in ((2, 4, (1, 1)), (4, 3, (0, 0))):
                for shift in ((0., 0.), (1.25, -.75)):
                    kernel = rng.uniform(.1, 1., size=kernel_shape)
                    kernel /= kernel.sum()
                    flux = 3.5
                    dense = detector_matrix(shape, kernel, factor, origin, size, flux, shift)
                    x, y = rng.normal(size=shape), rng.normal(size=(size, size))
                    actual = fourier_shift_image(padded_detector_crop(x, kernel, origin, size, factor, flux), shift)
                    adjoint = padded_detector_crop_adjoint(fourier_shift_adjoint(y, shift), kernel, origin, shape, factor, flux)
                    expected = (dense @ x.ravel()).reshape(y.shape)
                    expected_adjoint = (dense.T @ y.ravel()).reshape(shape)
                    rows.append({'shape': shape, 'kernel_shape': kernel_shape, 'bin_factor': factor,
                                 'crop_size': size, 'origin': origin, 'shift': shift, 'flux': flux,
                                 'forward_relative_error': relative_l2(actual, expected),
                                 'adjoint_relative_error': relative_l2(adjoint, expected_adjoint)})
    return rows


def run(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    protocol = {'status': 'diagnostic', 'q3_authorized': False, 'seeds': list(C.DEV_SEEDS),
                'image_size': 8, 'support_radii': [.26, 1.], 'regularizers': [.03],
                'budgets': [128, 512, 1024], 'initializations': ['zero', 'positive_pattern'],
                'solver_tolerance': 1e-8, 'oracle_relative_image_tolerance': 1e-4,
                'oracle_relative_objective_gap_tolerance': 1e-8,
                'operator_relative_tolerance': 1e-11,
                'oracle': 'Dense spatial convolution, real trigonometric basis, SLSQP positivity constraints',
                'limits': 'Small known-transfer quadratic only; no TV, atmospheric fit, real-capture qualification or gate authorization.',
                'design_revision': 'v5 retains v4 restart and tightens solver tolerance from 1e-6 to 1e-8 after v4 stopped within its tolerance but one oracle image error exceeded 1e-4; compare 128/512/1024 budgets without loosening any acceptance tolerance.',
                'restart_reference': 'https://arxiv.org/abs/1204.3982'}
    (directory/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    started = time.monotonic()
    operators = operator_cases()
    n = protocol['image_size']
    y, x = np.indices((n, n))
    kernels = []
    for width, dx, dy in ((.8, .2, -.3), (1.3, -.4, .1)):
        kernel = np.exp(-((x-n//2-dx)**2+(y-n//2-dy)**2)/(2*width**2))
        kernels.append(kernel/kernel.sum())
    kernels = np.array(kernels)
    otfs = np.array([otf_from_centered_psf(p) for p in kernels])
    matrices = np.array([convolution_matrix((n, n), p, circular=True) for p in kernels])
    variances = np.array([.04, .09])
    cases = []
    for seed in protocol['seeds']:
        rng = np.random.default_rng(seed)
        truth = np.exp(-((x-3.1)**2+(y-4.2)**2)/2.)*3
        images = (matrices @ truth.ravel()).reshape(2, n, n)
        images += rng.normal(size=images.shape)*np.sqrt(variances)[:, None, None]
        for radius in protocol['support_radii']:
            freq = np.fft.fftfreq(n)
            support = (np.hypot(freq[:, None], freq[None, :]) <= radius).astype(float)
            for lam in protocol['regularizers']:
                oracle, oracle_info, objective = quadratic_oracle(kernels, images, variances, lam, support)
                value = objective(oracle)
                record = {'seed': seed, 'support_radius': radius, 'regularizer': lam,
                          'oracle': oracle_info, 'oracle_objective': value, 'solves': []}
                closed = e1(otfs, images, variances, np.full((n, n), lam))
                cg, cg_info = e2a0(otfs, images, variances, np.full((n, n), lam))
                record['unconstrained'] = {**cg_info, 'E1_relative_error': relative_l2(cg, closed)}
                for init in protocol['initializations']:
                    for budget in protocol['budgets']:
                        start = np.zeros((n, n)) if init == 'zero' else images.mean()*(1.+.5*np.cos(2*np.pi*x/n))
                        fitted, info = e2a(otfs, images, variances, np.full((n, n), lam), support,
                                          maxiter=budget, tol=protocol['solver_tolerance'], x0=start)
                        error = relative_l2(fitted, oracle)
                        gap = (objective(fitted)-value)/max(abs(value), 1e-12)
                        passed = (oracle_info['success'] and oracle_info['positivity_violation'] <= C.FEASIBLE_POS_TOL
                                  and info['converged'] and info['feasible']
                                  and error <= protocol['oracle_relative_image_tolerance']
                                  and abs(gap) <= protocol['oracle_relative_objective_gap_tolerance'])
                        record['solves'].append({'initialization': init, 'budget': budget, 'solver': info,
                                                 'relative_image_error': error, 'relative_objective_gap': gap,
                                                 'bounded_check_passed': bool(passed)})
                cases.append(record)
                (directory/f'case-{seed}-{radius}.json').write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
                print(f'seed={seed} radius={radius} lambda={lam}: '+
                      ', '.join(f"{s['initialization']}/{s['budget']}:{s['bounded_check_passed']}" for s in record['solves']), flush=True)
    report = {'status': 'diagnostic', 'q3_authorized': False, 'source_hash': source_hash(),
              'estimator_operator_version': C.ESTIMATOR_OPERATOR_VERSION, 'operators': operators,
              'operator_checks_passed': all(max(r['forward_relative_error'], r['adjoint_relative_error'])
                                            < protocol['operator_relative_tolerance'] for r in operators),
              'cases': cases, 'wall_s': time.monotonic()-started}
    # A controlled brightness map measures the scalar-variance approximation;
    # it does not silently substitute a different likelihood into the estimator.
    report['noise_controls'] = {
        'constant': scalar_noise_mismatch(np.full((n, n), 50.), read_rms=2.),
        'structured': scalar_noise_mismatch(50.+400.*truth/3., read_rms=2.)}
    (directory/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


if __name__ == '__main__':
    from planetrecon.runtime import apply_thread_limits
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    apply_thread_limits(2)
    run(args.out)
