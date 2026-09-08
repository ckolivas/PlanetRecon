"""Algebraic interpretation of a fixed-residual quadratic gap diagnostic."""
import math


def fixed_residual_limit(initial, tolerance=1e-5):
    """Necessary lower limit of this certificate family, NOT true scene error.

H <= L I implies r'H^-1r >= ||r||²/L. Combining that energy with the
global ridge distance inequality cannot yield less than ||r||/sqrt(ridge*L).
"""
    ridge, upper = initial['ridge'], initial['majorizer_max']
    raw = initial['raw_certificate']['relative_solution_error_bound']
    if not all(math.isfinite(v) for v in (ridge, upper, raw, tolerance)) or not 0 < ridge <= upper or raw < 0 or tolerance <= 0:
        raise ValueError('finite valid Hessian bounds, certificate and tolerance required')
    maximum_improvement = math.sqrt(upper/ridge)
    floor = raw/maximum_improvement
    return {'fixed_residual_gap_certificate_floor': floor,
            'maximum_possible_improvement_factor': maximum_improvement,
            'threshold_ruled_out_for_fixed_residual_family': floor > tolerance,
            'original_tolerance': tolerance,
            'scope': 'Lower limit on this fixed-normal-cone energy/global-ridge certificate; not a lower bound on actual scene error or on other certificate families.'}
