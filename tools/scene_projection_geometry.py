"""Exploratory geometry of a proposed step; no objective or convergence claim."""
import numpy as np


def projection_geometry(x, direction):
    x, direction = np.asarray(x, dtype=float), np.asarray(direction, dtype=float)
    if x.shape != direction.shape or not np.isfinite(x).all() or not np.isfinite(direction).all() or np.any(x < 0):
        raise ValueError('finite feasible scene and matching direction required')
    candidate = x+direction
    if not np.isfinite(candidate).all():
        raise ValueError('nonfinite candidate')
    step = np.maximum(candidate, 0.)-x
    norm = float(np.linalg.norm(direction))
    return {'crossing_fraction': float(np.mean(candidate < 0)),
            'newly_bound_fraction': float(np.mean((x > 0) & (candidate <= 0))),
            'released_fraction': float(np.mean((x == 0) & (candidate > 0))),
            'removed_direction_relative_norm': float(np.linalg.norm(step-direction)/norm) if norm else 0.,
            'projected_step_relative_norm': float(np.linalg.norm(step)/norm) if norm else 0.}
