"""Experimental field constraints fitted to identical measured AP displacements."""
from copy import copy

import numpy as np
from scipy import sparse
import torch

from planetrecon.backends.torch_circular import sample
from planetrecon.pipeline.circular_align import CircularMultiscaleRegistration
from tools.coherent_registration import CoherentRegistration
from tools.local_warp_trace import trace

POLICIES = {
    'baseline': (16., .01, 1e-6),
    'bend_01': (16., .1, 1e-6),
    'bend_1': (16., 1., 1e-6),
    'bend_10': (16., 10., 1e-6),
    'anchor_001': (16., .1, .001),
    'anchor_01': (16., .1, .01),
    'grid32': (32., .1, 1e-6),
    'grid64': (64., .1, 1e-6),
}


class RegularizedAPFit:
    def __init__(self, reference, *, device='cuda:0', policies=None):
        policies = list(POLICIES) if policies is None else policies
        matcher = CircularMultiscaleRegistration(reference)
        self.engine = CoherentRegistration(matcher, device=device, spacing=16., stiffness=.01, patch_average=True)
        self.fitters = {}
        grids = {16.: self.engine.fitter}
        for name in policies:
            spacing, stiffness, ridge = POLICIES[name]
            if spacing not in grids:
                grids[spacing] = CoherentRegistration(matcher, device=device, spacing=spacing,
                    stiffness=.01, patch_average=True).fitter
            fitter = copy(grids[spacing])
            if name != 'baseline':
                identity = sparse.eye(fitter.ny*fitter.nx)
                fitter.penalty = (fitter.penalty-identity*1e-6)*(stiffness/.01)+identity*ridge
            self.fitters[name] = fitter

    def observations(self, proxy, shift):
        engine = self.engine
        stages, records = trace(engine, proxy, shift)
        if not stages:
            return np.empty((0, 2)), np.empty(0)
        origin = torch.tensor(shift, device=engine.device, dtype=torch.float64)[:, None, None]
        parent = origin.expand(2, *engine.shape)
        values, confidence = [], []
        for j, (stage, record, layer) in enumerate(zip(stages, records, reversed(engine.layers))):
            measurement = record['measurements']
            delta = torch.as_tensor(measurement[:, :2], device=engine.device)
            yy = layer['y'][:, None, None]+layer['py']+delta[:, 1, None, None]
            xx = layer['x'][:, None, None]+layer['px']+delta[:, 0, None, None]
            previous = (sample(parent, yy, xx, nearest=True)*engine.kernels[j]).sum((-2, -1))
            values.append((delta.T+previous-origin[:, 0]).T.cpu().numpy())
            confidence.append(measurement[:, 4]*record['accepted'])
            parent = stage
        return np.concatenate(values), np.concatenate(confidence)

    def field(self, name, observations, shift):
        engine = self.engine
        residual, stats = self.fitters[name].fit(*observations)
        # Freeze spatial support at the baseline grid even for coarser fits.
        residual *= engine.support
        residual = torch.as_tensor(residual, device=engine.device)
        ux_y, ux_x = torch.gradient(residual[0])
        uy_y, uy_x = torch.gradient(residual[1])
        determinant = (1+ux_x)*(1+uy_y)-ux_y*uy_x
        valid = bool(torch.isfinite(residual).all() & (determinant.min() >= .25)
                     & (torch.linalg.vector_norm(residual, dim=0).max() <= 6.))
        origin = torch.tensor(shift, device=engine.device, dtype=torch.float64)[:, None, None]
        stats.update(field_guard_accepted=valid, minimum_jacobian=float(determinant.min()),
                     maximum_residual_px=float(torch.linalg.vector_norm(residual, dim=0).max()))
        return origin+residual if valid else origin.expand(2, *engine.shape).clone(), stats
