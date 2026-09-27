"""Diagnostic AP position/acceptance factorial with baseline parents and IRLS weights."""
import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse.linalg import spsolve

from planetrecon.backends.torch_circular import sample
from tools.fractional_ap_trace import refined_peak, observations


def frozen_parent_observations(engine, proxy, shift, stages, records):
    """Refine measurements on baseline stage inputs; never propagate new stages."""
    image0 = torch.as_tensor(proxy, device=engine.device, dtype=torch.float64)
    origin = torch.as_tensor(shift, device=engine.device, dtype=torch.float64)[:, None, None]
    parent = origin.expand(2, *engine.shape)
    new_records = []
    for layer, stage, record in zip(reversed(engine.layers), stages, records):
        image = sample(image0, engine.yy+parent[1], engine.xx+parent[0])
        observed = ((engine.xx+parent[0] >= 0)&(engine.xx+parent[0] <= engine.shape[1]-1)
                    &(engine.yy+parent[1] >= 0)&(engine.yy+parent[1] <= engine.shape[0]-1))
        integral = F.pad((~observed).long().cumsum(0).cumsum(1), (1, 0, 1, 0))
        y, x = layer['y'], layer['x']
        margin = layer['h']+layer['m']
        supported = (integral[y+margin+1,x+margin+1]-integral[y-margin,x+margin+1]
                     -integral[y+margin+1,x-margin]+integral[y-margin,x-margin]) == 0
        details = []
        for start in range(0, len(y), layer['chunk']):
            section = slice(start, start+layer['chunk'])
            cy, cx = y[section,None,None], x[section,None,None]
            delta, valid, peak = refined_peak(engine, image, layer, section,
                layer['templates'][section], layer['strength'][section], sample, True)
            patch = sample(image, cy+layer['py']+delta[:,1,None,None], cx+layer['px']+delta[:,0,None,None])
            patch -= (patch*layer['weight']).sum((-1,-2))[:,None,None]
            strength = (patch.square()*layer['weight']).sum((-1,-2)).sqrt()
            reverse, reverse_valid, _ = refined_peak(engine, engine.reference, layer, section, patch, strength, sample, True)
            valid &= supported[section] & reverse_valid & (torch.linalg.vector_norm(reverse,dim=1) <= .75)
            confidence = ((peak-.8)/.15).clamp(0.,1.)*valid
            details.append(torch.column_stack((delta,valid,peak,confidence,torch.linalg.vector_norm(reverse,dim=1))))
        new_records.append(dict(measurements=torch.cat(details).cpu().numpy(), accepted=record['accepted']))
        parent = stage
    return observations(engine, stages, new_records, shift)


def solve_fixed(fitter, values, weights):
    if not np.isfinite(values).all() or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError('Invalid fit inputs')
    if np.count_nonzero(weights) < 6:
        return np.zeros((2, *fitter.shape))
    weighted = fitter.design.multiply(weights[:,None])
    lhs = (fitter.design.T@weighted+fitter.penalty).tocsc()
    coefficients = spsolve(lhs, fitter.design.T@(weights[:,None]*values))
    return np.stack([fitter.by@coefficients[:,axis].reshape(fitter.ny,fitter.nx)@fitter.bx.T for axis in range(2)])


def baseline_robust_factors(fitter, values, confidence):
    """Weights used by the third solve, not the unused final IRLS update."""
    factor = np.ones_like(confidence)
    if np.count_nonzero(confidence) < 6:
        return factor
    for _ in range(2):
        weight = confidence*factor
        weighted = fitter.design.multiply(weight[:,None])
        lhs = (fitter.design.T@weighted+fitter.penalty).tocsc()
        coefficients = spsolve(lhs, fitter.design.T@(weight[:,None]*values))
        residual = np.linalg.norm(fitter.design@coefficients-values, axis=1)
        factor = np.minimum(1., fitter.huber/np.maximum(residual,1e-15))
    return factor


def fixed_field(model, values, confidence, factor, shift, baseline_accepted):
    """Force baseline final fallback; record hypothetical candidate guard failures."""
    engine = model.engine
    residual = solve_fixed(model.fitters['baseline'], values, confidence*factor)*engine.support
    residual = torch.as_tensor(residual, device=engine.device)
    if not bool(torch.isfinite(residual).all()):
        raise ValueError('Nonfinite diagnostic field')
    ux_y, ux_x = torch.gradient(residual[0])
    uy_y, uy_x = torch.gradient(residual[1])
    jacobian = (1+ux_x)*(1+uy_y)-ux_y*uy_x
    maximum = torch.linalg.vector_norm(residual,dim=0).max()
    natural = bool((jacobian.min() >= .25)&(maximum <= 6.))
    origin = torch.as_tensor(shift, device=engine.device,dtype=torch.float64)[:,None,None]
    stats = dict(field_guard_accepted=bool(baseline_accepted), natural_guard_accepted=natural,
        forced_guard_disagreement=bool(baseline_accepted) != natural,
        minimum_jacobian=float(jacobian.min()), maximum_residual_px=float(maximum),
        fallback=np.count_nonzero(confidence) < 6, accepted_aps=int(np.count_nonzero(confidence)))
    stats['fallback'] = bool(stats['fallback'])
    return (origin+residual if baseline_accepted else origin.expand(2,*engine.shape).clone()), stats
