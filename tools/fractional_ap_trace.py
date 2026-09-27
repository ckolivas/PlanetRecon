"""Experimental matching-only sampler and direct fractional NCC refinement."""
import numpy as np
import torch
import torch.nn.functional as F

from planetrecon.backends.torch_circular import sample, peaks


def cubic_sample(image, y, x):
    """Separable Catmull-Rom interpolation, with zero exterior samples."""
    h, w = image.shape[-2:]
    iy, ix = torch.floor(y).long(), torch.floor(x).long()
    def kernel(t):
        t = t.abs()
        return torch.where(t <= 1, 1.5*t**3-2.5*t*t+1,
                           torch.where(t < 2, -.5*t**3+2.5*t*t-4*t+2, 0.))
    result = torch.zeros(torch.broadcast_shapes(y.shape, x.shape), dtype=image.dtype, device=image.device)
    for dy in (-1, 0, 1, 2):
        for dx in (-1, 0, 1, 2):
            sy, sx = iy+dy, ix+dx
            weight = kernel(y-sy)*kernel(x-sx)*((sy >= 0)&(sy < h)&(sx >= 0)&(sx < w))
            result += image[sy.clamp(0, h-1), sx.clamp(0, w-1)]*weight
    return result


def fractional_score(image, layer, section, templates, strength, delta, sampler):
    cy, cx = layer['y'][section, None, None], layer['x'][section, None, None]
    patch = sampler(image, cy+layer['py']+delta[:, 1, None, None], cx+layer['px']+delta[:, 0, None, None])
    patch = patch-(patch*layer['weight']).sum((-2, -1))[:, None, None]
    norm = (patch.square()*layer['weight']).sum((-2, -1)).sqrt()*strength
    return (patch*templates*layer['weight']).sum((-2, -1))/norm.clamp_min(1e-15)


def refined_peak(engine, image, layer, section, templates, strength, sampler, refine):
    costs = engine._scores(image, layer, section, templates, strength)
    delta, valid, peak = peaks(costs)
    if not refine:
        return delta, valid, peak
    n = costs.shape[-1]
    index = costs.flatten(1).argmax(1)
    integer = torch.stack((index % n-n//2, index//n-n//2), dim=1).to(delta.dtype)
    # Start from the better of the quadratic and integer proposals. All updates
    # require an actual NCC improvement; original ambiguity/curvature gates stay.
    score = fractional_score(image, layer, section, templates, strength, delta, sampler)
    integer_score = fractional_score(image, layer, section, templates, strength, integer, sampler)
    use_integer = integer_score > score
    delta = torch.where(use_integer[:, None], integer, delta)
    score = torch.maximum(score, integer_score)
    for step in (.25, .125, .0625):
        values = {}
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            offset = torch.tensor([dx*step, dy*step], device=delta.device)
            values[dy, dx] = fractional_score(image, layer, section, templates, strength, delta+offset, sampler)
        gx = (values[0, 1]-values[0, -1])/(2*step)
        gy = (values[1, 0]-values[-1, 0])/(2*step)
        a = (2*score-values[0, 1]-values[0, -1])/step**2
        c = (2*score-values[1, 0]-values[-1, 0])/step**2
        cross = -(values[1, 1]-values[1, -1]-values[-1, 1]+values[-1, -1])/(4*step**2)
        determinant = a*c-cross*cross
        usable = (a > 0)&(c > 0)&(determinant > 1e-12)
        denominator = torch.where(usable, determinant, torch.ones_like(determinant))
        update = torch.stack(((c*gx-cross*gy)/denominator, (a*gy-cross*gx)/denominator), dim=1).clamp(-.25, .25)
        proposal = integer+(delta+update-integer).clamp(-1., 1.)
        candidate = fractional_score(image, layer, section, templates, strength, proposal, sampler)
        accept = valid & usable & torch.isfinite(candidate) & (candidate > score)
        delta = torch.where(accept[:, None], proposal, delta)
        score = torch.where(accept, candidate, score)
    return torch.where(valid[:, None], delta, torch.zeros_like(delta)), valid, peak


def trace(engine, proxy, global_shift, *, cubic=False, refine=False):
    sampler = cubic_sample if cubic else sample
    image0 = torch.as_tensor(proxy, device=engine.device, dtype=torch.float64)
    origin = torch.as_tensor(global_shift, device=engine.device, dtype=torch.float64)[:, None, None]
    field = origin.expand(2, *engine.shape).clone()
    stages, records = [], []
    for layer in reversed(engine.layers):
        image = sampler(image0, engine.yy+field[1], engine.xx+field[0])
        observed = ((engine.xx+field[0] >= 0)&(engine.xx+field[0] <= engine.shape[1]-1)
                    &(engine.yy+field[1] >= 0)&(engine.yy+field[1] <= engine.shape[0]-1))
        integral = F.pad((~observed).long().cumsum(0).cumsum(1), (1, 0, 1, 0))
        y, x = layer['y'], layer['x']
        margin = layer['h']+layer['m']
        supported = (integral[y+margin+1,x+margin+1]-integral[y-margin,x+margin+1]
                     -integral[y+margin+1,x-margin]+integral[y-margin,x-margin]) == 0
        numerator, denominator = torch.zeros_like(field), torch.zeros_like(image)
        details = []
        for start in range(0, len(y), layer['chunk']):
            section = slice(start, start+layer['chunk'])
            cy, cx = y[section,None,None], x[section,None,None]
            delta, valid, peak = refined_peak(engine, image, layer, section, layer['templates'][section], layer['strength'][section], sampler, refine)
            valid &= supported[section]
            patch = sampler(image, cy+layer['py']+delta[:,1,None,None], cx+layer['px']+delta[:,0,None,None])
            patch -= (patch*layer['weight']).sum((-1,-2))[:,None,None]
            strength = (patch.square()*layer['weight']).sum((-1,-2)).sqrt()
            reverse, reverse_valid, _ = refined_peak(engine, engine.reference, layer, section, patch, strength, sampler, refine)
            valid &= reverse_valid & (torch.linalg.vector_norm(reverse,dim=1) <= .75)
            confidence = ((peak-.8)/.15).clamp(0.,1.)*valid
            weight = layer['footprint']*confidence[:,None,None]
            indices = ((cy+layer['py'])*engine.shape[1]+cx+layer['px']).flatten()
            denominator.flatten().scatter_add_(0,indices,weight.flatten())
            for axis in range(2):
                numerator[axis].flatten().scatter_add_(0,indices,(weight*delta[:,axis,None,None]).flatten())
            details.append(torch.column_stack((delta,valid,peak,confidence,torch.linalg.vector_norm(reverse,dim=1))))
        residual = numerator/denominator.clamp_min(1e-300)
        residual *= denominator.clamp_max(1.)
        # Field composition remains the original sampler for every policy.
        proposed = residual+sample(field,engine.yy+residual[1],engine.xx+residual[0],nearest=True)
        ux, uy = proposed-origin
        ux_y, ux_x = torch.gradient(ux)
        uy_y, uy_x = torch.gradient(uy)
        jacobian = (1+ux_x)*(1+uy_y)-ux_y*uy_x
        accepted = (torch.isfinite(proposed).all() & (jacobian.min() >= .25) & (torch.hypot(ux,uy).max() <= 6.))
        field = torch.where(accepted,proposed,field)
        stages.append(field)
        records.append(dict(measurements=torch.cat(details).cpu().numpy(), accepted=bool(accepted)))
    return stages, records


def observations(engine, stages, records, shift):
    origin = torch.as_tensor(shift, device=engine.device, dtype=torch.float64)[:, None, None]
    parent = origin.expand(2, *engine.shape)
    values, confidence = [], []
    for j, (stage, record, layer) in enumerate(zip(stages, records, reversed(engine.layers))):
        measurement = record['measurements']
        delta = torch.as_tensor(measurement[:, :2], device=engine.device)
        yy = layer['y'][:,None,None]+layer['py']+delta[:,1,None,None]
        xx = layer['x'][:,None,None]+layer['px']+delta[:,0,None,None]
        previous = (sample(parent, yy, xx, nearest=True)*engine.kernels[j]).sum((-2,-1))
        values.append((delta.T+previous-origin[:,0]).T.cpu().numpy())
        confidence.append(measurement[:,4]*record['accepted'])
        parent = stage
    return np.concatenate(values), np.concatenate(confidence)
