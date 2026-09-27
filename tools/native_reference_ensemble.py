"""Disjoint native reference cohorts with fixed AP geometry and raw sampling."""
import numpy as np
import torch
from planetrecon.backends.torch_circular import sample
from planetrecon.pipeline.local_align import LocalRegistration, pull


NAMES = ['baseline', 'ref_a', 'ref_b', 'ref_c', 'ref_d', 'pooled', 'field_mean']


def cohorts(indices, quality, excluded, seed=270927):
    """Best 256 remaining frames, one from each chronological quartet per cohort."""
    available = np.setdiff1d(np.arange(len(indices)), excluded)
    chosen = available[np.argsort(-quality[available], kind='stable')[:256]]
    if len(chosen) != 256:
        raise ValueError('Need 256 independent reference candidates')
    chronological = chosen[np.argsort(indices[chosen], kind='stable')].reshape(64,4)
    rng = np.random.default_rng(seed)
    assigned = np.array([row[rng.permutation(4)] for row in chronological])
    return assigned.T.copy()


def build_references(source, indices, shifts, members, baseline):
    sums, supports = [], []
    ones = np.ones_like(baseline)
    for cohort in members:
        signal, support = np.zeros_like(baseline), np.zeros_like(baseline)
        for at in cohort:
            frame = source.read_raw(int(indices[at])).astype(float)
            signal += pull(frame, shifts[at])
            support += pull(ones, shifts[at])
        sums.append(signal)
        supports.append(support)
    sums, supports = np.array(sums), np.array(supports)
    references = {'baseline': baseline}
    for i,name in enumerate(NAMES[1:5]):
        references[name] = np.divide(sums[i], supports[i], out=baseline.copy(), where=supports[i]>0)
    references['pooled'] = np.divide(sums.sum(0),supports.sum(0),out=baseline.copy(),where=supports.sum(0)>0)
    return references, sums, supports


class NativeReferenceStates:
    """Only matching templates change; baseline AP layout and fitter stay fixed."""
    def __init__(self, engine, references):
        self.engine = engine
        self.states = {'baseline': (engine.reference,[(layer['templates'],layer['strength']) for layer in engine.layers])}
        for name,reference in references.items():
            if name == 'baseline':
                continue
            proxy = torch.as_tensor(LocalRegistration.proxy(reference), device=engine.device, dtype=torch.float64)
            patches = []
            for layer in engine.layers:
                patch = proxy[layer['y'][:,None,None]+layer['py'],layer['x'][:,None,None]+layer['px']]
                patch = patch-(patch*layer['weight']).sum((-2,-1))[:,None,None]
                strength = (patch.square()*layer['weight']).sum((-2,-1)).sqrt()
                patches.append((patch,strength))
            self.states[name] = proxy, patches

    def select(self,name):
        self.engine.reference,patches = self.states[name]
        for layer,(template,strength) in zip(self.engine.layers,patches,strict=True):
            layer['templates'],layer['strength'] = template,strength


def mean_field(fields, shift):
    mean = torch.stack(fields).mean(0)
    origin = torch.as_tensor(shift,device=mean.device,dtype=mean.dtype)[:,None,None]
    residual = mean-origin
    ux_y,ux_x = torch.gradient(residual[0])
    uy_y,uy_x = torch.gradient(residual[1])
    jacobian = (1+ux_x)*(1+uy_y)-ux_y*uy_x
    maximum = torch.linalg.vector_norm(residual,dim=0).max()
    accepted = bool(torch.isfinite(mean).all() & (jacobian.min() >= .25) & (maximum <= 6.))
    return (mean if accepted else origin.expand_as(mean).clone()), dict(field_guard_accepted=accepted,
        minimum_jacobian=float(jacobian.min()),maximum_residual_px=float(maximum),fallback=False)
