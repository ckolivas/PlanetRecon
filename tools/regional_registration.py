"""Diagnostic spatial motion checks using the existing independent-pixel model.

Only displacement confidence is blended. Original frame pixels and brightness
are untouched until the existing final sampler is applied.
"""
import numpy as np
from scipy.ndimage import center_of_mass, gaussian_filter
import torch

from planetrecon.backends.torch_circular import sample
from planetrecon.pipeline.local_align import LocalRegistration
from tools.refit_registration import guarded
from tools.validated_registration import ValidatedRegistration, split_proxies, blur_model_loss


def partitions(reference, mask, transition=16.):
    """Six fixed sectors: left/centre/right, each split above/below the centroid."""
    y,x = np.indices(reference.shape)
    signal = np.maximum(reference-.02*reference.max(),0.)
    cy,cx = center_of_mass(signal) if signal.sum() else (np.array(reference.shape)-1)/2
    _,support_x = np.where(mask)
    width = float(np.ptp(support_x)+1) if len(support_x) else reference.shape[1]
    boundaries = (cx-.22*width,cx+.22*width)
    column = (x>=boundaries[0]).astype(int)+(x>=boundaries[1]).astype(int)
    sector = column+3*(y>=cy)
    hard = np.stack([sector==i for i in range(6)])
    blend = np.stack([gaussian_filter(m.astype(float),transition) for m in hard])
    blend /= blend.sum(0)
    return hard,blend,dict(centre_yx=[float(cy),float(cx)],x_boundaries=list(boundaries),
                           transition_sigma_px=transition)


class RegionalRegistration(ValidatedRegistration):
    def __init__(self,reference,matcher,*,policy='regional_any',**kwargs):
        if policy not in ('coherent','full_any','regional_any','regional_both'):
            raise ValueError('unknown regional policy')
        super().__init__(reference,matcher,**kwargs)
        self.policy = policy
        hard,blend,self.partition = partitions(np.asarray(reference),self.mask.cpu().numpy().astype(bool))
        self.regions = torch.as_tensor(hard*self.mask.cpu().numpy(),device=self.device)
        self.blend = torch.as_tensor(blend,device=self.device)

    def variants(self,frame,shift,should_cancel):
        origin = torch.as_tensor(shift,device=self.device,dtype=torch.float64)[:,None,None]
        global_field = origin.expand(2,*self.shape)
        sensor_y,sensor_x = self.yy-origin[1],self.xx-origin[0]
        masks = sample(torch.cat((self.mask[None],self.regions)),sensor_y,sensor_x)>.5
        sufficient = [int(m.sum())>=64 for m in masks]
        halves = split_proxies(frame)
        global_prediction = self.render(sensor_x,sensor_y)
        checks = []
        for half in (0,1):
            field = self.engine.displacement(halves[half],shift,should_cancel)
            observed = torch.as_tensor(halves[1-half],device=self.device)
            if sufficient[0]:
                tx,ty,error = self.inverse(field,masks[0])
            else:
                tx,ty,error = sensor_x,sensor_y,float('inf')
            prediction = self.render(tx,ty) if error<=.02 else global_prediction
            scores = []
            for mask,enough in zip(masks,sufficient):
                if not enough:
                    scores.append(dict(accepted=False,pixels=int(mask.sum()),reason='insufficient support'))
                    continue
                gl,gb = blur_model_loss(global_prediction,observed,mask)
                ll,lb = blur_model_loss(prediction,observed,mask)
                scores.append(dict(accepted=bool(ll<gl and error<=.02),pixels=int(mask.sum()),
                                   global_loss=gl,local_loss=ll,global_blur_model=gb,local_blur_model=lb))
            checks.append(scores)
        full = self.engine.displacement(LocalRegistration.proxy(frame),shift,should_cancel)
        counts = np.sum([[s['accepted'] for s in check] for check in checks],axis=0)
        stats = dict(points=self.engine.stats[-1]['points'],fallback=self.engine.stats[-1]['fallback'],
                     field_guard_accepted=self.engine.stats[-1]['field_guard_accepted'],
                     full_fit=self.engine.stats[-1],accepted_halves=int(counts[0]),
                     region_accepted_halves=counts[1:].tolist(),checks=checks)
        fields = dict(coherent=full,full_any=full if counts[0] else global_field.clone())
        for name,threshold in [('regional_any',1),('regional_both',2)]:
            accepted = torch.as_tensor(counts[1:]>=threshold,device=self.device,dtype=torch.float64)
            confidence = torch.einsum('r,rhw->hw',accepted,self.blend)
            candidate = origin+(full-origin)*confidence
            fields[name] = guarded(candidate,origin)
            stats[name+'_guard_accepted'] = fields[name] is candidate
        self.stats.append(stats)
        return fields

    def displacement_raw(self,frame,shift,should_cancel):
        return self.variants(frame,shift,should_cancel)[self.policy]
