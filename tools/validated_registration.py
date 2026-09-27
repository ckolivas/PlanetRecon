"""Experimental motion validation on disjoint detector samples.

Fit a coherent field from each checkerboard half, then score its forward model
against the other half. Only model proxies are blurred or contrast fitted;
stacking still resamples the original raw frame once.
"""
import numpy as np
from scipy.ndimage import gaussian_filter, binary_dilation
import torch
import torch.nn.functional as F

from planetrecon.pipeline.local_align import LocalRegistration
from planetrecon.backends.torch_circular import sample
from tools.coherent_registration import CoherentRegistration


def split_proxies(frame):
    y,x=np.indices(frame.shape)
    parity=(x+y)%2
    result=[]
    for half in (0,1):
        mask=(parity==half).astype(float)
        smooth=gaussian_filter(np.asarray(frame)*mask,1.5)/gaussian_filter(mask,1.5)
        result.append(smooth-gaussian_filter(smooth,8.))
    return result


def blur_model_loss(templates, observed, mask):
    """Fit adjacent blur templates with positive gain, without resampling data.

    Both competing motion models have the same nuisance blur/contrast freedom.
    Noise in the held-out pixels is consequently the same in both residuals.
    """
    y=observed[mask]
    y=y-y.mean()
    x=templates[:,mask]
    x=x-x.mean(1,keepdim=True)
    gram=x@x.T/len(y)
    cross=x@y/len(y)
    energy=y.square().mean()
    diagonal=gram.diag().clamp_min(1e-30)
    gain=(cross/diagonal).clamp(.5,2.)
    losses=energy-2*gain*cross+gain.square()*diagonal
    # A positive combination of adjacent templates interpolates the blur bank.
    aa,bb,ab=diagonal[:-1],diagonal[1:],gram.diagonal(1)
    determinant=aa*bb-ab.square()
    valid=determinant>1e-10*aa*bb
    divisor=torch.where(valid,determinant,torch.ones_like(determinant))
    ca=(bb*cross[:-1]-ab*cross[1:])/divisor
    cb=(aa*cross[1:]-ab*cross[:-1])/divisor
    valid &= (ca>=0)&(cb>=0)&(ca+cb>=.5)&(ca+cb<=2.)
    mixtures=energy-2*(ca*cross[:-1]+cb*cross[1:])+ca.square()*aa+cb.square()*bb+2*ca*cb*ab
    mixtures=mixtures.masked_fill(~valid,torch.inf)
    all_losses=torch.cat((losses,mixtures))
    value,index=all_losses.min(0)
    return float(value.clamp_min(0)),int(index)


class ValidatedRegistration:
    def __init__(self, reference, matcher, *, device='cuda:0', spacing=16, stiffness=.01):
        self.engine=CoherentRegistration(matcher,device=device,spacing=spacing,
                                         stiffness=stiffness,patch_average=True)
        self.device=device
        self.shape=self.engine.shape
        self.yy,self.xx=self.engine.yy,self.engine.xx
        self.sigmas=np.arange(0.,2.01,.25)
        bank=np.stack([LocalRegistration.proxy(gaussian_filter(reference,s) if s else reference)
                       for s in self.sigmas])
        self.bank=torch.as_tensor(bank,device=device)
        mask=binary_dilation(gaussian_filter(reference,3.)>reference.max()*.04,iterations=6)
        mask[:8]=mask[-8:]=False
        mask[:,:8]=mask[:,-8:]=False
        self.mask=torch.as_tensor(mask.astype(float),device=device)
        self.stats=[]

    def render(self, tx, ty):
        h,w=self.shape
        grid=torch.stack((2*tx/(w-1)-1,2*ty/(h-1)-1),dim=-1)[None]
        return F.grid_sample(self.bank[None],grid,mode='bicubic',padding_mode='border',align_corners=True)[0]

    def inverse(self, field, mask):
        tx,ty=self.xx-field[0],self.yy-field[1]
        for _ in range(20):
            delta=sample(field,ty,tx,nearest=True)
            tx,ty=self.xx-delta[0],self.yy-delta[1]
        delta=sample(field,ty,tx,nearest=True)
        error=torch.hypot(tx+delta[0]-self.xx,ty+delta[1]-self.yy)
        return tx,ty,float(error[mask].max())

    def displacement_raw(self, frame, global_shift, should_cancel):
        halves=split_proxies(frame)
        origin=torch.as_tensor(global_shift,device=self.device,dtype=torch.float64)[:,None,None]
        global_field=origin.expand(2,*self.shape)
        mask=sample(self.mask,self.yy-origin[1],self.xx-origin[0])>.5
        if not bool(mask.any()):
            self.stats.append({'points':0,'fallback':True,'field_guard_accepted':True,'accepted_halves':0})
            return global_field.clone()
        global_prediction=self.render(self.xx-origin[0],self.yy-origin[1])
        accepted=[]; scores=[]; fit_stats=[]
        for half in (0,1):
            field=self.engine.displacement(halves[half],global_shift,should_cancel)
            fit_stats.append(self.engine.stats[-1])
            validation=torch.as_tensor(halves[1-half],device=self.device)
            global_loss,global_blur=blur_model_loss(global_prediction,validation,mask)
            tx,ty,inverse_error=self.inverse(field,mask)
            if inverse_error<=.02:
                local_loss,local_blur=blur_model_loss(self.render(tx,ty),validation,mask)
            else:
                local_loss,local_blur=global_loss,-1
            use_local=local_loss<global_loss and inverse_error<=.02
            accepted.append(field if use_local else global_field)
            scores.append({'global_loss':global_loss,'local_loss':local_loss,'global_blur_model':global_blur,
                           'local_blur_model':local_blur,'inverse_error_px':inverse_error,'accepted':use_local})
        result=(accepted[0]+accepted[1])*.5
        ux,uy=result-origin
        ux_y,ux_x=torch.gradient(ux);uy_y,uy_x=torch.gradient(uy)
        valid=bool(torch.isfinite(result).all() & (((1+ux_x)*(1+uy_y)-ux_y*uy_x).min()>=.25))
        self.stats.append({'points':float(np.mean([s['points'] for s in fit_stats])),
                           'fallback':all(s['fallback'] for s in fit_stats),'field_guard_accepted':valid,
                           'accepted_halves':sum(s['accepted'] for s in scores),'validation':scores})
        return result if valid else global_field.clone()
