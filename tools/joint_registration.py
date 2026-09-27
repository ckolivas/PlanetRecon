"""Experimental joint raw-pixel forward fit of blur and smooth inverse motion.

Only a quarter of sensor samples train the model. The other three quarters
measure predictive error. Original frame pixels are never filtered or scaled
for stacking; fitted gain/offset belong only to the forward-model score.
"""
import numpy as np
from scipy.ndimage import gaussian_filter, binary_dilation, distance_transform_edt
from scipy.optimize import minimize
import torch
import torch.nn.functional as F

from planetrecon.backends.torch_circular import sample
from planetrecon.pipeline.local_align import LocalRegistration
from tools.coherent_registration import CoherentRegistration, cubic_basis


def photometric_loss(prediction, observed):
    x, y = prediction-prediction.mean(), observed-observed.mean()
    gain = ((x*y).mean()/x.square().mean().clamp_min(1e-20)).clamp(.5, 2.)
    offset = observed.mean()-gain*prediction.mean()
    return (gain*prediction+offset-observed).square().mean(), gain, offset


def global_blur_fit(predictions, observed, variances):
    """Profile the same adjacent-template mixture and gain bounds without motion.

    Evaluate endpoints and stationary points for the free-gain and each
    gain-boundary branch. This avoids giving the moving model a more flexible
    blur hypothesis than the stationary alternative.
    """
    means = predictions.mean(1)
    x = predictions-means[:,None]
    y = observed-observed.mean()
    gram = (x@x.T/len(y)).cpu().numpy()
    cross = (x@y/len(y)).cpu().numpy()
    energy = float(y.square().mean())
    best = None
    for i in range(len(variances)-1):
        aa = gram[i,i]
        ad = gram[i,i+1]-aa
        dd = max(gram[i+1,i+1]+aa-2*gram[i,i+1],0.)
        c,dc = cross[i],cross[i+1]-cross[i]
        candidates = [0.,1.]
        denominator = c*dd-dc*ad
        if abs(denominator)>1e-20:
            candidates.append(float(np.clip((dc*aa-c*ad)/denominator,0.,1.)))
        if dd>1e-20:
            candidates.extend(float(np.clip((dc/g-ad)/dd,0.,1.)) for g in (.5,2.))
        for fraction in candidates:
            variance = max(aa+2*fraction*ad+fraction*fraction*dd,1e-20)
            covariance = c+fraction*dc
            gain = float(np.clip(covariance/variance,.5,2.))
            loss = max(energy-2*gain*covariance+gain*gain*variance,0.)
            if best is None or loss<best[0]:
                best = loss, i, fraction, gain
    loss,i,fraction,gain = best
    offset = float(observed.mean()-gain*((1-fraction)*means[i]+fraction*means[i+1]))
    value = variances[i]+fraction*(variances[i+1]-variances[i])
    return float(value), gain, offset, loss


class JointRegistration:
    def __init__(self, reference, matcher, *, device='cuda:0', spacing=32., maxiter=100):
        if spacing < 8 or maxiter < 1:
            raise ValueError('spacing >= 8 and positive iteration count required')
        self.device, self.shape = device, reference.shape
        self.spacing, self.maxiter = spacing, maxiter
        self.engine = CoherentRegistration(matcher, device=device, spacing=16., stiffness=.01, patch_average=True)
        self.yy, self.xx = self.engine.yy, self.engine.xx
        self.sigmas = np.arange(0.,2.01,.25)
        self.variances = self.sigmas**2
        self.bank = torch.as_tensor(np.stack([gaussian_filter(reference,s) if s else reference
                                             for s in self.sigmas]), device=device, dtype=torch.float64)
        mask = binary_dilation(gaussian_filter(reference,3.)>reference.max()*.04, iterations=6)
        mask[:8] = mask[-8:] = False
        mask[:,:8] = mask[:,-8:] = False
        self.mask = torch.as_tensor(mask.astype(float), device=device)
        distance = distance_transform_edt(~mask)
        self.support = torch.as_tensor(.5*(1+np.cos(np.pi*np.clip(distance/32.,0,1))), device=device)
        self.ny, self.nx = [int(np.floor((n-1)/spacing))+4 for n in self.shape]
        self.bx = self.basis(np.arange(self.shape[1]), self.nx)
        self.by = self.basis(np.arange(self.shape[0]), self.ny)
        self.stats = []

    def basis(self, coords, count):
        return torch.as_tensor(cubic_basis(coords,count,self.spacing).toarray(), device=self.device)

    def render(self, reference, x, y):
        h,w = self.shape
        grid = torch.stack((2*x/(w-1)-1,2*y/(h-1)-1),dim=-1)[None]
        return F.grid_sample(reference[None,None],grid,mode='bicubic',padding_mode='border',align_corners=True)[0,0]

    def mixture(self, parameter):
        value = float(parameter.detach())
        i = min(len(self.variances)-2, max(0, int(np.searchsorted(self.variances,value,side='right')-1)))
        fraction = (parameter-self.variances[i])/(self.variances[i+1]-self.variances[i])
        return self.bank[i]+fraction*(self.bank[i+1]-self.bank[i])

    def fit(self, frame, shift, stiffness, should_cancel):
        observed = torch.as_tensor(frame, device=self.device, dtype=torch.float64)
        origin = torch.as_tensor(shift, device=self.device, dtype=torch.float64)[:,None,None]
        qx,qy = self.xx-origin[0], self.yy-origin[1]
        mask = sample(self.mask,qy,qx)>.5
        train = mask[::2,::2]
        if int(train.sum()) < 64:
            return origin.expand(2,*self.shape).clone(), dict(fallback=True, reason='insufficient support')
        target = observed[::2,::2][train]
        global_predictions = torch.stack([self.render(ref,qx[::2,::2],qy[::2,::2])[train] for ref in self.bank])
        initial_variance,global_gain,global_offset,_ = global_blur_fit(global_predictions,target,self.variances)
        bx = self.basis(np.clip(np.arange(0,self.shape[1],2)-shift[0],0,self.shape[1]-1),self.nx)
        by = self.basis(np.clip(np.arange(0,self.shape[0],2)-shift[1],0,self.shape[0]-1),self.ny)
        n = 2*self.nx*self.ny
        start = np.zeros(n+1)
        start[-1] = initial_variance

        def objective(values):
            if should_cancel and should_cancel():raise InterruptedError('cancelled during joint fit')
            parameters = torch.tensor(values, device=self.device, requires_grad=True)
            coefficients = parameters[:n].reshape(2,self.ny,self.nx)
            inverse = by@coefficients@bx.T
            reference = self.mixture(parameters[-1])
            prediction = self.render(reference,qx[::2,::2]-inverse[0],qy[::2,::2]-inverse[1])[train]
            loss,_,_ = photometric_loss(prediction,target)
            dx = torch.diff(coefficients,n=2,dim=2)
            dy = torch.diff(coefficients,n=2,dim=1)
            mixed = torch.diff(torch.diff(coefficients,dim=2),dim=1)
            penalty = stiffness*(32/self.spacing)**4*(dx.square().mean()+dy.square().mean()+2*mixed.square().mean())
            total = loss+penalty+1e-4*coefficients.square().mean()
            total.backward()
            return float(total.detach()), parameters.grad.cpu().numpy()

        result = minimize(objective,start,jac=True,method='L-BFGS-B',
            bounds=[(-6.,6.)]*n+[(0.,4.)],
            options=dict(maxiter=self.maxiter,maxls=30,ftol=1e-9,gtol=1e-5))
        parameters = torch.tensor(result.x, device=self.device)
        inverse = (self.by@parameters[:n].reshape(2,self.ny,self.nx)@self.bx.T)*self.support
        # q = x + u(x), x = q - v(q), hence u(x) = v(x + u(x)).
        residual = inverse.clone()
        for _ in range(30):
            residual = sample(inverse,self.yy+residual[1],self.xx+residual[0],nearest=True)
        sampled = sample(inverse,self.yy+residual[1],self.xx+residual[0],nearest=True)
        inversion_error = float(torch.linalg.vector_norm(residual-sampled,dim=0).max())
        ux_y,ux_x = torch.gradient(residual[0])
        uy_y,uy_x = torch.gradient(residual[1])
        determinant = (1+ux_x)*(1+uy_y)-ux_y*uy_x
        valid = bool(torch.isfinite(residual).all() & (determinant.min()>=.25)
                     & (torch.linalg.vector_norm(residual,dim=0).max()<=6.)) and inversion_error<=.02
        heldout = mask.clone()
        heldout[::2,::2] = False
        iv = sample(inverse,qy,qx,nearest=True)
        ref = self.mixture(parameters[-1])
        predicted = self.render(ref,qx-iv[0],qy-iv[1])
        _,gain,offset = photometric_loss(predicted[::2,::2][train],target)
        heldout_loss = float((gain*predicted[heldout]+offset-observed[heldout]).square().mean())
        global_ref = self.render(self.mixture(torch.tensor(initial_variance,device=self.device,dtype=torch.float64)),qx,qy)
        global_heldout = float((global_gain*global_ref[heldout]+global_offset-observed[heldout]).square().mean())
        stats = dict(fallback=not valid, success=bool(result.success), message=str(result.message),
            iterations=int(result.nit), evaluations=int(result.nfev), objective=float(result.fun),
            gradient_inf=float(np.max(np.abs(result.jac))),
            sigma=float(np.sqrt(result.x[-1])), initial_sigma=float(np.sqrt(initial_variance)),
            stiffness=stiffness, inverse_error_px=inversion_error, minimum_jacobian=float(determinant.min()),
            train_pixels=int(train.sum()), heldout_pixels=int(heldout.sum()),
            heldout_loss=heldout_loss, global_heldout_loss=global_heldout,
            heldout_improves=heldout_loss<global_heldout-1e-9*max(1.,global_heldout))
        return origin+residual if valid else origin.expand(2,*self.shape).clone(), stats

    def variants(self, frame, shift, should_cancel):
        fields = {'coherent':self.engine.displacement(LocalRegistration.proxy(frame),shift,should_cancel)}
        stats = {}
        for name, stiffness in [('joint',.03),('joint_strong',.3)]:
            fields[name],stats[name] = self.fit(frame,shift,stiffness,should_cancel)
        origin = torch.as_tensor(shift,device=self.device,dtype=torch.float64)[:,None,None]
        fields['joint_validated'] = (fields['joint_strong'] if stats['joint_strong'].get('heldout_improves',False)
                                     else origin.expand(2,*self.shape).clone())
        self.stats.append(stats)
        return fields
