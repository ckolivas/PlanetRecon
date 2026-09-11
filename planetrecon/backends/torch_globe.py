"""Float64 CUDA globe projection and resident measurement accumulation.

Uses the CPU operator's pixel centres, visibility and near-integer snapping.
Only small attitude matrices are computed on the host. No capture-sized cache
or interpolated-frame stack is retained; memory is bounded by detector size.
"""
import numpy as np
import torch

from planetrecon.detector import cfa_labels
from planetrecon.geometry.globe import body_to_obs_matrix


class TorchGlobeWarp:
    def __init__(self, shape, model, device='cuda:0'):
        self.model = model
        self.device = device
        self.h, self.w = shape
        self.y, self.x = torch.meshgrid(
            torch.arange(self.h, dtype=torch.float64, device=device)+.5,
            torch.arange(self.w, dtype=torch.float64, device=device)+.5, indexing='ij')

    def tensor(self, value):
        return torch.tensor(np.ascontiguousarray(value), dtype=torch.float64, device=self.device)

    @staticmethod
    def rotate(x, y, angle):
        c, s = np.cos(angle), np.sin(angle)
        return c*x-s*y, s*x+c*y

    def map(self, src, ref, *, limb_taper=None, x=None, y=None):
        g = self.model.globe
        x, y = (self.x,self.y) if x is None else (x,y)
        sx, sy = x-src.cx, src.cy-y
        if not self.model.apply_surface or g.surface_rate_rad_s*(src.t_s-ref.t_s) == 0:
            angle = ref.field_angle_rad-src.field_angle_rad if self.model.apply_field else 0.
            x, y = self.rotate(sx, sy, angle)
            x, y = x+ref.cx, ref.cy-y
            return x, y, torch.isfinite(x) & torch.isfinite(y)
        if self.model.apply_field:
            sx, sy = self.rotate(sx, sy, -src.field_angle_rad)
        a, c = g.equatorial_radius_px, g.polar_radius_px
        ia, ic = 1/(a*a), 1/(c*c)
        r = body_to_obs_matrix(g.pole_pa_rad, g.sub_obs_lat_rad, g.sub_obs_lon(src.t_s))
        q = [r[0,i]*sx+r[1,i]*sy for i in range(3)]
        d = r[2]
        aa = (d[0]**2+d[1]**2)*ia+d[2]**2*ic
        bb = 2*((q[0]*d[0]+q[1]*d[1])*ia+q[2]*d[2]*ic)
        cc = (q[0]**2+q[1]**2)*ia+q[2]**2*ic-1
        disc = bb*bb-4*aa*cc
        t = torch.where(disc >= 0, (-bb+disc.clamp_min(0).sqrt())/max(2*aa,1e-30), torch.nan)
        xb, yb, zb = [q[i]+t*d[i] for i in range(3)]
        nz = r[2,0]*(xb*ia)+r[2,1]*(yb*ia)+r[2,2]*(zb*ic)
        mu_src = nz / ((xb*ia)**2+(yb*ia)**2+(zb*ic)**2).sqrt().clamp_min(1e-30)
        on_globe = (disc >= 0) & torch.isfinite(t) & (nz >= -1e-12)
        lon = torch.where(on_globe, torch.atan2(yb, xb), torch.nan)
        lat = torch.where(on_globe, torch.atan2(zb/c, torch.hypot(xb,yb)/a), torch.nan)
        xb, yb, zb = a*lat.cos()*lon.cos(), a*lat.cos()*lon.sin(), c*lat.sin()
        r = body_to_obs_matrix(g.pole_pa_rad, g.sub_obs_lat_rad, g.sub_obs_lon(ref.t_s))
        rx = r[0,0]*xb+r[0,1]*yb+r[0,2]*zb
        ry = r[1,0]*xb+r[1,1]*yb+r[1,2]*zb
        nz = r[2,0]*(xb*ia)+r[2,1]*(yb*ia)+r[2,2]*(zb*ic)
        nlen = ((xb*ia)**2+(yb*ia)**2+(zb*ic)**2).sqrt()
        visible = torch.isfinite(rx) & (nz/nlen.clamp_min(1e-30) >= -1e-12)
        mu_ref = nz/nlen.clamp_min(1e-30)
        if self.model.apply_field:
            rx, ry = self.rotate(rx, ry, ref.field_angle_rad)
            sx, sy = self.rotate(sx, sy, ref.field_angle_rad)
        use = on_globe & visible
        if limb_taper is not None:
            blend = torch.ones_like(sx)
            for mu in (mu_src,mu_ref):
                u = (torch.where(use,mu,0.) / limb_taper).clamp(0.,1.).square()
                blend *= u*u*(3.-2.*u)
            x = sx+ref.cx+blend*torch.where(blend > 0,rx-sx,0.)
            y = ref.cy-sy-blend*torch.where(blend > 0,ry-sy,0.)
            return x,y,torch.isfinite(x)&torch.isfinite(y)
        x, y = torch.where(use,rx,sx)+ref.cx, ref.cy-torch.where(use,ry,sy)
        valid = torch.isfinite(x) & torch.isfinite(y) & (~on_globe | visible)
        return x, y, valid

    def corners(self, x, y, valid):
        x, y = x-.5, y-.5
        finite = torch.isfinite(x) & torch.isfinite(y)
        x, y = torch.where(finite,x,-2.), torch.where(finite,y,-2.)
        x = torch.where((x-x.round()).abs() < 1e-10,x.round(),x)
        y = torch.where((y-y.round()).abs() < 1e-10,y.round(),y)
        ix, iy = x.floor().long(), y.floor().long()
        fx, fy = x-ix, y-iy
        for dy, wy in ((0,1-fy),(1,fy)):
            for dx, wx in ((0,1-fx),(1,fx)):
                xx, yy = ix+dx, iy+dy
                keep = valid & (xx >= 0) & (xx < self.w) & (yy >= 0) & (yy < self.h)
                yield (yy.clamp(0,self.h-1)*self.w+xx.clamp(0,self.w-1)).flatten(), (wy*wx*keep).flatten()

    def render(self, image, src, ref):
        image = self.tensor(image) if not isinstance(image,torch.Tensor) else image
        x, y, valid = self.map(src, ref)
        flat = image.reshape(self.h*self.w,-1)
        out = torch.zeros_like(flat)
        for index, weight in self.corners(x,y,valid):
            out += flat[index]*weight[:,None]
        return out.reshape(image.shape).cpu().numpy()


class TorchGlobeAccumulator(TorchGlobeWarp):
    def __init__(self, shape, model, color, accum, weight, demosaic_accum, demosaic_weight, device='cuda:0'):
        super().__init__(shape,model,device)
        self.accum, self.weight = self.tensor(accum), self.tensor(weight)
        self.demosaic_accum = None if demosaic_accum is None else self.tensor(demosaic_accum)
        self.demosaic_weight = None if demosaic_weight is None else self.tensor(demosaic_weight)
        self.channels = None
        if demosaic_accum is not None:
            labels = cfa_labels(*shape,color)
            self.channels = torch.as_tensor(np.where(labels=='R',0,np.where(labels=='G',1,2)),
                                            device=self.device).flatten()

    def sample_corners(self, pose, ref, sample_xy=None):
        coords = {} if sample_xy is None else {'x':self.tensor(sample_xy[0]),'y':self.tensor(sample_xy[1])}
        return self.corners(*self.map(pose,ref,**coords))

    def add_layer_coverage(self, index, weighted):
        pass

    def add(self, frame, pose, ref, score, demo=None, sample_xy=None):
        values = self.tensor(frame).reshape(self.h*self.w,-1)
        demo = None if demo is None else self.tensor(demo).reshape(-1,3)
        covered = torch.zeros((), dtype=torch.float64, device=self.device)
        for index, weight in self.sample_corners(pose,ref,sample_xy):
            covered += weight.sum()
            weighted = weight*score
            if self.channels is not None:
                dest = index*3+self.channels
                self.accum.flatten().scatter_add_(0,dest,weighted*values[:,0])
                self.weight.flatten().scatter_add_(0,dest,weighted)
            else:
                channels = values.shape[1]
                dest = index[:,None].expand(-1,channels)
                self.accum.view(-1,channels).scatter_add_(0,dest,weighted[:,None]*values)
                self.weight.view(-1,channels).scatter_add_(0,dest,weighted[:,None].expand(-1,channels))
            if demo is not None:
                dest = index[:,None].expand(-1,3)
                self.demosaic_accum.view(-1,3).scatter_add_(0,dest,weighted[:,None]*demo)
                self.demosaic_weight.view(-1,3).scatter_add_(0,dest,weighted[:,None].expand(-1,3))
            self.add_layer_coverage(index,weighted)
        return bool(covered > 0)

    def download(self):
        # Detached snapshots: later device updates cannot mutate UI/checkpoints.
        return tuple(None if a is None else a.cpu().numpy().copy() for a in
                     (self.accum,self.weight,self.demosaic_accum,self.demosaic_weight))
