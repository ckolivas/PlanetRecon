"""CUDA Saturn geometry with the CPU model's disjoint-layer support rules."""
import torch

from planetrecon.backends.torch_globe import TorchGlobeAccumulator, TorchGlobeWarp
from planetrecon.geometry.globe import body_to_obs_matrix
from planetrecon.geometry.rings import EDGE_ON_MU, ring_normal_obs, sun_direction_obs
from planetrecon.geometry.saturn import LAYER_FAR_RING, LAYER_GLOBE, LAYER_NEAR_RING, LAYER_MOON, LIMB_TAPER_MU


class TorchSaturnWarp(TorchGlobeWarp):
    def __init__(self, shape, model, device='cuda:0'):
        super().__init__(shape,model,device)
        g = model.globe
        self.ring_matrix = body_to_obs_matrix(g.pole_pa_rad,g.sub_obs_lat_rad,0.)
        self.normal = ring_normal_obs(g)
        self.sun = sun_direction_obs(g,model.rings)

    def classify(self, pose, x=None, y=None, *, mask_moon=True):
        x, y = (self.x,self.y) if x is None else (x,y)
        model, g, rings = self.model, self.model.globe, self.model.rings
        sx, sy = x-pose.cx, pose.cy-y
        if model.apply_field:
            sx, sy = self.rotate(sx,sy,-pose.field_angle_rad)
        t_s = pose.t_s if model.apply_surface else g.reference_epoch_s
        r = body_to_obs_matrix(g.pole_pa_rad,g.sub_obs_lat_rad,g.sub_obs_lon(t_s))
        ia, ic = 1/g.equatorial_radius_px**2, 1/g.polar_radius_px**2
        q = [r[0,i]*sx+r[1,i]*sy for i in range(3)]
        d = r[2]
        aa = (d[0]**2+d[1]**2)*ia+d[2]**2*ic
        bb = 2*((q[0]*d[0]+q[1]*d[1])*ia+q[2]*d[2]*ic)
        cc = (q[0]**2+q[1]**2)*ia+q[2]**2*ic-1
        disc = bb*bb-4*aa*cc
        tg = torch.where(disc >= 0,(-bb+disc.clamp_min(0).sqrt())/max(2*aa,1e-30),torch.nan)
        xb,yb,zb = [q[i]+tg*d[i] for i in range(3)]
        nz = r[2,0]*(xb*ia)+r[2,1]*(yb*ia)+r[2,2]*(zb*ic)
        on_globe = (disc >= 0) & torch.isfinite(tg) & (nz >= -1e-12)
        tg = torch.where(on_globe,tg,torch.nan)
        n, rt = self.normal, self.ring_matrix.T
        if model.edge_on:
            tr = torch.full_like(sx,torch.nan)
            on_ring = torch.zeros_like(sx,dtype=torch.bool)
        else:
            tr = -(n[0]*sx+n[1]*sy)/n[2]
            xb = rt[0,0]*sx+rt[0,1]*sy+rt[0,2]*tr
            yb = rt[1,0]*sx+rt[1,1]*sy+rt[1,2]*tr
            radius = torch.hypot(xb,yb)
            on_ring = torch.isfinite(tr) & (radius >= rings.inner_radius_px) & (radius <= rings.outer_radius_px)
            tr = torch.where(on_ring,tr,torch.nan)
        near, far = on_ring & (tr >= 0), on_ring & (tr < 0)
        labels = torch.zeros_like(sx,dtype=torch.int8)
        labels = torch.where(far,LAYER_FAR_RING,labels)
        labels = torch.where(on_globe,LAYER_GLOBE,labels)
        labels = torch.where(near,LAYER_NEAR_RING,labels)

        # A ray from each ring point toward the Sun intersects the oblate globe.
        pb = [rt[i,0]*sx+rt[i,1]*sy+rt[i,2]*tr for i in range(3)]
        db = rt @ self.sun
        aa = (db[0]**2+db[1]**2)*ia+db[2]**2*ic
        bb = 2*((pb[0]*db[0]+pb[1]*db[1])*ia+pb[2]*db[2]*ic)
        cc = (pb[0]**2+pb[1]**2)*ia+pb[2]**2*ic-1
        disc = bb*bb-4*aa*cc
        root = disc.clamp_min(0).sqrt()
        u1,u2 = (-bb-root)/max(2*aa,1e-30),(-bb+root)/max(2*aa,1e-30)
        globe_shadow = on_ring & (disc >= 0) & ((u1 > 1e-4) | (u2 > 1e-4))
        # The reverse shadow test intersects globe-to-Sun rays with the annulus.
        denom = float(n @ self.sun)
        ring_shadow = torch.zeros_like(on_globe)
        if abs(denom) >= EDGE_ON_MU:
            u = -(n[0]*sx+n[1]*sy+n[2]*tg)/denom
            px,py,pz = sx+u*self.sun[0],sy+u*self.sun[1],tg+u*self.sun[2]
            xb = rt[0,0]*px+rt[0,1]*py+rt[0,2]*pz
            yb = rt[1,0]*px+rt[1,1]*py+rt[1,2]*pz
            radius = torch.hypot(xb,yb)
            ring_shadow = on_globe & (u > 1e-4) & (radius >= rings.inner_radius_px) & (radius <= rings.outer_radius_px)

        invalid = torch.zeros_like(on_globe)
        if model.edge_on:
            invalid |= ((n[0]*sx+n[1]*sy).abs() <= rings.outer_radius_px*abs(n[2])+.5) & (torch.hypot(sx,sy) <= rings.outer_radius_px+.5)
        moon = model.moon
        if moon is not None and mask_moon:
            dt = pose.t_s-g.reference_epoch_s
            mx,my = moon.x+moon.vx_px_s*dt-pose.cx,pose.cy-moon.y-moon.vy_px_s*dt
            mx,my = self.rotate(mx,my,-model.field_angle0_rad if model.apply_field else 0.)
            mx,my = self.rotate(mx,my,pose.field_angle_rad if model.apply_field else 0.)
            moon_mask = torch.hypot(x-(mx+pose.cx),y-(pose.cy-my)) <= moon.radius_px
            labels = torch.where(moon_mask,LAYER_MOON,labels)
            invalid |= moon_mask
        motion_labels = torch.where(on_globe & (labels != LAYER_MOON),LAYER_GLOBE,labels)
        regions = torch.zeros_like(labels)
        regions = torch.where(motion_labels == LAYER_GLOBE,torch.where(ring_shadow,3,1),regions)
        regions = torch.where((motion_labels == LAYER_NEAR_RING) | (motion_labels == LAYER_FAR_RING),
                              torch.where(globe_shadow,4,2),regions)
        regions = torch.where(invalid,-1,regions)
        # Preserve quality scoring on exposed globe texture, independently of
        # the added overlap coverage. Ring edges must not inflate frame scores.
        quality_regions = torch.where((labels == LAYER_GLOBE) & ~(on_ring & model.low_opening),regions,-1)
        return {'labels':labels,'motion_labels':motion_labels,'regions':regions,'invalid':invalid,
                'quality_regions':quality_regions}

    def map(self, src, ref, info=None, *, x=None, y=None):
        # The continuous warp only consumes exclusions, not ring/shadow labels.
        # Classification is still performed for scoring and layer coverage.
        needs_mask = self.model.moon is not None or self.model.edge_on
        mapped_info = (self.classify(src,x,y) if x is not None or info is None else info) if needs_mask else None
        x,y,valid = super().map(src,ref,limb_taper=LIMB_TAPER_MU,x=x,y=y)
        if mapped_info is not None:
            valid &= ~mapped_info['invalid']
        if info is not None and needs_mask:
            valid &= ~info['invalid']
        return x,y,valid


class TorchSaturnAccumulator(TorchGlobeAccumulator, TorchSaturnWarp):
    """Reuse colour scatter sums with continuous Saturn motion mapping.

    The accumulator's cooperative initializer creates the Saturn warp, then
    allocates resident sums. Only scoring's region mask returns to the CPU.
    """
    def __init__(self, shape, model, color, accum, weight, demosaic_accum, demosaic_weight,
                 globe_weight, ring_weight, ref, device='cuda:0'):
        super().__init__(shape,model,color,accum,weight,demosaic_accum,demosaic_weight,device)
        self.globe_weight,self.ring_weight = self.tensor(globe_weight),self.tensor(ring_weight)
        self.prepared_pose,self.source_info = None,None

    def prepare_source(self, pose):
        self.source_info = self.classify(pose)
        self.prepared_pose = pose
        return self.source_info['quality_regions'].cpu().numpy()

    def sample_corners(self, pose, ref, sample_xy=None):
        if self.prepared_pose != pose:
            self.prepare_source(pose)
        coords = {} if sample_xy is None else {'x':self.tensor(sample_xy[0]),'y':self.tensor(sample_xy[1])}
        yield from self.corners(*self.map(pose,ref,self.source_info,**coords))

    def add_layer_coverage(self, index, weighted):
        labels = self.source_info['motion_labels'].flatten()
        self.globe_weight.flatten().scatter_add_(0,index,weighted*(labels == LAYER_GLOBE))
        self.ring_weight.flatten().scatter_add_(0,index,weighted*((labels == LAYER_NEAR_RING)|(labels == LAYER_FAR_RING)))

    def download(self):
        return (*super().download(),self.globe_weight.cpu().numpy().copy(),self.ring_weight.cpu().numpy().copy())
