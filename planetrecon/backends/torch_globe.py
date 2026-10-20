"""Float64 CUDA globe projection and resident measurement accumulation.

Uses the CPU operator's pixel centres, visibility and near-integer snapping.
Only small attitude matrices are computed on the host. No capture-sized cache
or interpolated-frame stack is retained; memory is bounded by detector size.

Poses may be a single ``FramePose`` or a sequence of them (a batch). Batched
calls compute the same per-pose host scalars and return a leading batch axis,
so several frames share each kernel launch.
"""
import numpy as np
import torch

from planetrecon.detector import cfa_labels
from planetrecon.geometry.globe import body_to_obs_matrix
from planetrecon.geometry.pose import FramePose


class PoseColumns:
    """Per-pose host scalars: floats for one pose, (B, 1, 1) device columns for a batch."""

    def __init__(self, src, ref, device):
        self.single = isinstance(src, FramePose) and isinstance(ref, FramePose)
        srcs = [src] if isinstance(src, FramePose) else list(src)
        refs = [ref] if isinstance(ref, FramePose) else list(ref)
        n = max(len(srcs), len(refs))
        if {len(srcs), len(refs)} - {1, n}:
            raise ValueError('source and reference pose batches differ in length')
        self.srcs, self.refs = srcs * (n // len(srcs)), refs * (n // len(refs))
        self.device = device

    def __len__(self):
        return len(self.srcs)

    def __call__(self, values):
        values = np.asarray(values, dtype=np.float64)
        if self.single:
            return float(values.reshape(-1)[0])
        return torch.as_tensor(values, device=self.device).view(-1, 1, 1)

    def angles(self, values):
        """Rotation angles for ``TorchGlobeWarp.rotate``: a float, or one per pose."""
        values = np.asarray(values, dtype=np.float64)
        return float(values.reshape(-1)[0]) if self.single else values

    def matrices(self, lons, g):
        """body_to_obs_matrix per pose; element (i, j) is ``self(m[:, i, j])``."""
        return np.stack([body_to_obs_matrix(g.pole_pa_rad, g.sub_obs_lat_rad, lon) for lon in lons])

    def mask(self, flags):
        """Per-pose booleans as a broadcasting device column."""
        return torch.as_tensor(np.asarray(flags, dtype=bool), device=self.device).view(-1, 1, 1)


def _fused_kernels(device):
    """Triton bilinear pull/scatter on CUDA (identical arithmetic), else None."""
    if not str(device).startswith('cuda'):
        return None
    try:
        from planetrecon.backends import triton_warp
    except ImportError:
        return None
    return triton_warp


class TorchGlobeWarp:
    def __init__(self, shape, model, device='cuda:0'):
        self.model = model
        self.device = device
        self.fused = _fused_kernels(device)
        self.h, self.w = shape
        self.y, self.x = torch.meshgrid(
            torch.arange(self.h, dtype=torch.float64, device=device)+.5,
            torch.arange(self.w, dtype=torch.float64, device=device)+.5, indexing='ij')

    def tensor(self, value):
        if torch.is_tensor(value):
            return value.to(device=self.device, dtype=torch.float64)
        return torch.tensor(np.ascontiguousarray(value), dtype=torch.float64, device=self.device)

    def poses(self, src, ref):
        return PoseColumns(src, ref, self.device)

    @staticmethod
    def rotate(x, y, angle):
        c, s = np.cos(angle), np.sin(angle)
        if np.ndim(angle):
            # One angle per pose in a batch: (B, 1, 1) columns.
            c, s = (torch.as_tensor(v, dtype=torch.float64, device=x.device).view(-1, 1, 1) for v in (c, s))
        return c*x-s*y, s*x+c*y

    def map(self, src, ref, *, limb_taper=None, x=None, y=None):
        g = self.model.globe
        P = self.poses(src, ref)
        explicit = x is not None
        x, y = (self.x,self.y) if x is None else (x,y)
        sx, sy = x-P([p.cx for p in P.srcs]), P([p.cy for p in P.srcs])-y
        static = np.array([not self.model.apply_surface or g.surface_rate_rad_s*(s.t_s-r.t_s) == 0
                           for s, r in zip(P.srcs, P.refs)])
        if static.any():
            angle = P.angles([r.field_angle_rad-s.field_angle_rad if self.model.apply_field else 0.
                              for s, r in zip(P.srcs, P.refs)])
            fx, fy = self.rotate(sx, sy, angle)
            fx, fy = fx+P([r.cx for r in P.refs]), P([r.cy for r in P.refs])-fy
            fixed = fx, fy, torch.isfinite(fx) & torch.isfinite(fy)
            if static.all():
                return fixed
        if self.fused is not None:
            moved = self.fused_map(P, x if explicit else None, y if explicit else None, limb_taper)
            if not static.any():
                return moved
            keep = P.mask(static)
            return tuple(torch.where(keep, a, b) for a, b in zip(fixed, moved))
        if self.model.apply_field:
            sx, sy = self.rotate(sx, sy, P.angles([-s.field_angle_rad for s in P.srcs]))
        a, c = g.equatorial_radius_px, g.polar_radius_px
        ia, ic = 1/(a*a), 1/(c*c)
        r = P.matrices([g.sub_obs_lon(s.t_s) for s in P.srcs], g)
        q = [P(r[:,0,i])*sx+P(r[:,1,i])*sy for i in range(3)]
        d = [P(r[:,2,i]) for i in range(3)]
        aa = (r[:,2,0]**2+r[:,2,1]**2)*ia+r[:,2,2]**2*ic
        bb = 2*((q[0]*d[0]+q[1]*d[1])*ia+q[2]*d[2]*ic)
        cc = (q[0]**2+q[1]**2)*ia+q[2]**2*ic-1
        disc = bb*bb-4*P(aa)*cc
        t = torch.where(disc >= 0, (-bb+disc.clamp_min(0).sqrt())/P(np.maximum(2*aa,1e-30)), torch.nan)
        xb, yb, zb = [q[i]+t*d[i] for i in range(3)]
        nz = P(r[:,2,0])*(xb*ia)+P(r[:,2,1])*(yb*ia)+P(r[:,2,2])*(zb*ic)
        mu_src = nz / ((xb*ia)**2+(yb*ia)**2+(zb*ic)**2).sqrt().clamp_min(1e-30)
        on_globe = (disc >= 0) & torch.isfinite(t) & (nz >= -1e-12)
        # The CPU operator's longitude/latitude round trip, written as the radial
        # projection onto the ellipsoid it equals (no float64 transcendentals).
        scale = torch.where(on_globe, ((xb*xb+yb*yb)/(a*a)+zb*zb/(c*c)).sqrt(), torch.nan)
        xb, yb, zb = xb/scale, yb/scale, zb/scale
        r = P.matrices([g.sub_obs_lon(p.t_s) for p in P.refs], g)
        rx = P(r[:,0,0])*xb+P(r[:,0,1])*yb+P(r[:,0,2])*zb
        ry = P(r[:,1,0])*xb+P(r[:,1,1])*yb+P(r[:,1,2])*zb
        nz = P(r[:,2,0])*(xb*ia)+P(r[:,2,1])*(yb*ia)+P(r[:,2,2])*(zb*ic)
        nlen = ((xb*ia)**2+(yb*ia)**2+(zb*ic)**2).sqrt()
        visible = torch.isfinite(rx) & (nz/nlen.clamp_min(1e-30) >= -1e-12)
        mu_ref = nz/nlen.clamp_min(1e-30)
        if self.model.apply_field:
            angle = P.angles([p.field_angle_rad for p in P.refs])
            rx, ry = self.rotate(rx, ry, angle)
            sx, sy = self.rotate(sx, sy, angle)
        use = on_globe & visible
        cx, cy = P([p.cx for p in P.refs]), P([p.cy for p in P.refs])
        if limb_taper is not None:
            blend = torch.ones_like(sx)
            for mu in (mu_src,mu_ref):
                u = (torch.where(use,mu,0.) / limb_taper).clamp(0.,1.).square()
                blend *= u*u*(3.-2.*u)
            x = sx+cx+blend*torch.where(blend > 0,rx-sx,0.)
            y = cy-sy-blend*torch.where(blend > 0,ry-sy,0.)
            moved = x, y, torch.isfinite(x)&torch.isfinite(y)
        else:
            x, y = torch.where(use,rx,sx)+cx, cy-torch.where(use,ry,sy)
            moved = x, y, torch.isfinite(x) & torch.isfinite(y) & (~on_globe | visible)
        if not static.any():
            return moved
        # A mixed batch (e.g. the anchor frame itself) keeps the static mapping per pose.
        keep = P.mask(static)
        return tuple(torch.where(keep, a, b) for a, b in zip(fixed, moved))

    def fused_map(self, P, x, y, limb_taper):
        """The rotating-globe branch of ``map`` as one fused kernel; per-pose scalars in one table."""
        from planetrecon.backends import triton_warp as K
        g, field = self.model.globe, self.model.apply_field
        params = np.zeros((len(P), K.PARAMETERS))
        for row, (s, r) in zip(params, zip(P.srcs, P.refs)):
            row[K.SRC_CX], row[K.SRC_CY] = s.cx, s.cy
            row[K.SRC_COS], row[K.SRC_SIN] = np.cos(-s.field_angle_rad), np.sin(-s.field_angle_rad)
            row[K.REF_CX], row[K.REF_CY] = r.cx, r.cy
            row[K.REF_COS], row[K.REF_SIN] = np.cos(r.field_angle_rad), np.sin(r.field_angle_rad)
            rs = body_to_obs_matrix(g.pole_pa_rad, g.sub_obs_lat_rad, g.sub_obs_lon(s.t_s))
            aa = (rs[2,0]**2+rs[2,1]**2)/g.equatorial_radius_px**2+rs[2,2]**2/g.polar_radius_px**2
            row[K.AA4], row[K.DENOM] = 4*aa, max(2*aa, 1e-30)
            row[K.RS:K.RS+9] = rs.reshape(-1)
            row[K.RR:K.RR+9] = body_to_obs_matrix(g.pole_pa_rad, g.sub_obs_lat_rad, g.sub_obs_lon(r.t_s)).reshape(-1)
        out = K.globe_map(torch.as_tensor(params, device=self.device), (self.h, self.w), g, x=x, y=y,
                          field=field, limb_taper=limb_taper)
        return tuple(o[0] for o in out) if P.single else out

    def corners(self, x, y, valid):
        """Bilinear destination pixels and weights, flattened per frame (leading batch axes kept)."""
        lead = x.shape[:-2]
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
                yield ((yy.clamp(0,self.h-1)*self.w+xx.clamp(0,self.w-1)).reshape(*lead, -1),
                       (wy*wx*keep).reshape(*lead, -1))

    def render_tensor(self, image, src, ref, *, per_pose=False):
        """Pull ``image`` through the mapping; ``per_pose`` images carry the batch axis first."""
        image = self.tensor(image) if not isinstance(image,torch.Tensor) else image
        x, y, valid = self.map(src, ref)
        lead = x.shape[:-2]
        if self.fused is not None:
            out = self.fused.pull(image, x, y, valid, per_pose)
            return out.reshape(*lead, *(image.shape[len(lead):] if per_pose else image.shape))
        if per_pose:
            flat = image.reshape(*lead, self.h*self.w, -1)
            rows = torch.arange(flat.shape[0], device=self.device)[:, None]
            gather = lambda index: flat[rows, index]
        else:
            flat = image.reshape(self.h*self.w, -1)
            gather = lambda index: flat[index]
        out = None
        for index, weight in self.corners(x,y,valid):
            part = gather(index)*weight[..., None]
            out = part if out is None else out+part
        shape = image.shape[len(lead):] if per_pose else image.shape
        return out.reshape(*lead, *shape)

    def render(self, image, src, ref):
        return self.render_tensor(image, src, ref).cpu().numpy()


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

    def sample_map(self, pose, ref, sample_xy=None):
        coords = {} if sample_xy is None else {'x':self.tensor(sample_xy[0]),'y':self.tensor(sample_xy[1])}
        return self.map(pose,ref,**coords)

    def sample_corners(self, pose, ref, sample_xy=None):
        return self.corners(*self.sample_map(pose,ref,sample_xy))

    def add_layer_coverage(self, index, weighted):
        pass

    def layer_labels(self):
        """Per-source-pixel layer labels for the fused scatter (None without layers)."""
        return None, (0, 0, 0), None

    def add(self, frame, pose, ref, score, demo=None, sample_xy=None):
        """Scatter one frame, or a batch when ``pose`` is a sequence.

        ``score`` is a scalar or a reference-geometry map per frame (batches:
        shape (B,) or (B, h, w)). Returns whether each frame covered any pixel.
        """
        batch = not isinstance(pose, FramePose)
        lead = (len(pose),) if batch else ()
        values = self.tensor(frame).reshape(*lead, self.h*self.w, -1)
        demo = None if demo is None else self.tensor(demo).reshape(*lead, -1, 3)
        if self.fused is not None:
            n = lead[0] if batch else 1
            score = self.tensor(score)
            score = score.reshape(n) if score.numel() == n else score.reshape(n, -1)
            x, y, valid = self.sample_map(pose, ref, sample_xy)
            labels, codes, layer_sums = self.layer_labels()
            covered = self.fused.scatter(
                x.reshape(-1, self.h, self.w), y.reshape(-1, self.h, self.w), valid.reshape(-1, self.h, self.w),
                values.reshape(n, self.h*self.w, -1), score,
                (self.accum.view(-1), self.weight.view(-1),
                 None if self.demosaic_accum is None else self.demosaic_accum.view(-1),
                 None if self.demosaic_weight is None else self.demosaic_weight.view(-1)),
                demo=None if demo is None else demo.reshape(n, -1, 3), cfa=self.channels,
                labels=None if labels is None else labels.reshape(n, -1), layer_sums=layer_sums,
                layer_codes=codes)
            return (covered > 0).tolist() if batch else bool(covered[0] > 0)
        covered = torch.zeros(lead, dtype=torch.float64, device=self.device)
        # A per-pixel score (reference geometry) applies at each destination pixel.
        score_map = None
        if torch.is_tensor(score) and score.ndim > len(lead):
            score_map = score.reshape(*lead, -1)
        elif batch:
            score = self.tensor(score).reshape(-1, 1)
        rows = torch.arange(lead[0], device=self.device)[:, None] if batch else None
        for index, weight in self.sample_corners(pose,ref,sample_xy):
            covered += weight.sum(-1)
            if score_map is None:
                weighted = weight*score
            else:
                weighted = weight*(score_map[rows, index] if batch else score_map[index])
            flat_index, flat_weighted = index.reshape(-1), weighted.reshape(-1)
            if self.channels is not None:
                dest = (index*3+self.channels).reshape(-1)
                self.accum.flatten().scatter_add_(0,dest,flat_weighted*values[...,0].reshape(-1))
                self.weight.flatten().scatter_add_(0,dest,flat_weighted)
            else:
                channels = values.shape[-1]
                dest = flat_index[:,None].expand(-1,channels)
                self.accum.view(-1,channels).scatter_add_(0,dest,flat_weighted[:,None]*values.reshape(-1,channels))
                self.weight.view(-1,channels).scatter_add_(0,dest,flat_weighted[:,None].expand(-1,channels))
            if demo is not None:
                dest = flat_index[:,None].expand(-1,3)
                self.demosaic_accum.view(-1,3).scatter_add_(0,dest,flat_weighted[:,None]*demo.reshape(-1,3))
                self.demosaic_weight.view(-1,3).scatter_add_(0,dest,flat_weighted[:,None].expand(-1,3))
            self.add_layer_coverage(flat_index,flat_weighted)
        return (covered > 0).tolist() if batch else bool(covered > 0)

    def download(self):
        # Detached snapshots: later device updates cannot mutate UI/checkpoints.
        return tuple(None if a is None else a.cpu().numpy().copy() for a in
                     (self.accum,self.weight,self.demosaic_accum,self.demosaic_weight))
