"""Experimental robust spline fit to circular-AP displacement measurements.

Only the displacement model is regularized. No frame brightness adjustment,
output filtering, or temporal mean subtraction is performed.
"""
import numpy as np
from scipy import sparse
from scipy.ndimage import distance_transform_edt
from scipy.sparse.linalg import spsolve
import torch

from planetrecon.backends.torch_circular import TorchCircularRegistration, sample
from tools.local_warp_trace import trace


def cubic_basis(coordinates, count, spacing):
    position = np.asarray(coordinates)/spacing
    index = np.floor(position).astype(int)
    t = position-index
    values = np.stack(((1-t)**3, 3*t**3-6*t*t+4,
                       -3*t**3+3*t*t+3*t+1, t**3), axis=1)/6
    return sparse.csr_matrix((values.ravel(),
        (np.repeat(np.arange(len(t)), 4), (index[:, None]+np.arange(4)).ravel())),
        shape=(len(t), count))


class SplineField:
    """Weighted point fit with bending penalty and robust vector residuals."""
    def __init__(self, shape, points_xy, spacing=32, stiffness=.1, huber=.35):
        if not np.isfinite([spacing, stiffness, huber]).all() or spacing < 4 or stiffness <= 0 or huber <= 0:
            raise ValueError('positive stiffness/huber and spacing >= 4 required')
        self.shape, self.spacing, self.huber = shape, spacing, huber
        self.ny, self.nx = [int(np.floor((n-1)/spacing))+4 for n in shape]
        points = np.asarray(points_xy)
        bx = cubic_basis(points[:, 0], self.nx, spacing)
        by = cubic_basis(points[:, 1], self.ny, spacing)
        # Each observation touches 4x4 coefficients; preserve this sparsity.
        rows, cols, vals = [], [], []
        for i in range(len(points)):
            xx, yy = bx.getrow(i), by.getrow(i)
            rows.extend([i]*16)
            cols.extend((yy.indices[:, None]*self.nx+xx.indices[None, :]).ravel())
            vals.extend((yy.data[:, None]*xx.data[None, :]).ravel())
        self.design = sparse.csr_matrix((vals, (rows, cols)), shape=(len(points), self.ny*self.nx))
        self.bx = cubic_basis(np.arange(shape[1]), self.nx, spacing).toarray()
        self.by = cubic_basis(np.arange(shape[0]), self.ny, spacing).toarray()
        def differences(n, order):
            coefficients = [1., -2., 1.] if order == 2 else [-1., 1.]
            return sparse.diags(coefficients, range(order+1), shape=(n-order, n), format='csr')
        dx = sparse.kron(sparse.eye(self.ny), differences(self.nx, 2), format='csr')
        dy = sparse.kron(differences(self.ny, 2), sparse.eye(self.nx), format='csr')
        mixed = sparse.kron(differences(self.ny, 1), differences(self.nx, 1), format='csr')
        # Scale the discrete bending energy consistently between grid spacings.
        self.penalty = stiffness*(32/spacing)**2*(dx.T@dx+dy.T@dy+2*mixed.T@mixed)
        self.penalty += sparse.eye(self.ny*self.nx)*1e-6

    def fit(self, observations_xy, confidence):
        values, confidence = np.asarray(observations_xy), np.asarray(confidence)
        if values.shape != (self.design.shape[0], 2) or confidence.shape != (len(values),):
            raise ValueError('observation/confidence shape mismatch')
        if not np.isfinite(values).all() or not np.isfinite(confidence).all() or np.any(confidence < 0):
            raise ValueError('finite observations and nonnegative confidence required')
        if np.count_nonzero(confidence) < 6:
            return np.zeros((2, *self.shape)), {'points': int(np.count_nonzero(confidence)), 'fallback': True}
        weight = confidence.copy()
        for _ in range(3):
            weighted = self.design.multiply(weight[:, None])
            lhs = (self.design.T@weighted+self.penalty).tocsc()
            coefficients = spsolve(lhs, self.design.T@(weight[:, None]*values))
            residual = np.linalg.norm(self.design@coefficients-values, axis=1)
            weight = confidence*np.minimum(1., self.huber/np.maximum(residual, 1e-15))
        image = np.stack([self.by@coefficients[:, axis].reshape(self.ny, self.nx)@self.bx.T
                          for axis in range(2)])
        return image, {'points': int(np.count_nonzero(confidence)), 'fallback': False,
                       'downweighted_points': int(np.count_nonzero((weight < .5*confidence)&(confidence > 0)))}


class CoherentRegistration(TorchCircularRegistration):
    def __init__(self, matcher, *, device='cuda:0', spacing=32, stiffness=.1, patch_average=False):
        super().__init__(matcher, device=device)
        point_sets = [torch.column_stack((layer['x'], layer['y'])).cpu().numpy()
                      for layer in reversed(self.layers)]
        self.points = np.concatenate(point_sets) if point_sets else np.empty((0,2))
        self.fitter = SplineField(self.shape, self.points, spacing, stiffness)
        self.patch_average = patch_average
        self.kernels = []
        if patch_average:
            rows = []
            for layer in reversed(self.layers):
                templates = layer['templates'].cpu().numpy()
                gy,gx = np.gradient(templates,axis=(-2,-1))
                kernel = (gx*gx+gy*gy)*layer['weight'].cpu().numpy()
                kernel /= np.maximum(kernel.sum((-2,-1),keepdims=True),1e-30)
                self.kernels.append(torch.as_tensor(kernel,device=device))
                half = layer['h']
                for x,y,k in zip(layer['x'].cpu().numpy(),layer['y'].cpu().numpy(),kernel):
                    bx = self.fitter.bx[x-half:x+half+1]
                    by = self.fitter.by[y-half:y+half+1]
                    rows.append((by.T@k@bx).ravel())
            if rows:
                self.fitter.design = sparse.csr_matrix(np.asarray(rows))
        anchors = np.zeros(self.shape, dtype=bool)
        if len(self.points):
            anchors[self.points[:,1].astype(int),self.points[:,0].astype(int)] = True
        distance = distance_transform_edt(~anchors)
        phase = np.clip((distance-spacing)/(2*spacing), 0., 1.)
        self.support = .5*(1+np.cos(np.pi*phase)) if len(self.points) else np.zeros(self.shape)
        self.stats = []

    def displacement(self, proxy, global_shift, should_cancel):
        stages, records = trace(self, proxy, global_shift)
        origin = torch.as_tensor(global_shift, dtype=torch.float64, device=self.device)[:, None, None]
        parent = origin.expand(2, *self.shape)
        if not stages:
            self.stats.append({'points': 0, 'fallback': True, 'field_guard_accepted': True})
            return parent.clone()
        values, confidence = [], []
        for j,(stage, record, layer) in enumerate(zip(stages, records, reversed(self.layers))):
            if should_cancel and should_cancel():
                raise InterruptedError('cancelled during coherent registration')
            measurement = record['measurements']
            delta = torch.as_tensor(measurement[:, :2], device=self.device)
            # Compose each unfaded AP measurement with the preceding warp.
            # Fitting the tapered dense production field would treat missing
            # measurements as zero motion and retain its support-boundary bias.
            if self.patch_average:
                yy = layer['y'][:,None,None]+layer['py']+delta[:,1,None,None]
                xx = layer['x'][:,None,None]+layer['px']+delta[:,0,None,None]
                previous = (sample(parent,yy,xx,nearest=True)*self.kernels[j]).sum((-2,-1))
            else:
                previous = sample(parent, layer['y']+delta[:, 1], layer['x']+delta[:, 0], nearest=True)
            values.append((delta.T+previous-origin[:, 0]).T.cpu().numpy())
            confidence.append(measurement[:, 4]*record['accepted'])
            parent = stage
        residual, stats = self.fitter.fit(np.concatenate(values), np.concatenate(confidence))
        # Do not extrapolate an affine spline tail across blank detector space.
        # The taper is one near reference APs, independent of this frame's noise.
        residual *= self.support
        residual = torch.as_tensor(residual, device=self.device)
        ux_y, ux_x = torch.gradient(residual[0])
        uy_y, uy_x = torch.gradient(residual[1])
        determinant = (1+ux_x)*(1+uy_y)-ux_y*uy_x
        valid = bool(torch.isfinite(residual).all() & (determinant.min() >= .25)
                     & (torch.linalg.vector_norm(residual, dim=0).max() <= 6.))
        stats['field_guard_accepted'] = valid
        stats['minimum_jacobian'] = float(determinant.min())
        stats['maximum_residual_px'] = float(torch.linalg.vector_norm(residual, dim=0).max())
        self.stats.append(stats)
        # A rejected fit falls back to global translation, retaining a truthful
        # count in the report instead of silently substituting the old matcher.
        return origin+residual if valid else origin.expand(2, *self.shape).clone()
