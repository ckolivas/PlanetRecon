"""Confidence-gated local translation matching; image data are never sharpened."""
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates


def patch_centres(length, window, step, max_shift=3):
    """Regular supported grid; centre an axis when only one patch fits."""
    margin = window // 2 + max_shift
    centres = np.arange(margin, length-margin, step)
    if len(centres) == 1:
        centres[0] = (length-1) // 2
    return centres


def pull(image, shift_xy):
    image = np.asarray(image, dtype=np.float64)
    yy, xx = np.indices(image.shape[:2], dtype=np.float64)
    coordinates = [yy + shift_xy[1], xx + shift_xy[0]]
    if image.ndim == 2:
        return map_coordinates(image, coordinates, order=1, mode='grid-constant', prefilter=False)
    return np.stack([map_coordinates(image[..., c], coordinates, order=1, mode='grid-constant', prefilter=False) for c in range(image.shape[2])], axis=-1)

class LocalRegistration:

    def __init__(self, reference, *, window=65, step=32, max_shift=3, use_cuda=False, valid_mask=None):
        reference = np.asarray(reference, dtype=np.float64)
        if reference.ndim != 2 or not np.isfinite(reference).all():
            raise ValueError('finite two-dimensional reference required')
        if type(window) is not int or window < 15 or window % 2 != 1:
            raise ValueError('odd window of at least 15 pixels required')
        if type(step) is not int or step < 1:
            raise ValueError('positive integer patch step required')
        if type(max_shift) is not int or not 1 <= max_shift <= 5:
            raise ValueError('integer local shift limit in [1, 5] required')
        self.use_cuda = use_cuda
        self.shape = reference.shape
        self.window = window
        self.step = step
        self.max_shift = max_shift
        if valid_mask is not None:
            valid_mask = np.asarray(valid_mask,dtype=bool)
            if valid_mask.shape != reference.shape:
                raise ValueError('local reference support must match its shape')
        self.ref = self.proxy(reference)
        self.ys = patch_centres(reference.shape[0], window, step, max_shift)
        self.xs = patch_centres(reference.shape[1], window, step, max_shift)
        self.weight = np.outer(np.hanning(window), np.hanning(window))
        self.weight /= self.weight.sum()
        gy, gx = np.gradient(self.ref)
        self.texture_valid = []
        self.templates = []
        self.strength = []
        for y in self.ys:
            for x in self.xs:
                patch = self.ref[y - window // 2:y + window // 2 + 1, x - window // 2:x + window // 2 + 1]
                section = np.s_[y-window//2:y+window//2+1, x-window//2:x+window//2+1]
                dx, dy = gx[section], gy[section]
                xx, yy, xy = ((a*self.weight).sum() for a in (dx*dx, dy*dy, dx*dy))
                smallest = .5*(xx+yy-np.sqrt(max((xx-yy)**2+4*xy**2, 0.)))
                margin = window//2+max_shift
                observed = (valid_mask is None or valid_mask[y-margin:y+margin+1,x-margin:x+margin+1].all())
                self.texture_valid.append(observed and smallest > max(.01*(xx+yy), 1e-12))
                patch = patch - (patch * self.weight).sum()
                self.templates.append(patch)
                self.strength.append(np.sqrt((patch ** 2 * self.weight).sum()))
        self.templates = np.asarray(self.templates)
        self.strength = np.asarray(self.strength)

    @staticmethod
    def proxy(frame):
        a = gaussian_filter(np.asarray(frame, dtype=float), 1.5)
        return a - gaussian_filter(a, 8.0)

    def displacement(self, frame, global_shift):
        frame = np.asarray(frame, dtype=np.float64)
        if frame.shape != self.shape or not np.isfinite(frame).all():
            raise ValueError('finite frame matching the reference required')
        if len(global_shift) != 2 or not np.isfinite(global_shift).all():
            raise ValueError('finite global displacement required')
        if not len(self.templates) or self.strength.max() < 1e-12:
            return tuple((np.full(self.shape, v) for v in global_shift))
        img = pull(self.proxy(frame), global_shift)
        all_costs = self._cuda_costs(img) if self.use_cuda else None
        h = self.window // 2
        m = self.max_shift
        values = np.zeros((2, len(self.ys), len(self.xs)))
        confidence = np.zeros((len(self.ys), len(self.xs)))
        for j, y in enumerate(self.ys):
            for i, x in enumerate(self.xs):
                k = j * len(self.xs) + i
                # Pulling the globally aligned proxy pads unobserved detector
                # samples with zero. Every candidate must have real support;
                # otherwise the padding edge can win or bias the local peak.
                if (x-h-m+global_shift[0] < 0 or x+h+m+global_shift[0] > self.shape[1]-1
                        or y-h-m+global_shift[1] < 0 or y+h+m+global_shift[1] > self.shape[0]-1):
                    continue
                if not self.texture_valid[k] or self.strength[k] < self.strength.max() * 0.08:
                    continue
                if all_costs is not None:
                    costs = all_costs[k]
                else:
                    ref = self.templates[k]
                    costs = np.empty((2*m+1,2*m+1))
                    candidates = np.lib.stride_tricks.sliding_window_view(
                        img[y-h-m:y+h+m+1,x-h-m:x+h+m+1], (self.window,self.window))
                    # Centered correlations, with bounded candidate blocks and
                    # fused reductions instead of per-offset temporary arrays.
                    rows = max(1,262144//((2*m+1)*self.window**2))
                    for start in range(0,2*m+1,rows):
                        patch = candidates[start:start+rows]
                        mean = np.einsum('ijxy,xy->ij',patch,self.weight)
                        patch = patch-mean[...,None,None]
                        variance = np.einsum('ijxy,ijxy,xy->ij',patch,patch,self.weight)
                        numerator = np.einsum('ijxy,xy,xy->ij',patch,ref,self.weight)
                        costs[start:start+rows] = numerator/np.maximum(np.sqrt(variance)*self.strength[k],1e-15)
                py, px = np.unravel_index(costs.argmax(), costs.shape)
                if py in (0, 2 * m) or px in (0, 2 * m) or costs[py, px] < 0.8:
                    continue
                patch = costs[py-1:py+2, px-1:px+2]
                if not np.isfinite(patch).all():
                    continue
                peak = patch[1, 1]
                gx = .5*(patch[1, 2]-patch[1, 0])
                gy = .5*(patch[2, 1]-patch[0, 1])
                hxx = patch[1, 0]-2*peak+patch[1, 2]
                hyy = patch[0, 1]-2*peak+patch[2, 1]
                hxy = .25*(patch[2, 2]-patch[2, 0]-patch[0, 2]+patch[0, 0])
                curvature = -np.array([[hxx, hxy], [hxy, hyy]])
                # Diagonal features couple the axes. Slice-wise parabolas give
                # biased offsets and can mistake a ridge for a constrained peak.
                if np.linalg.eigvalsh(curvature)[0] <= .001:
                    continue
                delta = np.zeros(2) if peak >= 1-1e-10 else np.linalg.solve(curvature, [gx, gy])
                # The joint centre can lie beyond half a pixel from the best
                # integer sample, but must stay within its observed neighbours.
                if not np.isfinite(delta).all() or np.any(np.abs(delta) > 1):
                    continue
                values[:, j, i] = np.array([px-m, py-m])+delta
                confidence[j, i] = 1.0
        reliable = confidence.copy()
        confidence = gaussian_filter(confidence, 0.7)
        yy, xx = np.indices(self.shape)
        # Add a zero ring one grid interval outside the measured patch centres.
        # Interpolating to that ring returns smoothly to global motion instead
        # of creating an abrupt jump that trips the whole-frame fold guard.
        coords = [1 + (yy - self.ys[0]) / self.step, 1 + (xx - self.xs[0]) / self.step]
        support = map_coordinates(np.pad(confidence, 1), coords, order=1, mode='constant')
        result = []
        for axis in range(2):
            grid = gaussian_filter(values[axis] * reliable, 0.7)
            displacement = map_coordinates(np.pad(grid, 1), coords, order=1, mode='constant')
            result.append(displacement * np.clip(support, 0, 1) + global_shift[axis])
        # Still reject a deformation that collapses or folds coordinates.
        ux, uy = result[0]-global_shift[0], result[1]-global_shift[1]
        jacobian = ((1+np.gradient(ux, axis=1))*(1+np.gradient(uy, axis=0))
                    - np.gradient(ux, axis=0)*np.gradient(uy, axis=1))
        if not np.isfinite(jacobian).all() or jacobian.min() < .25:
            return tuple(np.full(self.shape, value) for value in global_shift)
        return tuple(result)

    def _cuda_costs(self, image):
        import torch
        w, m = (self.window, self.max_shift)
        active = np.flatnonzero(np.asarray(self.texture_valid)
                                & (self.strength >= self.strength.max()*.08))
        costs = np.full((len(self.templates),2*m+1,2*m+1), -np.inf)
        if not len(active):
            return costs
        img = torch.as_tensor(image, device='cuda:0', dtype=torch.float64)
        margin = w//2+m
        img = img[int(self.ys[0])-margin:int(self.ys[-1])+margin+1,
                  int(self.xs[0])-margin:int(self.xs[-1])+margin+1]
        tile = img.unfold(0, w + 2 * m, self.step).unfold(1, w + 2 * m, self.step)
        weight = torch.as_tensor(self.weight, device='cuda:0', dtype=torch.float64)
        # Only patches eligible for the existing confidence gates need scores.
        # Gather bounded chunks directly from the strided detector view instead
        # of copying every tile (including unsupported sky) to a dense tensor.
        for start in range(0, len(active), 8):
            ids = active[start:start+8]
            rows = torch.as_tensor(ids//len(self.xs), device='cuda:0')
            cols = torch.as_tensor(ids%len(self.xs), device='cuda:0')
            patches = tile[rows,cols].unfold(1, w, 1).unfold(2, w, 1)
            ref = torch.as_tensor(self.templates[ids], device='cuda:0', dtype=torch.float64)
            strength = torch.as_tensor(self.strength[ids], device='cuda:0', dtype=torch.float64)
            mean = (patches * weight).sum(dim=(-1, -2))
            variance = torch.clamp((patches.square() * weight).sum(dim=(-1, -2)) - mean.square(), min=0)
            numerator = (patches * weight * ref[:, None, None]).sum(dim=(-1, -2))
            denom = torch.sqrt(variance) * strength[:, None, None]
            costs[ids] = (numerator / denom.clamp(min=1e-15)).cpu().numpy()
        return costs


def build_template(reference, indices, read_plane, correlate, max_shift, should_cancel=None):
    """Average at most 64 aligned proxies; retain the best-frame origin."""
    total = np.zeros_like(reference)
    weight = np.zeros_like(reference)
    ones = np.ones_like(reference)
    used = 0
    for index in indices:
        if should_cancel is not None and should_cancel():
            raise InterruptedError('cancelled while building the alignment template')
        frame = read_plane(int(index))
        shift = correlate(reference, frame)
        if not np.isfinite(shift).all() or max(abs(v) for v in shift) > max_shift:
            continue
        total += pull(frame, shift)
        weight += pull(ones, shift)
        used += 1
    if used < 4:
        raise ValueError('Local alignment needs four valid aligned template frames; use global alignment.')
    template = np.divide(total, weight, out=reference.copy(), where=weight > 0)
    origin = correlate(reference, template)
    if not np.isfinite(origin).all() or max(abs(v) for v in origin) > max_shift:
        raise ValueError('Local template origin is unreliable; use global alignment.')
    # Re-centring also resamples the detector footprint. Normalise partial
    # coverage, and use the best observation in its original coordinates where
    # the aligned mean has no data. Padding must not become a dark registration
    # feature, nor may the pre-fill reference be shifted as if it were the mean.
    observed = weight > 0
    anchored = pull(np.where(observed, template, 0.), origin)
    support = pull(observed, origin)
    return np.divide(anchored, support, out=reference.copy(), where=support > 0)


def cpu_backproject(raw, shift, color, *, demosaiced=None):
    """Resample original colour measurements and their support exactly once."""
    from planetrecon.detector import is_bayer, cfa_labels, bilinear_demosaic
    raw = np.asarray(raw, dtype=np.float64)
    support = pull(np.ones(raw.shape[:2]), shift)
    if is_bayer(color):
        masks = (cfa_labels(*raw.shape, color)[..., None] == np.array(list('RGB'))).astype(float)
        if demosaiced is None:
            demosaiced = bilinear_demosaic(raw, color)
        return (pull(raw[..., None]*masks, shift), pull(masks, shift),
                pull(demosaiced, shift), support)
    return pull(raw, shift), np.broadcast_to(support[..., None], raw.shape) if raw.ndim == 3 else support, None, None
