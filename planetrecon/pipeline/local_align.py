"""Confidence-gated local translation matching; image data are never sharpened."""
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

def pull(image, shift_xy):
    image = np.asarray(image, dtype=np.float64)
    yy, xx = np.indices(image.shape[:2], dtype=np.float64)
    coordinates = [yy + shift_xy[1], xx + shift_xy[0]]
    if image.ndim == 2:
        return map_coordinates(image, coordinates, order=1, mode='grid-constant', prefilter=False)
    return np.stack([map_coordinates(image[..., c], coordinates, order=1, mode='grid-constant', prefilter=False) for c in range(image.shape[2])], axis=-1)

class LocalRegistration:

    def __init__(self, reference, *, window=65, step=32, max_shift=3, use_cuda=False):
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
        self.ref = self.proxy(reference)
        self.ys = np.arange(window // 2 + max_shift, reference.shape[0] - window // 2 - max_shift, step)
        self.xs = np.arange(window // 2 + max_shift, reference.shape[1] - window // 2 - max_shift, step)
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
                self.texture_valid.append(smallest > max(.01*(xx+yy), 1e-12))
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
                    costs = np.empty((2 * m + 1, 2 * m + 1))
                    for dy in range(-m, m + 1):
                        for dx in range(-m, m + 1):
                            patch = img[y - h + dy:y + h + dy + 1, x - h + dx:x + h + dx + 1]
                            if patch.shape != ref.shape:
                                costs[dy + m, dx + m] = -1.0
                                continue
                            patch = patch - (patch * self.weight).sum()
                            denom = np.sqrt((patch ** 2 * self.weight).sum()) * self.strength[k]
                            costs[dy + m, dx + m] = (patch * ref * self.weight).sum() / max(denom, 1e-15)
                py, px = np.unravel_index(costs.argmax(), costs.shape)
                if py in (0, 2 * m) or px in (0, 2 * m) or costs[py, px] < 0.8:
                    continue
                curvatures = []
                for axis, p in enumerate((px, py)):
                    a, b, c = costs[py, px - 1:px + 2] if axis == 0 else costs[py - 1:py + 2, px]
                    curve = a - 2 * b + c
                    curvatures.append(-curve)
                    delta = np.clip(0.5 * (a - c) / curve, -0.5, 0.5) if curve < 0 and b < 1 - 1e-10 else 0.0
                    values[axis, j, i] = p - m + delta
                if min(curvatures) > 0.001:
                    confidence[j, i] = 1.0
        reliable = confidence.copy()
        confidence = gaussian_filter(confidence, 0.7)
        yy, xx = np.indices(self.shape)
        coords = [(yy - self.ys[0]) / self.step, (xx - self.xs[0]) / self.step]
        result = []
        for axis in range(2):
            grid = gaussian_filter(values[axis] * reliable, 0.7)
            displacement = map_coordinates(grid, coords, order=1, mode='nearest')
            support = map_coordinates(confidence, coords, order=1, mode='constant')
            result.append(displacement * np.clip(support, 0, 1) + global_shift[axis])
        # Patch support ends at the grid boundary. Reject a deformation that
        # collapses or folds coordinates there; retain this frame globally.
        ux, uy = result[0]-global_shift[0], result[1]-global_shift[1]
        jacobian = ((1+np.gradient(ux, axis=1))*(1+np.gradient(uy, axis=0))
                    - np.gradient(ux, axis=0)*np.gradient(uy, axis=1))
        if not np.isfinite(jacobian).all() or jacobian.min() < .25:
            return tuple(np.full(self.shape, value) for value in global_shift)
        return tuple(result)

    def _cuda_costs(self, image):
        import torch
        img = torch.as_tensor(image, device='cuda:0', dtype=torch.float64)
        w, m = (self.window, self.max_shift)
        tile = img.unfold(0, w + 2 * m, self.step).unfold(1, w + 2 * m, self.step)
        tile = tile.contiguous().reshape(-1, w + 2 * m, w + 2 * m)
        weight = torch.as_tensor(self.weight, device='cuda:0', dtype=torch.float64)
        ref = torch.as_tensor(self.templates, device='cuda:0', dtype=torch.float64)
        strength = torch.as_tensor(self.strength, device='cuda:0', dtype=torch.float64)
        costs = []
        for start in range(0, len(tile), 8):
            patches = tile[start:start + 8].unfold(1, w, 1).unfold(2, w, 1)
            mean = (patches * weight).sum(dim=(-1, -2))
            variance = torch.clamp((patches.square() * weight).sum(dim=(-1, -2)) - mean.square(), min=0)
            numerator = (patches * weight * ref[start:start + 8, None, None]).sum(dim=(-1, -2))
            denom = torch.sqrt(variance) * strength[start:start + 8, None, None]
            costs.append((numerator / denom.clamp(min=1e-15)).cpu().numpy())
        costs = np.concatenate(costs)
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
    return pull(template, origin)


def cpu_backproject(raw, shift, color):
    """Resample original colour measurements and their support exactly once."""
    from planetrecon.detector import is_bayer, cfa_labels, bilinear_demosaic
    raw = np.asarray(raw, dtype=np.float64)
    support = pull(np.ones(raw.shape[:2]), shift)
    if is_bayer(color):
        masks = (cfa_labels(*raw.shape, color)[..., None] == np.array(list('RGB'))).astype(float)
        return (pull(raw[..., None]*masks, shift), pull(masks, shift),
                pull(bilinear_demosaic(raw, color), shift), support)
    return pull(raw, shift), np.broadcast_to(support[..., None], raw.shape) if raw.ndim == 3 else support, None, None
