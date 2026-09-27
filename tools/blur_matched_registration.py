"""Diagnostic reference-blur matching with fixed AP geometry and raw samples.

Estimate additional reference blur at the supplied global pose, or at the
existing coherent fit. Only matching templates change; output pixels do not.
"""
import numpy as np
from scipy.ndimage import gaussian_filter
import torch

from planetrecon.pipeline.local_align import LocalRegistration
from tools.validated_registration import ValidatedRegistration


def choose_blur(templates, observed, mask):
    """Discrete reference blur with score-only positive gain and offset."""
    if int(mask.sum()) < 64:
        return 0, []
    y = observed[mask]
    y = y-y.mean()
    x = templates[:, mask]
    x = x-x.mean(1, keepdim=True)
    energy = x.square().mean(1).clamp_min(1e-30)
    cross = (x*y).mean(1)
    gain = (cross/energy).clamp(.5, 2.)
    losses = (y.square().mean()-2*gain*cross+gain.square()*energy).clamp_min(0)
    return int(losses.argmin()), losses.cpu().tolist()


class BlurMatchedRegistration(ValidatedRegistration):
    def __init__(self, reference, matcher, **kwargs):
        super().__init__(reference, matcher, **kwargs)
        self.raw_reference = np.asarray(reference, dtype=float)
        # Keep centres, eligibility, spline design, spatial kernels and support
        # fixed. This isolates template appearance from changes to AP placement.
        self.template_states = {0.: (self.engine.reference,
            [(layer['templates'], layer['strength']) for layer in self.engine.layers])}
        self.active_sigma = 0.

    def set_template(self, sigma):
        sigma = float(sigma)
        if not np.isfinite(sigma) or sigma < 0 or sigma > 2:
            raise ValueError('reference blur must be between zero and two pixels')
        if sigma not in self.template_states:
            proxy = torch.as_tensor(LocalRegistration.proxy(gaussian_filter(self.raw_reference, sigma)),
                                    device=self.device)
            states = []
            for layer in self.engine.layers:
                patch = proxy[layer['y'][:, None, None]+layer['py'],
                              layer['x'][:, None, None]+layer['px']]
                centred = patch-(patch*layer['weight']).sum((-2, -1))[:, None, None]
                strength = (centred.square()*layer['weight']).sum((-2, -1)).sqrt()
                states.append((centred, strength))
            self.template_states[sigma] = proxy, states
        self.engine.reference, states = self.template_states[sigma]
        for layer, (templates, strength) in zip(self.engine.layers, states, strict=True):
            layer['templates'], layer['strength'] = templates, strength
        self.active_sigma = sigma

    def fit(self, proxy, shift, sigma, should_cancel):
        self.set_template(sigma)
        return self.engine.displacement(proxy, shift, should_cancel)

    def variants(self, frame, shift, should_cancel, known_sigma=None):
        proxy = LocalRegistration.proxy(frame)
        observed = torch.as_tensor(proxy, device=self.device)
        origin = torch.as_tensor(shift, device=self.device, dtype=torch.float64)[:, None, None]
        sensor_x, sensor_y = self.xx-origin[0], self.yy-origin[1]
        from planetrecon.backends.torch_circular import sample
        mask = sample(self.mask, sensor_y, sensor_x) > .5
        baseline = self.fit(proxy, shift, 0., should_cancel)
        fits = {'coherent': self.engine.stats[-1]}
        global_index, global_losses = choose_blur(self.render(sensor_x, sensor_y), observed, mask)
        if bool(mask.any()):
            tx, ty, error = self.inverse(baseline, mask)
        else:
            tx, ty, error = sensor_x, sensor_y, float('inf')
        local_index, local_losses = (choose_blur(self.render(tx, ty), observed, mask)
                                     if error <= .02 else (global_index, global_losses))
        fields = {'coherent': baseline}
        cached = {0.: baseline}
        for name, sigma in [('matched_global', self.sigmas[global_index]),
                            ('matched_local', self.sigmas[local_index])]:
            if sigma not in cached:
                cached[sigma] = self.fit(proxy, shift, sigma, should_cancel)
                fits[str(float(sigma))] = self.engine.stats[-1]
            fields[name] = cached[sigma]
        if known_sigma is not None:
            # Exact known-blur diagnostic; bound the cache in synthetic runs.
            fields['known_blur'] = (cached[known_sigma] if known_sigma in cached else
                                   self.fit(proxy, shift, known_sigma, should_cancel))
            if known_sigma not in self.sigmas:
                self.template_states.pop(float(known_sigma), None)
        stats = dict(global_sigma=float(self.sigmas[global_index]),
                     local_sigma=float(self.sigmas[local_index]),
                     global_losses=global_losses, local_losses=local_losses,
                     inverse_error_px=error if np.isfinite(error) else None, fits=fits)
        self.stats.append(stats)
        return fields
