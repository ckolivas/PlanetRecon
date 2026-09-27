"""Ablate detector splitting, motion gating, and a full-data refit.

The refit uses held-out evidence to choose whether to apply local motion, but
its final full-data field is not itself independently validated. Diagnostic only.
"""
import torch

from planetrecon.pipeline.local_align import LocalRegistration
from tools.validated_registration import ValidatedRegistration


class _Capture:
    def __init__(self, engine):
        self.engine = engine
        self.fields = []

    @property
    def stats(self):
        return self.engine.stats

    def displacement(self, *args):
        field = self.engine.displacement(*args)
        self.fields.append(field)
        return field


def guarded(field, origin):
    residual = field-origin
    ux_y, ux_x = torch.gradient(residual[0])
    uy_y, uy_x = torch.gradient(residual[1])
    valid = bool(torch.isfinite(field).all() &
                 (((1+ux_x)*(1+uy_y)-ux_y*uy_x).min() >= .25) &
                 (torch.linalg.vector_norm(residual, dim=0).max() <= 6.))
    return field if valid else origin.expand_as(field).clone()


class RefitRegistration(ValidatedRegistration):
    def __init__(self, *args, policy='full_any', **kwargs):
        if policy not in ('coherent', 'half_average', 'validated', 'full_any', 'full_scaled'):
            raise ValueError('unknown refit policy')
        super().__init__(*args, **kwargs)
        self.policy = policy

    def variants(self, frame, shift, should_cancel):
        # Capture the original implementation's two fields without duplicating
        # its scoring or changing the saved baseline model's source/hash.
        engine = self.engine
        capture = _Capture(engine)
        self.engine = capture
        try:
            validated = super().displacement_raw(frame, shift, should_cancel)
        finally:
            self.engine = engine
        coherent = engine.displacement(LocalRegistration.proxy(frame), shift, should_cancel)
        origin = torch.as_tensor(shift, dtype=torch.float64, device=self.device)[:, None, None]
        stats = self.stats[-1]
        passed = stats['accepted_halves'] if stats['field_guard_accepted'] else 0
        half_average = (capture.fields[0]+capture.fields[1])*.5 if capture.fields else validated
        result = {'coherent': coherent, 'half_average': guarded(half_average, origin),
                  'validated': validated,
                  'full_any': coherent if passed else origin.expand_as(coherent).clone(),
                  'full_scaled': guarded(origin+(coherent-origin)*(passed/2.), origin)}
        stats['full_fit'] = engine.stats[-1]
        return result

    def displacement_raw(self, frame, shift, should_cancel):
        return self.variants(frame, shift, should_cancel)[self.policy]
