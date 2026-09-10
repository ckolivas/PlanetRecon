"""Exposed-ring translation with optional known field-rotation compensation."""

import numpy as np
from scipy.ndimage import binary_erosion, gaussian_filter
from planetrecon.pipeline.masked_align import MaskedRegistration


class RingRegistration(MaskedRegistration):
    """Masked normalized correlation; supplied field rotation, no moon tracks."""

    def __init__(self, reference, cx, cy, globe_radius, ring_radius, *, valid_mask=None):
        reference = np.asarray(reference, dtype=float)
        self.reference = reference
        self.geometry = (cx, cy, globe_radius, ring_radius)
        self.shape = reference.shape
        y, x = np.indices(self.shape)
        radius = np.hypot(x + .5 - cx, y + .5 - cy)
        # Keep the smoothing footprint (and Bayer proxy interpolation) clear
        # of globe/ring mixtures. Include the outer ring edge and nearby sky.
        self.mask = (radius > globe_radius + 8) & (radius < ring_radius + 8)
        self.mask[:7] = self.mask[-7:] = False
        self.mask[:, :7] = self.mask[:, -7:] = False
        if valid_mask is not None:
            self.mask &= valid_mask
        super().__init__(reference, self.mask)

    def displacement(self, frame, *, min_improvement=0.):
        # Resolved wide rings produce broad correlation shoulders. Only
        # separate maxima compete with the fitted peak; periodic/flat ring
        # patterns must still fail the existing uniqueness/contrast guards.
        return super().displacement(frame, min_improvement=min_improvement,
                                     distinct_peaks=True)

    def displacement_at_pose(self, frame, pose, reference_pose):
        """Follow field rotation, gating drift against measured interpolation loss."""
        from planetrecon.geometry.model import FieldOnlyModel, render_observed
        field = FieldOnlyModel()
        predicted = render_observed(self.reference, field, pose, reference_pose)
        roundtrip = render_observed(predicted, field, reference_pose, pose)
        values = gaussian_filter(roundtrip, 1.5)[self.mask]
        values -= values.mean() if values.size else 0.
        norm = np.sqrt(self.energy*np.sum(values**2))
        if norm <= 1e-12:
            return None
        loss = max(1e-12, 1.-float(np.dot(self.template[self.mask], values)/norm))
        support = render_observed(np.ones(self.shape), field, pose, reference_pose)
        # Rotation may expose pixels outside the original capture. Neither
        # those zeros nor their smoothing footprint are a reference observation.
        supported = binary_erosion(support >= 1.-1e-12, structure=np.ones((15, 15), bool))
        matcher = RingRegistration(predicted, pose.cx, pose.cy, *self.geometry[2:],
                                   valid_mask=supported)
        return matcher.displacement(frame, min_improvement=loss)
