"""Floating linear reconstruction result and provenance."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from planetrecon import constants as C


@dataclass
class ReconstructionResult:
    image: np.ndarray
    coverage: np.ndarray
    validity: np.ndarray
    units: str
    channel_order: str
    backend: str
    precision: str
    stage: str
    incomplete: bool
    reference_epoch: str | None = None
    n_used: int = 0
    n_rejected: int = 0
    provenance: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    schema_name: str = C.RESULT_SCHEMA
    schema_version: str = C.RESULT_SCHEMA_VERSION

    def copy_preview(self, max_side: int = 512) -> ReconstructionResult:
        img = np.asarray(self.image, dtype=np.float64)
        cov = np.asarray(self.coverage, dtype=np.float64)
        vis = np.asarray(self.validity)
        h, w = img.shape[:2]
        scale = max(h, w) / float(max_side) if max(h, w) > max_side else 1.0
        if scale > 1.0:
            step = int(np.ceil(scale))
            sl = (slice(None, None, step), slice(None, None, step))
            img = img[sl]
            cov = cov[sl]
            vis = vis[sl]
        return ReconstructionResult(
            image=np.array(img, copy=True),
            coverage=np.array(cov, copy=True),
            validity=np.array(vis, copy=True),
            units=self.units,
            channel_order=self.channel_order,
            backend=self.backend,
            precision=self.precision,
            stage=self.stage,
            incomplete=self.incomplete,
            reference_epoch=self.reference_epoch,
            n_used=self.n_used,
            n_rejected=self.n_rejected,
            provenance=dict(self.provenance),
            warnings=list(self.warnings),
        )
