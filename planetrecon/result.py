"""Floating linear reconstruction result and provenance."""

from __future__ import annotations

from dataclasses import dataclass, field
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
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
    layer_coverage: dict[str, np.ndarray] = field(default_factory=dict)
    schema_name: str = C.RESULT_SCHEMA
    schema_version: str = C.RESULT_SCHEMA_VERSION
    spatial_stride: int = 1

    def metadata(self) -> dict[str, Any]:
        """Detached, JSON-ready description; image arrays are stored separately."""
        return deepcopy({key: getattr(self, key) for key in self.__dataclass_fields__
                         if key not in ("image", "coverage", "validity", "layer_coverage")}
                        | {"layer_names": sorted(self.layer_coverage)})

    def copy_preview(self, max_side: int = 512) -> ReconstructionResult:
        img = np.asarray(self.image, dtype=np.float64)
        cov = np.asarray(self.coverage, dtype=np.float64)
        vis = np.asarray(self.validity)
        h, w = img.shape[:2]
        scale = max(h, w) / float(max_side) if max(h, w) > max_side else 1.0
        layers = {}
        step = 1
        if scale > 1.0:
            step = int(np.ceil(scale))
            sl = (slice(None, None, step), slice(None, None, step))
            img = img[sl]
            cov = cov[sl]
            vis = vis[sl]
            for key, arr in self.layer_coverage.items():
                layers[key] = np.array(np.asarray(arr)[sl], copy=True)
        else:
            for key, arr in self.layer_coverage.items():
                layers[key] = np.array(arr, copy=True)
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
            provenance=deepcopy(self.provenance),
            warnings=list(self.warnings),
            layer_coverage=layers,
            spatial_stride=self.spatial_stride * step,
        )


def save_snapshot(path: Path, result: ReconstructionResult) -> None:
    """Atomically publish a full-resolution scientific snapshot, never a display."""
    path = Path(path)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(stream, image=result.image, coverage=result.coverage,
                                validity=result.validity,
                                metadata=json.dumps(result.metadata(), sort_keys=True, allow_nan=False),
                                # Keep the previous CLI array keys for analysis scripts.
                                units=result.units, reference_epoch=result.reference_epoch or "",
                                provenance=json.dumps(result.provenance, sort_keys=True, allow_nan=False),
                                **{f"layer_coverage__{k}": v for k, v in result.layer_coverage.items()})
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def load_snapshot(path: Path) -> ReconstructionResult:
    """Read a self-describing snapshot or new full-resolution worker checkpoint.

    Legacy files lack completion/provenance fields and cannot safely be guessed.
    Export does not require the original source or current algorithm settings.
    """
    with np.load(path, allow_pickle=False) as data:
        if "metadata" not in data:
            raise ValueError("snapshot lacks result metadata; regenerate with the current stack command")
        meta = json.loads(str(data["metadata"]))
        if meta.get("schema") == C.JOB_SCHEMA:
            if meta.get("schema_version") != C.JOB_SCHEMA_VERSION or "result" not in meta:
                raise ValueError("checkpoint lacks supported export metadata")
            meta = meta["result"]
        if meta.get("schema_name") != C.RESULT_SCHEMA or meta.get("schema_version") != C.RESULT_SCHEMA_VERSION:
            raise ValueError("unsupported result schema")
        names = meta.pop("layer_names")
        return ReconstructionResult(**meta, **{k: data[k] for k in ("image", "coverage", "validity")},
                                    layer_coverage={k: data[f"layer_coverage__{k}"] for k in names})
