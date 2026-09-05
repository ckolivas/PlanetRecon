"""Immutable reconstruction job configuration."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Literal

from planetrecon import constants as C
from planetrecon.provenance import sha256_json
from planetrecon.runtime import DEFAULT_CPU_THREADS


DeviceChoice = Literal["cpu", "auto", "gpu"]


@dataclass(frozen=True)
class ReconstructionConfig:
    schema_name: str = C.CONFIG_SCHEMA
    schema_version: str = C.CONFIG_SCHEMA_VERSION
    device: DeviceChoice = "auto"
    threads: int = DEFAULT_CPU_THREADS
    batch_frames: int = 32
    max_ram_bytes: int | None = None
    max_vram_bytes: int | None = None
    bayer_override: str | None = None
    endian_override: str | None = None
    endian_convention: str = "ecosystem"
    recover_complete_frames: bool = False
    reject_saturated: bool = True
    max_shift_px: float = 32.0
    reference_index: int = 0
    crop: str = "feature"
    baseline_operator_version: str = C.BASELINE_OPERATOR_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    def cache_key(self, source_identity: str) -> str:
        return sha256_json(
            {
                "config": self.to_dict(),
                "source": source_identity,
                "operator": self.baseline_operator_version,
            }
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReconstructionConfig:
        if data.get("schema_name") not in (None, C.CONFIG_SCHEMA):
            raise ValueError(f"unsupported config schema {data.get('schema_name')!r}")
        version = data.get("schema_version", C.CONFIG_SCHEMA_VERSION)
        if version != C.CONFIG_SCHEMA_VERSION:
            raise ValueError(f"unsupported config schema_version {version!r}")
        known = set(cls.__dataclass_fields__)
        payload = {k: v for k, v in data.items() if k in known}
        return cls(**payload)
