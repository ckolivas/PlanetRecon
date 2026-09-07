"""Immutable reconstruction job configuration."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
import math
from typing import Any, Literal

from planetrecon import constants as C
from planetrecon.provenance import sha256_json
from planetrecon.runtime import default_thread_count


DeviceChoice = Literal["cpu", "auto", "gpu"]
GeometryMode = Literal["none", "field", "surface", "combined", "saturn"]


@dataclass(frozen=True)
class ReconstructionConfig:
    schema_name: str = C.CONFIG_SCHEMA
    schema_version: str = C.CONFIG_SCHEMA_VERSION
    device: DeviceChoice = "auto"
    threads: int = field(default_factory=default_thread_count)
    batch_frames: int = 32
    max_ram_bytes: int | None = None
    max_vram_bytes: int | None = None
    bayer_override: str | None = None
    endian_override: str | None = None
    endian_convention: str = "ecosystem"
    recover_complete_frames: bool = False
    reject_saturated: bool = True
    frame_preselection: bool = True
    max_shift_px: float = 32.0
    reference_index: int = 0
    crop: str = "feature"
    baseline_operator_version: str = C.BASELINE_OPERATOR_VERSION
    geometry_operator_version: str = C.GEOMETRY_OPERATOR_VERSION
    geometry_mode: GeometryMode = "none"
    reference_epoch_s: float = 0.0
    field_angle0_rad: float = 0.0
    field_rate_rad_s: float | None = None
    field_center_x: float | None = None
    field_center_y: float | None = None
    equatorial_radius_px: float | None = None
    flattening: float = 0.0
    pole_pa_rad: float = 0.0
    sub_obs_lat_rad: float | None = None
    sub_obs_lon0_rad: float = 0.0
    surface_rate_rad_s: float | None = None
    exposure_s: float = 0.0
    cadence_s: float | None = None
    geometry_duration_warn_s: float = C.GEOMETRY_DURATION_WARN_S
    freeze_mid_exposure: bool = True
    ring_inner_radius_px: float | None = None
    ring_outer_radius_px: float | None = None
    ring_transmission: float = 0.35
    sun_lon_rad: float | None = None
    sun_lat_rad: float | None = None
    moon_x: float | None = None
    moon_y: float | None = None
    moon_radius_px: float | None = None
    moon_vx_px_s: float = 0.0
    moon_vy_px_s: float = 0.0
    bias_path: str | None = None
    dark_path: str | None = None
    flat_path: str | None = None
    gain_e_per_adu: float | None = None
    read_noise_e: float | None = None
    saturate_adu: float | None = None

    def __post_init__(self) -> None:
        if type(self.frame_preselection) is not bool:
            raise ValueError('frame_preselection must be a bool')
        for name in ("bias_path", "dark_path", "flat_path"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} must be a nonempty filename or null")
        for name in ("gain_e_per_adu", "read_noise_e", "saturate_adu"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 0 or (name != "read_noise_e" and value == 0)):
                raise ValueError(f"invalid {name}")
        if self.schema_name != C.CONFIG_SCHEMA or self.schema_version != C.CONFIG_SCHEMA_VERSION:
            raise ValueError("unsupported config schema or schema_version")
        if self.baseline_operator_version != C.BASELINE_OPERATOR_VERSION:
            raise ValueError("unsupported baseline operator version")
        if self.geometry_operator_version != C.GEOMETRY_OPERATOR_VERSION:
            raise ValueError("unsupported geometry operator version")
        if self.device not in ("cpu", "auto", "gpu"):
            raise ValueError("unknown device")
        for name in ("threads", "batch_frames", "reference_index"):
            value = getattr(self, name)
            if type(value) is not int or value < (0 if name == "reference_index" else 1):
                raise ValueError(f"invalid {name}")
        if not math.isfinite(self.max_shift_px) or self.max_shift_px < 0:
            raise ValueError("max_shift_px must be finite and non-negative")
        if self.bayer_override not in (None, "mono", "RGGB", "GRBG", "GBRG", "BGGR"):
            raise ValueError("unknown Bayer override")
        if self.endian_override not in (None, "little", "big"):
            raise ValueError("unknown endian override")
        if self.endian_convention not in ("ecosystem", "spec"):
            raise ValueError("unknown endian convention")
        if self.crop not in ("feature", "bland"):
            raise ValueError("unknown crop")
        if self.geometry_mode not in ("none", "field", "surface", "combined", "saturn"):
            raise ValueError("unknown geometry_mode")
        if type(self.freeze_mid_exposure) is not bool:
            raise ValueError("freeze_mid_exposure must be a bool")
        for name in (
            "reference_epoch_s",
            "field_angle0_rad",
            "flattening",
            "pole_pa_rad",
            "sub_obs_lon0_rad",
            "exposure_s",
            "geometry_duration_warn_s",
        ):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if float(self.flattening) < 0.0 or float(self.flattening) >= 1.0:
            raise ValueError("flattening must be in [0, 1)")
        if float(self.exposure_s) < 0.0 or float(self.geometry_duration_warn_s) < 0.0:
            raise ValueError("exposure and duration warning must be non-negative")
        for name in (
            "sub_obs_lat_rad",
            "field_rate_rad_s",
            "field_center_x",
            "field_center_y",
            "equatorial_radius_px",
            "surface_rate_rad_s",
            "cadence_s",
        ):
            value = getattr(self, name)
            if value is None:
                continue
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite or null")
        if self.equatorial_radius_px is not None and float(self.equatorial_radius_px) <= 0:
            raise ValueError("equatorial_radius_px must be positive")
        if self.cadence_s is not None and float(self.cadence_s) <= 0:
            raise ValueError("cadence_s must be positive")
        if not isinstance(self.ring_transmission, (int, float)) or isinstance(self.ring_transmission, bool):
            raise ValueError("ring_transmission must be finite")
        if not math.isfinite(float(self.ring_transmission)) or not (0.0 <= float(self.ring_transmission) <= 1.0):
            raise ValueError("ring_transmission must be in [0, 1]")
        for name in ("moon_vx_px_s", "moon_vy_px_s"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        for name in (
            "ring_inner_radius_px",
            "ring_outer_radius_px",
            "sun_lon_rad",
            "sun_lat_rad",
            "moon_x",
            "moon_y",
            "moon_radius_px",
        ):
            value = getattr(self, name)
            if value is None:
                continue
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite or null")
        if self.ring_inner_radius_px is not None and float(self.ring_inner_radius_px) <= 0:
            raise ValueError("ring_inner_radius_px must be positive")
        if self.ring_outer_radius_px is not None and float(self.ring_outer_radius_px) <= 0:
            raise ValueError("ring_outer_radius_px must be positive")
        if (
            self.ring_inner_radius_px is not None
            and self.ring_outer_radius_px is not None
            and float(self.ring_outer_radius_px) <= float(self.ring_inner_radius_px)
        ):
            raise ValueError("ring outer radius must exceed inner radius")
        if self.moon_radius_px is not None and float(self.moon_radius_px) <= 0:
            raise ValueError("moon_radius_px must be positive")
        if self.sun_lat_rad is not None and abs(float(self.sun_lat_rad)) > math.pi / 2:
            raise ValueError("sun_lat_rad must be in [-pi/2, pi/2]")
        moon_fields = (self.moon_x, self.moon_y, self.moon_radius_px)
        if any(v is not None for v in moon_fields) and not all(v is not None for v in moon_fields):
            raise ValueError("moon_x, moon_y and moon_radius_px must be set together")
        if self.geometry_mode != "saturn" and (
            self.ring_inner_radius_px is not None or self.ring_outer_radius_px is not None
        ):
            raise ValueError("ring radii require geometry_mode='saturn'")
        if self.geometry_mode != "saturn" and (
            any(v is not None for v in moon_fields) or self.sun_lon_rad is not None
            or self.sun_lat_rad is not None or self.ring_transmission != 0.35
            or self.moon_vx_px_s != 0 or self.moon_vy_px_s != 0
        ):
            raise ValueError("ring, Sun and moon settings require geometry_mode='saturn'")
        if self.moon_x is None and (self.moon_vx_px_s != 0 or self.moon_vy_px_s != 0):
            raise ValueError("moon velocity requires a complete moon track")
        if self.sub_obs_lat_rad is not None and abs(self.sub_obs_lat_rad) > math.pi / 2:
            raise ValueError("sub_obs_lat_rad must be in [-pi/2, pi/2]")
        if self.geometry_mode != "none" and self.exposure_s > 0 and not self.freeze_mid_exposure:
            raise ValueError("exposure quadrature is not supported; use freeze_mid_exposure")
        if self.max_ram_bytes is not None:
            import sys
            if type(self.max_ram_bytes) is not int or self.max_ram_bytes <= 0:
                raise ValueError('max_ram_bytes must be a positive integer or null')
            if self.device != 'cpu' or sys.platform != 'linux':
                raise ValueError('CPU process memory limits require explicit CPU processing on Linux')
        if self.max_vram_bytes is not None:
            if type(self.max_vram_bytes) is not int or self.max_vram_bytes <= 0:
                raise ValueError("max_vram_bytes must be a positive integer or null")
            if self.device == 'cpu' or self.geometry_mode != 'none':
                raise ValueError("CUDA allocation budgets require auto/gpu translation processing")

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
        if set(data) - known:
            raise ValueError(f"unknown config fields: {sorted(set(data) - known)}")
        return cls(**data)
