"""Per-frame pose, time bases, angle unwrapping and reference epoch."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from planetrecon.geometry.globe import GlobeParams
from planetrecon.io.source import FrameSource


@dataclass(frozen=True)
class FramePose:
    t_s: float
    field_angle_rad: float
    cx: float
    cy: float
    field_origin: str = "user"
    surface_origin: str = "user"
    degeneracy: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("t_s", "field_angle_rad", "cx", "cy"):
            if not np.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")


def unwrap_angles(angles) -> np.ndarray:
    """Unwrap radian samples so consecutive frames do not jump by ~2π."""
    arr = np.asarray(angles, dtype=np.float64)
    if arr.size == 0:
        return arr
    return np.unwrap(arr)


def source_times_s(
    source: FrameSource,
    *,
    cadence_s: float | None = None,
    n_frames: int | None = None,
) -> tuple[np.ndarray, str]:
    """Seconds from the first sample. SER FILETIME is converted; else index*cadence."""
    n = int(source.n_frames() if n_frames is None else n_frames)
    ts = source.timestamps()
    if ts is not None:
        t = np.asarray(ts, dtype=np.float64).reshape(-1)
        if t.size != n:
            raise ValueError("timestamp count must match the frame count")
        if not np.all(np.isfinite(t)):
            raise ValueError("timestamps must be finite")
        # Windows FILETIME ticks (100 ns) are ~1e17; Unix/ns are still >> 1e12.
        if float(np.max(np.abs(t))) > 1e12:
            t = t * 1.0e-7
        return t - t[0], "measured"
    dt = 1.0 if cadence_s is None else float(cadence_s)
    if not np.isfinite(dt) or dt < 0:
        raise ValueError("cadence_s must be finite and non-negative")
    origin = "inferred" if cadence_s is None else "user"
    return np.arange(n, dtype=np.float64) * dt, origin


def build_frame_poses(
    times_s: np.ndarray,
    *,
    cx: float,
    cy: float,
    field_angle0_rad: float,
    field_rate_rad_s: float,
    reference_epoch_s: float,
    field_origin: str = "user",
    surface_origin: str = "user",
    freeze_mid_exposure: bool = True,
    exposure_s: float = 0.0,
    degeneracy: tuple[str, ...] = (),
) -> list[FramePose]:
    times = np.asarray(times_s, dtype=np.float64)
    if freeze_mid_exposure and exposure_s:
        times = times + 0.5 * float(exposure_s)
    poses = []
    for t in times:
        angle = float(field_angle0_rad) + float(field_rate_rad_s) * (float(t) - float(reference_epoch_s))
        poses.append(
            FramePose(
                t_s=float(t),
                field_angle_rad=angle,
                cx=float(cx),
                cy=float(cy),
                field_origin=field_origin,
                surface_origin=surface_origin,
                degeneracy=tuple(degeneracy),
            )
        )
    return poses


def globe_for_config(radius_px: float, flattening: float, pole_pa_rad: float,
                     sub_obs_lat_rad: float, sub_obs_lon0_rad: float,
                     surface_rate_rad_s: float, reference_epoch_s: float) -> GlobeParams:
    return GlobeParams(
        equatorial_radius_px=float(radius_px),
        flattening=float(flattening),
        pole_pa_rad=float(pole_pa_rad),
        sub_obs_lat_rad=float(sub_obs_lat_rad),
        sub_obs_lon0_rad=float(sub_obs_lon0_rad),
        surface_rate_rad_s=float(surface_rate_rad_s),
        reference_epoch_s=float(reference_epoch_s),
    )
