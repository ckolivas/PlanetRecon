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
    """Recorded nondecreasing frame times; ties retain their measured time."""
    n = int(source.n_frames() if n_frames is None else n_frames)
    if n == 0:
        return np.empty(0, dtype=np.float64), "measured" if source.timestamps() is not None else "inferred"
    ts = source.timestamps()
    if ts is not None:
        t = np.asarray(ts).reshape(-1)
        if t.size != n:
            raise ValueError("timestamp count must match the frame count")
        if not np.all(np.isfinite(t)):
            raise ValueError("timestamps must be finite")
        if np.any(t[1:] < t[:-1]):
            raise ValueError("timestamps must be nondecreasing (reversed times)")
        if n > 1 and t[-1] == t[0]:
            raise ValueError("timestamps have no positive time span")
        scale = float(source.timestamp_scale_s())
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("timestamp scale must be positive and finite")
        # Subtract integer epochs before conversion: absolute SER ticks lose
        # sub-microsecond cadence when first converted to float64.
        if np.issubdtype(t.dtype, np.integer):
            relative = np.array([int(v) - int(t[0]) for v in t], dtype=np.float64)
        else:
            relative = np.asarray(t - t[0], dtype=np.float64)
        return relative * scale, "measured"
    dt = 1.0 if cadence_s is None else float(cadence_s)
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("cadence_s must be positive and finite")
    origin = "inferred" if cadence_s is None else "user"
    return np.arange(n, dtype=np.float64) * dt, origin


def capture_timing(source: FrameSource, *, cadence_s: float | None = None) -> dict:
    """Observed first-to-last frame-start span, without inventing missing time."""
    try:
        times, origin = source_times_s(source, cadence_s=cadence_s)
    except ValueError as exc:
        return {'status': 'invalid', 'duration_s': None, 'reason': str(exc)}
    if not times.size or origin == 'inferred':
        return {'status': 'unavailable', 'duration_s': None,
                'reason': 'No frame timestamps or supplied cadence.' if times.size else 'Empty capture.'}
    intervals = np.diff(times)
    return {'status': 'available', 'origin': origin, 'duration_s': float(times[-1]-times[0]),
            'n_frames': len(times), 'median_cadence_s': float(np.median(intervals)) if intervals.size else None,
            'duplicate_intervals': int(np.count_nonzero(intervals == 0)),
            'minimum_interval_s': float(intervals.min()) if intervals.size else None,
            'maximum_interval_s': float(intervals.max()) if intervals.size else None,
            'definition': 'First-to-last frame-start timestamp; final exposure length is not included.'}


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


def capture_exposure(source: FrameSource, override_s: float | None = None) -> dict:
    """Effective integration time with an explicit distinction from cadence."""
    if override_s is not None:
        return {'value_s': float(override_s), 'origin': 'user'}
    field = source.metadata().extras.get('exposure_s')
    if field is not None and isinstance(field.value, (int, float)) and not isinstance(field.value, bool):
        value = float(field.value)
        if np.isfinite(value) and value > 0:
            return {'value_s': value, 'origin': field.origin, 'note': field.note}
    return {'value_s': 0., 'origin': 'unavailable',
            'note': 'No recorded exposure; midpoint correction uses zero. Cadence is not substituted.'}
