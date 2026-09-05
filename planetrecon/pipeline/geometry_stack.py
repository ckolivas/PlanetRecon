"""Geometry-aware baseline: field and/or rigid globe accumulation with coverage."""

from __future__ import annotations

from typing import Callable

import numpy as np

from planetrecon import constants as C
from planetrecon.backends.base import select_backend
from planetrecon.calibration import Calibration, apply_calibration
from planetrecon.detector import bilinear_demosaic, cfa_labels, channel_mask, extract_green_proxy, is_bayer
from planetrecon.geometry.coords import detector_xy_grids
from planetrecon.geometry.fit import estimate_field_angle, fit_disc_ellipse, sequence_degeneracy
from planetrecon.geometry.model import SceneModel, select_scene_model
from planetrecon.geometry.pose import FramePose, build_frame_poses, globe_for_config, source_times_s, unwrap_angles
from planetrecon.geometry.warp import bilinear_push
from planetrecon.io.source import FrameSource
from planetrecon.rank import laplacian_score
from planetrecon.reconstruction import ReconstructionConfig
from planetrecon.result import ReconstructionResult


PreviewFn = Callable[[ReconstructionResult, dict], None]
CancelFn = Callable[[], bool]


def _alignment_plane(frame: np.ndarray, color_mode: str) -> np.ndarray:
    if frame.ndim == 3 and frame.shape[-1] == 3:
        return 0.5 * frame[..., 1] + 0.25 * frame[..., 0] + 0.25 * frame[..., 2]
    if is_bayer(color_mode):
        return extract_green_proxy(frame, color_mode)
    return np.asarray(frame, dtype=np.float64)


def _saturated(frame: np.ndarray, bit_depth: int) -> bool:
    if bit_depth > 16:
        return False
    peak = (1 << min(int(bit_depth), 16)) - 1
    frac = float(np.mean(np.asarray(frame) >= peak))
    return bool(frac > 0.05)


def _normalise_stack(accum: np.ndarray, weight: np.ndarray) -> np.ndarray:
    return np.divide(accum, weight, out=np.zeros_like(accum), where=weight > 0)


def _warn_duration_and_exposure(times: np.ndarray, config: ReconstructionConfig, radius: float) -> list[str]:
    warnings: list[str] = []
    if times.size:
        span = float(times[-1] - times[0])
        if span > float(config.geometry_duration_warn_s):
            warnings.append(
                "rigid_rotation_duration_limit: "
                f"clip {span:.3f}s exceeds {config.geometry_duration_warn_s:.3f}s"
            )
    rates = 0.0
    if config.field_rate_rad_s:
        rates += abs(float(config.field_rate_rad_s))
    if config.surface_rate_rad_s:
        rates += abs(float(config.surface_rate_rad_s))
    motion = rates * float(radius) * float(config.exposure_s)
    if motion > C.GEOMETRY_EXPOSURE_MOTION_PX and config.freeze_mid_exposure:
        warnings.append(
            "exposure_geometry_frozen: "
            f"limb motion {motion:.3f}px during exposure exceeds {C.GEOMETRY_EXPOSURE_MOTION_PX}"
        )
    return warnings


def prepare_geometry(
    source: FrameSource,
    config: ReconstructionConfig,
    planes: list[np.ndarray] | None = None,
) -> tuple[list[FramePose], SceneModel, dict, list[str]]:
    times, time_origin = source_times_s(source, cadence_s=config.cadence_s)
    shape = source.frame_shape()
    h, w = int(shape[0]), int(shape[1])
    if planes is None:
        color = source.color_mode()
        planes = [_alignment_plane(np.asarray(source.read_raw(0), dtype=np.float64), color)]
    disc = fit_disc_ellipse(planes[0])
    degeneracy = list(sequence_degeneracy(planes))
    cx = config.field_center_x
    cy = config.field_center_y
    radius = config.equatorial_radius_px
    centre_origin = "user"
    if cx is None or cy is None or radius is None:
        if not disc["ok"]:
            raise ValueError("geometry requires a disc centre/radius or a fitted disc")
        cx = float(disc["cx"] if cx is None else cx)
        cy = float(disc["cy"] if cy is None else cy)
        radius = float(disc["radius"] if radius is None else radius)
        centre_origin = "inferred" if config.field_center_x is None else "user"
        if config.field_center_x is None or config.field_center_y is None:
            degeneracy = list(dict.fromkeys([*degeneracy, *disc["degeneracy"]]))
    field_rate = config.field_rate_rad_s
    field_origin = "user"
    angles = None
    if field_rate is None and config.geometry_mode in ("field", "combined"):
        field_origin = "inferred"
        if len(planes) >= 2:
            est = [
                estimate_field_angle(planes[0], planes[i], cx, cy, radius)["angle_rad"]
                for i in range(len(planes))
            ]
            angles = unwrap_angles(est)
            dt = np.diff(times[: len(angles)])
            if np.any(np.abs(dt) > 1e-12) and len(angles) >= 2:
                field_rate = float(np.median(np.diff(angles) / dt))
            else:
                field_rate = 0.0
                degeneracy.append("roll_unconstrained")
        else:
            field_rate = 0.0
            degeneracy.append("roll_unconstrained")
    if field_rate is None:
        field_rate = 0.0
    surface_rate = 0.0 if config.surface_rate_rad_s is None else float(config.surface_rate_rad_s)
    surface_origin = "user" if config.surface_rate_rad_s is not None else "inferred"
    if config.geometry_mode in ("surface", "combined") and config.surface_rate_rad_s is None:
        degeneracy.append("spin_unconstrained")
        surface_origin = "inferred"
    degeneracy_t = tuple(dict.fromkeys(degeneracy))
    poses = build_frame_poses(
        times,
        cx=cx,
        cy=cy,
        field_angle0_rad=config.field_angle0_rad,
        field_rate_rad_s=float(field_rate),
        reference_epoch_s=config.reference_epoch_s,
        field_origin=field_origin,
        surface_origin=surface_origin,
        freeze_mid_exposure=config.freeze_mid_exposure,
        exposure_s=config.exposure_s,
        degeneracy=degeneracy_t,
    )
    globe = None
    if config.geometry_mode in ("surface", "combined"):
        globe = globe_for_config(
            radius,
            config.flattening,
            config.pole_pa_rad,
            config.sub_obs_lat_rad,
            config.sub_obs_lon0_rad,
            surface_rate,
            config.reference_epoch_s,
        )
    model = select_scene_model(config.geometry_mode, globe)
    diagnostics = {
        "time_origin": time_origin,
        "centre_origin": centre_origin,
        "field_origin": field_origin,
        "surface_origin": surface_origin,
        "cx": cx,
        "cy": cy,
        "radius": radius,
        "field_rate_rad_s": float(field_rate),
        "surface_rate_rad_s": surface_rate,
        "degeneracy": list(degeneracy_t),
        "disc": {k: (v if not isinstance(v, tuple) else list(v)) for k, v in disc.items()},
        "estimated_angles_rad": None if angles is None else [float(a) for a in angles],
        "geometry_operator_version": C.GEOMETRY_OPERATOR_VERSION,
        "geometry_mode": config.geometry_mode,
    }
    warnings = _warn_duration_and_exposure(times, config, radius)
    if "roll_unconstrained" in degeneracy_t:
        warnings.append("roll_unconstrained: near-circular or featureless disc cannot constrain field angle")
    if "spin_unconstrained" in degeneracy_t:
        warnings.append("spin_unconstrained: surface rate was not supplied and was not estimated")
    return poses, model, diagnostics, warnings


def stack_source_geometry(
    source: FrameSource,
    config: ReconstructionConfig,
    *,
    calibration: Calibration | None = None,
    on_event: PreviewFn | None = None,
    should_cancel: CancelFn | None = None,
) -> ReconstructionResult:
    backend, report = select_backend(config.device, threads=config.threads)
    meta = source.metadata()
    color = source.color_mode()
    n = source.n_frames()
    shape = source.frame_shape()
    h, w = int(shape[0]), int(shape[1])
    if color not in ("mono", "RGB", "BGR", "RGGB", "GRBG", "GBRG", "BGGR"):
        raise ValueError(f"unsupported reconstruction color mode {color!r}")
    if n < 1 or h < 5 or w < 5:
        raise ValueError("stack requires frames of at least 5 by 5 pixels")
    if config.reference_index >= n:
        raise ValueError("reference_index is outside the capture")
    units = "e-" if calibration and calibration.gain_e_per_adu is not None else meta.units
    if meta.units == "e-" and calibration and calibration.gain_e_per_adu is not None:
        raise ValueError("gain calibration cannot be applied to observations already in electrons")
    bayer = is_bayer(color)
    rgb = color in ("RGB", "BGR") or bayer
    if rgb:
        accum = np.zeros((h, w, 3), dtype=np.float64)
        weight = np.zeros((h, w, 3), dtype=np.float64)
        channel_order = "RGB"
    else:
        accum = np.zeros((h, w), dtype=np.float64)
        weight = np.zeros((h, w), dtype=np.float64)
        channel_order = "mono"
    demosaic_accum = np.zeros((h, w, 3), dtype=np.float64) if bayer else None
    demosaic_weight = np.zeros((h, w, 3), dtype=np.float64) if bayer else None

    sample_idx = []
    for i in (0, n // 2, n - 1, int(config.reference_index)):
        if 0 <= i < n and i not in sample_idx:
            sample_idx.append(i)
    sample_planes = [
        _alignment_plane(np.asarray(source.read_raw(i), dtype=np.float64), color) for i in sample_idx
    ]
    poses, model, diagnostics, geo_warnings = prepare_geometry(source, config, planes=sample_planes)
    ref_pose = _reference_pose(poses, config)
    xg, yg = detector_xy_grids(h, w)
    n_used = 0
    n_rejected = 0
    warnings = list(report.warnings) + list(geo_warnings)
    if report.fallback:
        warnings.append(report.reason)
    bit_depth = meta.bit_depth
    seq = 0
    cancelled = False

    def emit(stage: str, incomplete: bool) -> None:
        nonlocal seq
        if on_event is None:
            return
        cov = weight.copy()
        image = _normalise_stack(accum, weight)
        result = ReconstructionResult(
            image=image,
            coverage=cov,
            validity=cov > 0,
            units=units,
            channel_order=channel_order,
            backend=backend.name,
            precision=backend.precision,
            stage=stage,
            incomplete=incomplete,
            reference_epoch=str(config.reference_epoch_s),
            n_used=n_used,
            n_rejected=n_rejected,
            provenance={
                "device_report": report.__dict__,
                "color_mode": color,
                "baseline_operator_version": C.BASELINE_OPERATOR_VERSION,
                "geometry_operator_version": C.GEOMETRY_OPERATOR_VERSION,
                "source": meta.as_dict(),
                "geometry": diagnostics,
            },
            warnings=list(warnings),
        )
        seq += 1
        on_event(
            result,
            {"seq": seq, "n_used": n_used, "n_processed": n_used + n_rejected, "n_total": n, "backend": backend.name},
        )

    for indices, batch in source.iter_batches(config.batch_frames):
        if should_cancel is not None and should_cancel():
            cancelled = True
            break
        for local, index in enumerate(indices):
            if should_cancel is not None and should_cancel():
                cancelled = True
                break
            raw = batch[local]
            if not np.all(np.isfinite(raw)) or (config.reject_saturated and _saturated(raw, bit_depth)):
                n_rejected += 1
                continue
            calibrated, cal_info = apply_calibration(raw, calibration)
            if not np.all(np.isfinite(calibrated)) or (config.reject_saturated and cal_info["saturated"]):
                n_rejected += 1
                continue
            plane = _alignment_plane(calibrated, color)
            score = max(laplacian_score(plane), 1e-12)
            if not np.isfinite(score):
                n_rejected += 1
                continue
            pose = poses[int(index)]
            xd, yd, valid = model.src_to_ref(xg, yg, pose, ref_pose)
            y_idx = yd - 0.5
            x_idx = xd - 0.5
            if bayer:
                labels = cfa_labels(h, w, color)
                rgb_add = np.zeros((h, w, 3), dtype=np.float64)
                rgb_w = np.zeros((h, w, 3), dtype=np.float64)
                for name, idx in (("R", 0), ("G", 1), ("B", 2)):
                    mask = channel_mask(labels, name)
                    a, wt = bilinear_push(
                        calibrated,
                        y_idx,
                        x_idx,
                        (h, w),
                        valid=valid & mask,
                    )
                    rgb_add[..., idx] = a
                    rgb_w[..., idx] = wt
                accum += score * rgb_add
                weight += score * rgb_w
                demo = bilinear_demosaic(calibrated, color)
                da, dw = bilinear_push(demo, y_idx, x_idx, (h, w), valid=valid)
                demosaic_accum += score * da
                demosaic_weight += score * dw
            elif rgb:
                if calibrated.ndim == 2:
                    raise ValueError("RGB source produced a 2-D frame")
                a, wt = bilinear_push(calibrated, y_idx, x_idx, (h, w), valid=valid)
                accum += score * a
                weight += score * wt
            else:
                a, wt = bilinear_push(calibrated, y_idx, x_idx, (h, w), valid=valid)
                accum += score * a
                weight += score * wt
            n_used += 1
        emit("baseline", incomplete=True)
        if cancelled:
            break

    image = _normalise_stack(accum, weight)
    provenance = {
        "device_report": report.__dict__,
        "color_mode": color,
        "baseline_operator_version": C.BASELINE_OPERATOR_VERSION,
        "geometry_operator_version": C.GEOMETRY_OPERATOR_VERSION,
        "source": meta.as_dict(),
        "n_captured": n,
        "reference_index": int(config.reference_index),
        "config": config.to_dict(),
        "calibration_mode": (calibration.mode if calibration else "approximate-noise"),
        "geometry": diagnostics,
    }
    if bayer and demosaic_accum is not None:
        provenance["demosaic_first"] = {
            "label": "comparison",
            "mean": float(np.mean(_normalise_stack(demosaic_accum, demosaic_weight))),
        }
    if n_used == 0 and not cancelled:
        raise ValueError("no usable frames remain after rejection")
    result = ReconstructionResult(
        image=image,
        coverage=weight,
        validity=weight > 0,
        units=units,
        channel_order=channel_order,
        backend=backend.name,
        precision=backend.precision,
        stage="final" if not cancelled else "baseline",
        incomplete=cancelled or n_used == 0,
        reference_epoch=str(config.reference_epoch_s),
        n_used=n_used,
        n_rejected=n_rejected,
        provenance=provenance,
        warnings=warnings,
    )
    emit(result.stage, incomplete=result.incomplete)
    return result


def _reference_pose(poses: list[FramePose], config: ReconstructionConfig) -> FramePose:
    target = float(config.reference_epoch_s)
    if config.freeze_mid_exposure and config.exposure_s:
        target = target + 0.5 * float(config.exposure_s)
    idx = int(np.argmin([abs(p.t_s - target) for p in poses]))
    p = poses[idx]
    return FramePose(
        t_s=target,
        field_angle_rad=float(config.field_angle0_rad),
        cx=p.cx,
        cy=p.cy,
        field_origin=p.field_origin,
        surface_origin=p.surface_origin,
        degeneracy=p.degeneracy,
    )


def freeze_midexposure_error(
    reference: np.ndarray,
    model: SceneModel,
    src_t0: float,
    exposure_s: float,
    ref: FramePose,
    make_pose,
    n_quad: int = 5,
) -> float:
    """Relative L2 between freeze-at-mid-exposure and a J-sample geometry quadrature."""
    from planetrecon.geometry.model import render_observed
    from planetrecon.operators import relative_l2

    mid = src_t0 + 0.5 * float(exposure_s)
    frozen = render_observed(reference, model, make_pose(mid), ref)
    acc = np.zeros_like(frozen)
    for i in range(int(n_quad)):
        t = src_t0 + (i + 0.5) * float(exposure_s) / float(n_quad)
        acc = acc + render_observed(reference, model, make_pose(t), ref)
    quad = acc / float(n_quad)
    mask = np.abs(quad) > 1e-12
    if frozen.ndim == 3:
        mask = np.any(mask, axis=2)
        return relative_l2(frozen[mask], quad[mask])
    return relative_l2(frozen, quad, mask)
