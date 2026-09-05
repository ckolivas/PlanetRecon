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
from planetrecon.geometry.rings import RingParams
from planetrecon.geometry.saturn import LAYER_GLOBE, LAYER_FAR_RING, LAYER_NEAR_RING, MoonTrack, SaturnSceneModel, fit_saturn_geometry
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


def _warn_duration_and_exposure(
    times: np.ndarray, config: ReconstructionConfig, radius: float,
    field_rate: float, surface_rate: float,
) -> list[str]:
    warnings: list[str] = []
    if times.size:
        span = float(times[-1] - times[0])
        if span > float(config.geometry_duration_warn_s):
            warnings.append(
                "rigid_rotation_duration_limit: "
                f"clip {span:.3f}s exceeds {config.geometry_duration_warn_s:.3f}s"
            )
    rates = 0.0
    if config.geometry_mode in ("field", "combined"):
        rates += abs(field_rate)
    if config.geometry_mode in ("surface", "combined"):
        rates += abs(surface_rate)
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
    *,
    sample_indices: list[int] | None = None,
) -> tuple[list[FramePose], SceneModel, dict, list[str]]:
    times, time_origin = source_times_s(source, cadence_s=config.cadence_s)
    if times.size == 0:
        raise ValueError("geometry requires at least one frame")
    active_rates = []
    if config.geometry_mode in ("field", "combined", "saturn"):
        active_rates.append(config.field_rate_rad_s)
    if config.geometry_mode in ("surface", "combined", "saturn"):
        active_rates.append(config.surface_rate_rad_s)
    if time_origin == "inferred" and (
        any(rate is not None and rate != 0 for rate in active_rates)
        or config.exposure_s > 0 or config.reference_epoch_s != 0
    ):
        raise ValueError("geometry in seconds requires measured timestamps or an explicit cadence_s")
    if planes is None:
        color = source.color_mode()
        sample_indices = sorted({0, source.n_frames() // 2, source.n_frames() - 1, config.reference_index})
        planes = [_alignment_plane(np.asarray(source.read_raw(i), dtype=np.float64), color) for i in sample_indices]
    if sample_indices is None:
        sample_indices = list(range(len(planes)))
    if not planes or len(sample_indices) != len(planes):
        raise ValueError("geometry planes require matching sample_indices")
    indices = np.asarray(sample_indices)
    if (not np.issubdtype(indices.dtype, np.integer) or np.any(indices < 0)
            or np.any(indices >= len(times)) or np.any(np.diff(indices) <= 0)):
        raise ValueError("sample_indices must be unique increasing frame indices")
    if not all(np.all(np.isfinite(plane)) for plane in planes):
        raise ValueError("geometry sample planes must be finite")
    anchor = sample_indices.index(config.reference_index) if config.reference_index in sample_indices else 0
    saturn_fit = None
    if config.geometry_mode == "saturn":
        saturn_fit = fit_saturn_geometry(planes[anchor])
        disc = saturn_fit
        degeneracy = list(saturn_fit.get("degeneracy") or ())
    else:
        disc = fit_disc_ellipse(planes[anchor])
        degeneracy = list(sequence_degeneracy(planes))
    cx = config.field_center_x
    cy = config.field_center_y
    radius = config.equatorial_radius_px
    centre_origin = "user"
    if radius is None and not disc["ok"] and cx is not None and cy is not None and config.geometry_mode == "field":
        h, w = source.frame_shape()[:2]
        radius = float(np.hypot(h, w) / 2)
    if cx is None or cy is None or radius is None:
        if not disc["ok"]:
            raise ValueError("geometry requires a disc centre/radius or a fitted disc")
        cx = float(disc["cx"] if cx is None else cx)
        cy = float(disc["cy"] if cy is None else cy)
        radius = float(disc["radius"] if radius is None else radius)
        centre_origin = "inferred" if config.field_center_x is None or config.field_center_y is None else "user"
        if config.field_center_x is None or config.field_center_y is None:
            degeneracy = list(dict.fromkeys([*degeneracy, *disc["degeneracy"]]))
    field_rate = config.field_rate_rad_s
    field_origin = "user"
    angles = None
    if field_rate is None and config.geometry_mode in ("field", "combined", "saturn"):
        field_origin = "inferred"
        if len(planes) >= 2:
            estimates = [estimate_field_angle(planes[0], plane, cx, cy, radius) for plane in planes]
            for estimate in estimates:
                degeneracy.extend(estimate["degeneracy"])
            angles = unwrap_angles([estimate["angle_rad"] for estimate in estimates])
            dt = np.diff(times[indices])
            usable = np.array([not estimate["degeneracy"] for estimate in estimates])
            pairs = usable[:-1] & usable[1:]
            if np.any(pairs):
                field_rate = float(np.median(np.diff(angles)[pairs] / dt[pairs]))
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
    if config.geometry_mode in ("surface", "combined", "saturn") and config.surface_rate_rad_s is None:
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
    if config.geometry_mode in ("surface", "combined", "saturn"):
        globe = globe_for_config(
            radius,
            config.flattening,
            config.pole_pa_rad,
            config.sub_obs_lat_rad,
            config.sub_obs_lon0_rad,
            surface_rate,
            config.reference_epoch_s,
        )
    rings = None
    moon = None
    if config.geometry_mode == "saturn":
        inner = config.ring_inner_radius_px
        outer = config.ring_outer_radius_px
        if inner is None and saturn_fit is not None:
            inner = saturn_fit.get("ring_inner")
        if outer is None and saturn_fit is not None:
            outer = saturn_fit.get("ring_outer")
        if inner is None or outer is None:
            raise ValueError("saturn geometry requires ring inner/outer radii or a fitted ring annulus")
        rings = RingParams(
            inner_radius_px=float(inner),
            outer_radius_px=float(outer),
            transmission=float(config.ring_transmission),
            sun_lon_rad=config.sun_lon_rad,
            sun_lat_rad=config.sun_lat_rad,
        )
        if config.moon_x is not None:
            moon = MoonTrack(
                x=float(config.moon_x),
                y=float(config.moon_y),
                radius_px=float(config.moon_radius_px),
                vx_px_s=float(config.moon_vx_px_s),
                vy_px_s=float(config.moon_vy_px_s),
            )
    model = select_scene_model(
        config.geometry_mode, globe, rings=rings, moon=moon,
        field_angle0_rad=config.field_angle0_rad,
    )
    diagnostics = {
        "time_origin": time_origin,
        "time_unit": "frame" if time_origin == "inferred" else "s",
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
        "reference_index": int(sample_indices[anchor]),
        "sample_indices": [int(i) for i in sample_indices],
        "sample_times_s": [float(t) for t in times[indices]],
        "reference_epoch_s": float(config.reference_epoch_s),
        "estimated_angles_rad": None if angles is None else [float(a) for a in angles],
        "geometry_operator_version": C.GEOMETRY_OPERATOR_VERSION,
        "geometry_mode": config.geometry_mode,
        "rings": None if rings is None else {
            "inner_radius_px": rings.inner_radius_px,
            "outer_radius_px": rings.outer_radius_px,
            "transmission": rings.transmission,
        },
        "edge_on_rings": bool(getattr(model, "edge_on", False)),
        "low_opening": bool(getattr(model, "low_opening", False)),
    }
    warnings = (_warn_duration_and_exposure(times, config, radius, float(field_rate), surface_rate)
                if time_origin != "inferred" else [])
    if time_origin == "inferred":
        warnings.append("cadence_unknown: relative field motion uses frame indices; physical seconds are unmeasured")
    if "roll_unconstrained" in degeneracy_t and config.geometry_mode in ("field", "combined", "saturn"):
        warnings.append("roll_unconstrained: near-circular or featureless disc cannot constrain field angle")
    if getattr(model, "edge_on", False):
        warnings.append("edge_on_rings: ring plane is degenerate; ring samples are masked")
    elif getattr(model, "low_opening", False):
        warnings.append("low_opening: globe/ring overlap is conservatively masked")
    if "spin_unconstrained" in degeneracy_t and config.geometry_mode in ("surface", "combined", "saturn"):
        if config.surface_rate_rad_s is None:
            warnings.append("spin_unconstrained: surface rate was not supplied and was not estimated")
        else:
            warnings.append("spin_unconstrained: image texture cannot constrain spin; using supplied surface rate")
    return poses, model, diagnostics, warnings


def stack_source_geometry(
    source: FrameSource,
    config: ReconstructionConfig,
    *,
    calibration: Calibration | None = None,
    on_event: PreviewFn | None = None,
    should_cancel: CancelFn | None = None,
) -> ReconstructionResult:
    # All geometry operators currently execute in NumPy float64.
    backend, report = select_backend("cpu", threads=config.threads)
    report.requested = config.device
    if config.device != "cpu":
        report.fallback = True
        report.reason = "geometry_cpu_only: geometry operators currently run on CPU float64"
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
    globe_weight = np.zeros((h, w), dtype=np.float64)
    ring_weight = np.zeros((h, w), dtype=np.float64)

    def cancelled_before_geometry():
        result = ReconstructionResult(
            image=accum.copy(), coverage=weight.copy(), validity=weight > 0,
            units=units, channel_order=channel_order, backend=backend.name,
            precision=backend.precision, stage="baseline", incomplete=True,
            reference_epoch=str(config.reference_epoch_s),
            provenance={"device_report": report.__dict__, "source": meta.as_dict(),
                        "config": config.to_dict(), "geometry_operator_version": C.GEOMETRY_OPERATOR_VERSION},
            warnings=[report.reason] if report.fallback else [],
        )
        if on_event is not None:
            on_event(result, {"seq": 1, "n_used": 0, "n_processed": 0, "n_total": n, "backend": backend.name})
        return result

    def usable_frame(raw):
        if not np.all(np.isfinite(raw)) or (config.reject_saturated and _saturated(raw, meta.bit_depth)):
            return None
        calibrated, info = apply_calibration(raw, calibration)
        if not np.all(np.isfinite(calibrated)) or (config.reject_saturated and info["saturated"]):
            return None
        return calibrated

    sample_idx = []
    sample_planes = []
    candidates = sorted({0, n // 2, n - 1, config.reference_index})
    for index in candidates:
        if should_cancel is not None and should_cancel():
            return cancelled_before_geometry()
        frame = usable_frame(source.read_raw(index))
        if frame is None:
            if index == config.reference_index and config.reference_index != 0:
                raise ValueError("selected reference frame is invalid or saturated")
            continue
        sample_idx.append(index)
        sample_planes.append(_alignment_plane(frame, color))
    if not sample_planes:
        for index in range(n):
            if should_cancel is not None and should_cancel():
                return cancelled_before_geometry()
            if index in candidates:
                continue
            frame = usable_frame(source.read_raw(index))
            if frame is not None:
                sample_idx.append(index)
                sample_planes.append(_alignment_plane(frame, color))
                break
    if not sample_planes:
        raise ValueError("no usable frames remain for geometry estimation")
    poses, model, diagnostics, geo_warnings = prepare_geometry(
        source, config, planes=sample_planes, sample_indices=sample_idx,
    )
    ref_pose = _reference_pose(poses, config)
    xg, yg = detector_xy_grids(h, w)
    n_used = 0
    n_rejected = 0
    warnings = list(report.warnings) + list(geo_warnings)
    if report.fallback:
        warnings.append(report.reason)
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
            layer_coverage={"globe": globe_weight.copy(), "ring": ring_weight.copy()}
            if isinstance(model, SaturnSceneModel) else {},
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
            calibrated = usable_frame(batch[local])
            if calibrated is None:
                n_rejected += 1
                continue
            plane = _alignment_plane(calibrated, color)
            pose = poses[int(index)]
            layer_labels = None
            if isinstance(model, SaturnSceneModel):
                layer_labels = model.classify_detector(xg, yg, pose)["labels"]
                globe_pix = layer_labels == LAYER_GLOBE
                if np.any(globe_pix):
                    plane_score = np.where(globe_pix, plane, float(np.median(plane)))
                else:
                    plane_score = plane
                score = max(laplacian_score(plane_score), 1e-12)
            else:
                score = max(laplacian_score(plane), 1e-12)
            if not np.isfinite(score):
                n_rejected += 1
                continue
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
            if layer_labels is not None:
                ones = np.ones((h, w), dtype=np.float64)
                _ga, gw = bilinear_push(
                    ones, y_idx, x_idx, (h, w),
                    valid=valid & (layer_labels == LAYER_GLOBE),
                )
                _ra, rw = bilinear_push(
                    ones, y_idx, x_idx, (h, w),
                    valid=valid & ((layer_labels == LAYER_NEAR_RING) | (layer_labels == LAYER_FAR_RING)),
                )
                globe_weight += score * gw
                ring_weight += score * rw
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
        "reference_index": diagnostics["reference_index"],
        "config": config.to_dict(),
        "calibration_mode": (calibration.mode if calibration else "approximate-noise"),
        "geometry": diagnostics,
        "layers": {
            "globe_coverage_mean": float(np.mean(globe_weight)),
            "ring_coverage_mean": float(np.mean(ring_weight)),
        },
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
        layer_coverage={"globe": globe_weight, "ring": ring_weight}
        if isinstance(model, SaturnSceneModel) else {},
    )
    emit(result.stage, incomplete=result.incomplete)
    return result


def _reference_pose(poses: list[FramePose], config: ReconstructionConfig) -> FramePose:
    target = float(config.reference_epoch_s)
    # The output epoch is exact; only observed exposures use their midpoints.
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
