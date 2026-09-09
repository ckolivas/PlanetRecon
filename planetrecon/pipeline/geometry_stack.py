"""Geometry-aware baseline: field and/or rigid globe accumulation with coverage."""

from __future__ import annotations

from typing import Callable
from dataclasses import replace

import numpy as np

from planetrecon import constants as C
from planetrecon.backends.base import select_backend
from planetrecon.calibration import Calibration, apply_calibration
from planetrecon.detector import bilinear_demosaic, cfa_labels, channel_mask, extract_green_proxy, is_bayer
from planetrecon.geometry.coords import detector_xy_grids
from planetrecon.geometry.fit import estimate_field_angle, fit_disc_ellipse, sequence_degeneracy
from planetrecon.geometry.model import SceneModel, select_scene_model, render_observed
from planetrecon.geometry.pose import FramePose, build_frame_poses, globe_for_config, source_times_s, unwrap_angles
from planetrecon.geometry.rings import RingParams
from planetrecon.geometry.saturn import LAYER_GLOBE, LAYER_FAR_RING, LAYER_NEAR_RING, MoonTrack, SaturnSceneModel, fit_saturn_geometry
from planetrecon.geometry.warp import bilinear_push
from planetrecon.io.source import FrameSource
from planetrecon.rank import laplacian_score
from planetrecon.reconstruction import ReconstructionConfig
from planetrecon.result import ReconstructionResult
from planetrecon.pipeline.provenance import capture_provenance
from planetrecon.pipeline.colour import complete_bayer_rgb
from planetrecon.pipeline.align import phase_correlation_shift


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
    field_rate: float, surface_rate: float, ring_radius: float | None = None,
) -> list[str]:
    warnings: list[str] = []
    if times.size:
        span = float(times[-1] - times[0])
        if span > float(config.geometry_duration_warn_s):
            warnings.append(
                "rigid_rotation_duration_limit: "
                f"clip {span:.3f}s exceeds {config.geometry_duration_warn_s:.3f}s"
            )
    speed = 0.0
    if config.geometry_mode in ("field", "combined", "saturn"):
        speed += abs(field_rate) * max(radius, ring_radius or radius)
    if config.geometry_mode in ("surface", "combined", "saturn"):
        speed += abs(surface_rate) * radius
    if config.geometry_mode == "saturn" and config.moon_x is not None:
        speed = max(speed, float(np.hypot(config.moon_vx_px_s, config.moon_vy_px_s)))
    motion = speed * float(config.exposure_s)
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
    reference_index: int | None = None,
) -> tuple[list[FramePose], SceneModel, dict, list[str]]:
    from dataclasses import replace
    from planetrecon.geometry.pose import capture_exposure
    config.require_saturn_geometry()
    config = replace(config, exposure_s=capture_exposure(source, config.exposure_s)['value_s'])
    reference_index = config.reference_index if reference_index is None else reference_index
    times, time_origin = source_times_s(source, cadence_s=config.cadence_s)
    if times.size == 0:
        raise ValueError("geometry requires at least one frame")
    active_rates = []
    if config.geometry_mode in ("field", "combined", "saturn"):
        active_rates.append(config.field_rate_rad_s)
    if config.geometry_mode in ("surface", "combined", "saturn"):
        active_rates.append(config.surface_rate_rad_s)
    if config.geometry_mode == "saturn" and config.moon_x is not None:
        active_rates.extend([config.moon_vx_px_s, config.moon_vy_px_s])
    if time_origin == "inferred" and (
        any(rate is not None and rate != 0 for rate in active_rates)
        or config.exposure_s > 0 or config.reference_epoch_s != 0
    ):
        raise ValueError("geometry in seconds requires measured timestamps or an explicit cadence_s")
    if planes is None:
        color = source.color_mode()
        sample_indices = sorted({0, source.n_frames() // 2, source.n_frames() - 1, reference_index})
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
    anchor = sample_indices.index(reference_index) if reference_index in sample_indices else 0
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
    field_sample_shifts = None
    if field_rate is None and config.geometry_mode in ("field", "combined", "saturn"):
        field_origin = "inferred"
        if len(planes) >= 2:
            from scipy.ndimage import shift
            # Polar angles must be measured about the same planet centre.
            # Camera drift otherwise appears as rotation (even with no spin).
            # Resample these estimation proxies only; raw accumulation still
            # combines the fitted motion and translation in a single warp.
            angle_planes = []
            field_sample_shifts = []
            registered = []
            angle_radius = radius
            annulus = np.ones(planes[anchor].shape, dtype=bool)
            if config.geometry_mode == 'saturn':
                xp, yp = detector_xy_grids(*planes[anchor].shape)
                annulus = np.hypot(xp-cx, yp-cy) > radius + 1.
                angle_radius = config.ring_outer_radius_px
            angular_reference = np.where(annulus, planes[anchor], 0.)
            from planetrecon.geometry.model import FieldOnlyModel
            reference_pose = FramePose(0., 0., cx, cy)
            for sample, plane in enumerate(planes):
                aligned = plane
                dx = dy = 0.
                good = True
                if sample != anchor:
                    from planetrecon.geometry.fit import estimate_field_angle as fit_angle
                    angle = fit_angle(angular_reference, np.where(annulus, plane, 0.),
                                      cx, cy, angle_radius)['angle_rad']
                    # Alternate angle and displacement about the configured
                    # centre, rather than letting translation absorb rotation.
                    for _ in range(4):
                        predicted = render_observed(planes[anchor], FieldOnlyModel(),
                            FramePose(0., angle, cx, cy), reference_pose)
                        if config.geometry_mode == 'saturn':
                            from planetrecon.pipeline.ring_align import RingRegistration
                            displacement = RingRegistration(predicted, cx, cy, radius,
                                angle_radius).displacement(plane)
                            if displacement is None:
                                break
                            dx, dy = displacement
                        else:
                            dx, dy = phase_correlation_shift(predicted, plane)
                        good = (np.isfinite([dx, dy]).all() and abs(dx) <= config.max_shift_px
                                and abs(dy) <= config.max_shift_px)
                        if not good:
                            break
                        aligned = shift(plane, (-dy, -dx), order=1, mode='constant',
                                        cval=float(np.median(plane)), prefilter=False)
                        angle = fit_angle(angular_reference, np.where(annulus, aligned, 0.),
                                          cx, cy, angle_radius)['angle_rad']
                registered.append(good)
                field_sample_shifts.append([float(dx), float(dy)] if good else None)
                angle_planes.append(np.where(annulus, aligned, 0.))
            estimates = [estimate_field_angle(angle_planes[anchor], plane, cx, cy, angle_radius)
                         for plane in angle_planes]
            for estimate in estimates:
                degeneracy.extend(estimate["degeneracy"])
            angles = unwrap_angles([estimate["angle_rad"] for estimate in estimates])
            dt = np.diff(times[indices])
            usable = np.array([good and not estimate["degeneracy"]
                               for good, estimate in zip(registered, estimates)])
            pairs = usable[:-1] & usable[1:] & (dt > 0)
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
            config.sub_obs_lat_rad if config.sub_obs_lat_rad is not None else 0.0,
            config.sub_obs_lon0_rad,
            surface_rate,
            config.reference_epoch_s,
        )
    rings = None
    moon = None
    if config.geometry_mode == "saturn":
        inner = config.ring_inner_radius_px
        outer = config.ring_outer_radius_px
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
        "timestamp_duplicate_intervals": int(np.count_nonzero(np.diff(times) == 0)),
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
        "field_sample_shifts_px": field_sample_shifts,
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
    warnings = (_warn_duration_and_exposure(times, config, radius, float(field_rate), surface_rate,
                                           None if rings is None else rings.outer_radius_px)
                if time_origin != "inferred" else [])
    if diagnostics['timestamp_duplicate_intervals']:
        warnings.append('timestamp_ties: frames with equal recorded timestamps are retained at the same time; zero intervals are excluded from rate estimation')
    if time_origin == "inferred":
        warnings.append("cadence_unknown: relative field motion uses frame indices; physical seconds are unmeasured")
    if "roll_unconstrained" in degeneracy_t and config.geometry_mode in ("field", "combined", "saturn"):
        warnings.append("roll_unconstrained: near-circular or featureless disc cannot constrain field angle")
    if getattr(model, "edge_on", False):
        warnings.append("edge_on_rings: ring plane is degenerate; ring samples are masked")
    if isinstance(model, SaturnSceneModel) and model.transmission > 0:
        warnings.append("mixed_ring_globe: transparent foreground-ring overlap is excluded; a joint layer solve is not implemented")
    if getattr(model, "low_opening", False):
        warnings.append("low_opening: globe/ring overlap is conservatively masked")
    if "spin_unconstrained" in degeneracy_t and config.geometry_mode in ("surface", "combined", "saturn"):
        if config.surface_rate_rad_s is None:
            warnings.append("spin_unconstrained: surface rate was not supplied and was not estimated; no surface rotation correction is applied")
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
    resume_from=None,
    state_checkpoint=None,
    selection=None,
    cache_status=None,
) -> ReconstructionResult:
    if config.frame_preselection and cache_status is None:
        from planetrecon.pipeline.baseline import stack_source
        return stack_source(source, config, calibration=calibration, on_event=on_event,
                            should_cancel=should_cancel, resume_from=resume_from,
                            state_checkpoint=state_checkpoint)
    # All geometry operators currently execute in NumPy float64.
    if state_checkpoint is not None:
        from planetrecon import resume
        resume.validate_destination(state_checkpoint, source, config)
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
        if selection is not None:
            result.provenance['preprocessing'] = selection.summary
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
    chosen_reference = (config.reference_index if config.reference_index or selection is None else
                        selection.best_reference_index)
    candidates = sorted({0, n // 2, n - 1, chosen_reference})
    if selection is not None:
        accepted_indices = np.flatnonzero(selection.accepted)
        candidates = sorted({int(accepted_indices[0]), int(accepted_indices[len(accepted_indices)//2]),
                             int(accepted_indices[-1]), chosen_reference})
    for index in candidates:
        if should_cancel is not None and should_cancel():
            return cancelled_before_geometry()
        if selection is not None and not selection.accepted[index]:
            continue
        frame = usable_frame(source.read_raw(index))
        if frame is None:
            if index == chosen_reference and (selection is not None or config.reference_index != 0):
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
            if selection is not None and not selection.accepted[index]:
                continue
            frame = usable_frame(source.read_raw(index))
            if frame is not None:
                sample_idx.append(index)
                sample_planes.append(_alignment_plane(frame, color))
                break
    if not sample_planes:
        raise ValueError("no usable frames remain for geometry estimation")
    poses, model, diagnostics, geo_warnings = prepare_geometry(
        source, config, planes=sample_planes, sample_indices=sample_idx, reference_index=chosen_reference,
    )
    ref_pose = _reference_pose(poses, config)
    anchor_index = diagnostics['reference_index']
    anchor_plane = sample_planes[sample_idx.index(anchor_index)]
    anchor_pose = poses[anchor_index]
    static_attitude = (diagnostics['surface_rate_rad_s'] == 0 or config.geometry_mode == 'field') and (
        diagnostics['field_rate_rad_s'] == 0 or config.geometry_mode == 'surface')
    # Whole-frame Saturn predictions have visibility holes during rotation.
    # Static layers can share generic translation tracking; globe spin uses
    # exposed rings below. Moon tracks retain absolute detector coordinates.
    track_translation = (not isinstance(model, SaturnSceneModel)
                         or (model.moon is None and static_attitude))
    ring_registration = None
    if (isinstance(model, SaturnSceneModel) and model.moon is None
            and not model.edge_on and diagnostics['field_rate_rad_s'] == 0
            and not static_attitude):
        from planetrecon.pipeline.ring_align import RingRegistration
        ring_registration = RingRegistration(anchor_plane, anchor_pose.cx, anchor_pose.cy,
            model.globe.equatorial_radius_px, model.rings.outer_radius_px)
    diagnostics['registration'] = ('model-predicted reference plus Gaussian 1.5px subpixel translation'
                                   if track_translation else
                                   'exposed stationary rings; unconstrained matches keep fixed centre'
                                   if ring_registration is not None else
                                   'fixed centre (Saturn detector tracks)' if model.moon is not None else
                                   'fixed centre (Saturn moving layers)')
    xg, yg = detector_xy_grids(h, w)
    target_regions = (model.reconstruction_regions(model.classify_detector(xg, yg, ref_pose, mask_moon=False))
                      if isinstance(model, SaturnSceneModel) else None)
    n_used = 0
    n_rejected = 0
    warnings = list(report.warnings) + list(geo_warnings)
    if "warnings" in meta.extras:
        warnings.extend(meta.extras["warnings"].value)
    if report.fallback:
        warnings.append(report.reason)
    seq = 0
    cancelled = False

    snapshot_provenance = capture_provenance(source, config, calibration)
    snapshot_provenance["preprocessing_cache"] = cache_status or {"status": "disabled"}
    if selection is not None:
        snapshot_provenance['preprocessing'] = selection.summary
    next_index = 0
    state_identity = None
    if resume_from is not None or state_checkpoint is not None:
        from planetrecon import resume
        import hashlib
        import json
        from dataclasses import asdict
        state_identity = resume.identity(source, config, calibration, should_cancel)
        from planetrecon.pipeline.preprocess_cache import reconstruction_digest
        state_identity["preprocessing_digest"] = reconstruction_digest(selection) if selection is not None else None
        # Re-estimation is bounded in image count. Refuse continuation if any
        # fitted geometry or per-frame timing changed, even with identical sums.
        pose_hash = hashlib.sha256()
        for pose in poses:
            if should_cancel is not None and should_cancel():
                raise InterruptedError('checkpoint geometry verification cancelled')
            pose_hash.update(json.dumps(asdict(pose), sort_keys=True).encode())
        state_identity['geometry'] = diagnostics
        state_identity['poses_sha256'] = pose_hash.hexdigest()
    if resume_from is not None:
        restored = resume.load(resume_from, state_identity, accum.shape, n, bayer, geometry=True)
        accum, weight = restored['accum'], restored['weight']
        demosaic_accum, demosaic_weight = restored['demosaic_accum'], restored['demosaic_weight']
        globe_weight, ring_weight = restored['globe_weight'], restored['ring_weight']
        n_used, n_rejected = restored['n_used'], restored['n_rejected']
        next_index = restored['next_index']
        warnings = list(dict.fromkeys(restored['warnings'] + warnings))
        snapshot_provenance['resumed_from_frame'] = next_index

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
                **snapshot_provenance,
                "device_report": report.__dict__,
                "color_mode": color,
                "baseline_operator_version": C.BASELINE_OPERATOR_VERSION,
                "geometry_operator_version": C.GEOMETRY_OPERATOR_VERSION,
                "source": meta.as_dict(),
                "n_captured": n,
                "reference_index": diagnostics["reference_index"],
                "geometry": diagnostics,
            },
            warnings=list(warnings),
            layer_coverage={"globe": globe_weight.copy(), "ring": ring_weight.copy()}
            if isinstance(model, SaturnSceneModel) else {},
        )
        complete_bayer_rgb(result, target_regions)
        seq += 1
        on_event(
            result,
            {"seq": seq, "n_used": n_used, "n_processed": n_used + n_rejected, "n_total": n, "backend": backend.name},
        )

    if config.frame_preselection:
        emit("cache_ready", incomplete=True)
    for indices, batch in source.iter_batches(config.batch_frames, start=next_index, should_cancel=should_cancel):
        if should_cancel is not None and should_cancel():
            cancelled = True
            break
        for local, index in enumerate(indices):
            if should_cancel is not None and should_cancel():
                cancelled = True
                break
            if selection is not None and not selection.accepted[index]:
                n_rejected += 1
                continue
            calibrated = usable_frame(batch[local])
            if calibrated is None:
                n_rejected += 1
                continue
            plane = _alignment_plane(calibrated, color)
            pose = poses[int(index)]
            if ring_registration is not None:
                displacement = ((0., 0.) if int(index) == anchor_index else
                                ring_registration.displacement(plane))
                if displacement is None:
                    warning = 'ring_registration_unconstrained: exposed rings cannot constrain camera drift; retaining configured centre'
                    if warning not in warnings:
                        warnings.append(warning)
                else:
                    dx, dy = displacement
                    if (not np.isfinite([dx, dy]).all() or abs(dx) > config.max_shift_px
                            or abs(dy) > config.max_shift_px):
                        n_rejected += 1
                        continue
                    pose = replace(pose, cx=pose.cx + dx, cy=pose.cy + dy)
            if track_translation:
                # Predict the anchor at this frame's time before fitting camera
                # translation, so registration does not absorb the chosen spin.
                predicted = (anchor_plane if static_attitude or int(index) == anchor_index else
                             render_observed(anchor_plane, model, pose, anchor_pose))
                dx, dy = phase_correlation_shift(predicted, plane)
                if (not np.isfinite([dx, dy]).all() or abs(dx) > config.max_shift_px
                        or abs(dy) > config.max_shift_px):
                    n_rejected += 1
                    continue
                pose = replace(pose, cx=pose.cx + dx, cy=pose.cy + dy)
            layer_labels = None
            if isinstance(model, SaturnSceneModel):
                from scipy.ndimage import binary_erosion, convolve
                from planetrecon.rank import LAPLACIAN_KERNEL

                layer_info = model.classify_detector(xg, yg, pose)
                layer_labels = layer_info["labels"]
                source_regions = model.reconstruction_regions(layer_info)
                # Evaluate real globe texture only where the whole Laplacian
                # stencil stays in one illumination region; no artificial edge.
                score_mask = (binary_erosion(source_regions == 1, iterations=2 if bayer else 1)
                              | binary_erosion(source_regions == 3, iterations=2 if bayer else 1))
                score_mask[:2] = score_mask[-2:] = False
                score_mask[:, :2] = score_mask[:, -2:] = False
                lap = convolve(plane, LAPLACIAN_KERNEL, mode="nearest")
                score = max(float(np.mean(lap[score_mask] ** 2)), 1e-12) if score_mask.any() else 1e-12
            else:
                score = max(selection.measurements[index, 0] if selection is not None else laplacian_score(plane), 1e-12)
            if not np.isfinite(score):
                n_rejected += 1
                continue
            xd, yd, valid = model.src_to_ref(xg, yg, pose, ref_pose)
            y_idx = yd - 0.5
            x_idx = xd - 0.5
            def push_samples(samples, yd, xd, shape, valid):
                if target_regions is None:
                    return bilinear_push(samples, yd, xd, shape, valid=valid)
                out_shape = (*shape, samples.shape[-1]) if samples.ndim == 3 else shape
                added = np.zeros(out_shape, dtype=np.float64)
                covered = np.zeros_like(added)
                for region in range(5):
                    a, wt = bilinear_push(samples, yd, xd, shape,
                                          valid=valid & (source_regions == region))
                    target_mask = target_regions == region
                    if samples.ndim == 3:
                        target_mask = target_mask[..., None]
                    added += np.where(target_mask, a, 0.0)
                    covered += np.where(target_mask, wt, 0.0)
                return added, covered

            if bayer:
                labels = cfa_labels(h, w, color)
                rgb_add = np.zeros((h, w, 3), dtype=np.float64)
                rgb_w = np.zeros((h, w, 3), dtype=np.float64)
                for name, idx in (("R", 0), ("G", 1), ("B", 2)):
                    mask = channel_mask(labels, name)
                    a, wt = push_samples(
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
                frame_coverage = rgb_w
                demo = bilinear_demosaic(calibrated, color)
                da, dw = push_samples(demo, y_idx, x_idx, (h, w), valid=valid)
                demosaic_accum += score * da
                demosaic_weight += score * dw
            elif rgb:
                if calibrated.ndim == 2:
                    raise ValueError("RGB source produced a 2-D frame")
                a, wt = push_samples(calibrated, y_idx, x_idx, (h, w), valid=valid)
                accum += score * a
                weight += score * wt
                frame_coverage = wt
            else:
                a, wt = push_samples(calibrated, y_idx, x_idx, (h, w), valid=valid)
                accum += score * a
                weight += score * wt
                frame_coverage = wt
            if not np.any(frame_coverage > 0):
                n_rejected += 1
                continue
            if layer_labels is not None:
                ones = np.ones((h, w), dtype=np.float64)
                _ga, gw = push_samples(
                    ones, y_idx, x_idx, (h, w),
                    valid=valid & (layer_labels == LAYER_GLOBE),
                )
                _ra, rw = push_samples(
                    ones, y_idx, x_idx, (h, w),
                    valid=valid & ((layer_labels == LAYER_NEAR_RING) | (layer_labels == LAYER_FAR_RING)),
                )
                globe_weight += score * gw
                ring_weight += score * rw
            n_used += 1
        if state_checkpoint is not None:
            resume.save(state_checkpoint, state_identity, {
                'accum': accum, 'weight': weight, 'reference': None,
                'demosaic_accum': demosaic_accum, 'demosaic_weight': demosaic_weight,
                'globe_weight': globe_weight, 'ring_weight': ring_weight,
                'reference_index': diagnostics['reference_index'],
                'n_used': n_used, 'n_rejected': n_rejected, 'next_index': n_used+n_rejected,
                'execution_history': ['cpu'], 'warnings': warnings}, geometry=True)
        emit("baseline", incomplete=True)
        if cancelled:
            break

    cancelled = cancelled or bool(should_cancel is not None and should_cancel())
    image = _normalise_stack(accum, weight)
    provenance = {
        **snapshot_provenance,
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
    complete_bayer_rgb(result, target_regions)
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
