"""Translation baseline: quality-weighted stacking and raw CFA RGB backprojection."""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.ndimage import shift as ndshift

from planetrecon import constants as C
from planetrecon.backends.base import Backend, select_backend
from planetrecon.calibration import Calibration, apply_calibration, load_calibration
from planetrecon.detector import (
    bilinear_demosaic,
    cfa_accumulate,
    extract_green_proxy,
    is_bayer,
)
from planetrecon.io.source import FrameSource
from planetrecon.rank import laplacian_score
from planetrecon.reconstruction import ReconstructionConfig
from planetrecon.result import ReconstructionResult
from planetrecon.pipeline.provenance import capture_provenance


PreviewFn = Callable[[ReconstructionResult, dict], None]
CancelFn = Callable[[], bool]


def _alignment_plane(frame: np.ndarray, color_mode: str) -> np.ndarray:
    if frame.ndim == 3 and frame.shape[-1] == 3:
        return 0.5 * frame[..., 1] + 0.25 * frame[..., 0] + 0.25 * frame[..., 2]
    if is_bayer(color_mode):
        return extract_green_proxy(frame, color_mode)
    return np.asarray(frame, dtype=np.float64)


def _saturated(frame: np.ndarray, bit_depth: int) -> bool:
    if bit_depth > 16:  # floating observations have no integer ADC ceiling
        return False
    peak = (1 << min(int(bit_depth), 16)) - 1
    frac = float(np.mean(np.asarray(frame) >= peak))
    return bool(frac > 0.05)


def stack_source(
    source: FrameSource,
    config: ReconstructionConfig,
    *,
    calibration: Calibration | None = None,
    on_event: PreviewFn | None = None,
    should_cancel: CancelFn | None = None,
) -> ReconstructionResult:
    if any(getattr(config, key) is not None for key in (
            "bias_path", "dark_path", "flat_path", "gain_e_per_adu", "read_noise_e", "saturate_adu")):
        if calibration is not None:
            raise ValueError("use either calibration settings or an explicit Calibration, not both")
        calibration = load_calibration(config, source.frame_shape())
    if config.geometry_mode != "none":
        from planetrecon.pipeline.geometry_stack import stack_source_geometry

        return stack_source_geometry(
            source,
            config,
            calibration=calibration,
            on_event=on_event,
            should_cancel=should_cancel,
        )
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

    reference = None
    reference_index = None
    if config.reference_index:
        raw_ref = source.read_raw(config.reference_index)
        calibrated_ref, ref_info = apply_calibration(raw_ref, calibration)
        if (not np.all(np.isfinite(calibrated_ref)) or
                (config.reject_saturated and (_saturated(raw_ref, meta.bit_depth) or ref_info["saturated"]))):
            raise ValueError("selected reference frame is invalid or saturated")
        reference = _alignment_plane(calibrated_ref, color)
        reference_index = config.reference_index
    n_used = 0
    n_rejected = 0
    warnings = list(report.warnings)
    if report.fallback:
        warnings.append(report.reason)
    bit_depth = meta.bit_depth
    seq = 0
    cancelled = False

    snapshot_provenance = capture_provenance(source, config, calibration)

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
            n_used=n_used,
            n_rejected=n_rejected,
            provenance={
                **snapshot_provenance,
                "device_report": report.__dict__,
                "color_mode": color,
                "baseline_operator_version": C.BASELINE_OPERATOR_VERSION,
                "source": meta.as_dict(),
                "n_captured": n,
                "reference_index": reference_index,
            },
            warnings=list(warnings),
        )
        seq += 1
        on_event(result, {"seq": seq, "n_used": n_used, "n_processed": n_used + n_rejected, "n_total": n, "backend": backend.name})

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
            if reference is None:
                reference = plane
                reference_index = int(index)
                shift = (0.0, 0.0)
            else:
                shift = backend.phase_correlation(reference, plane)
            if not np.all(np.isfinite(shift)) or abs(shift[0]) > config.max_shift_px or abs(shift[1]) > config.max_shift_px:
                n_rejected += 1
                continue
            score = max(laplacian_score(plane), 1e-12)
            if not np.isfinite(score):
                n_rejected += 1
                continue
            support = ndshift(np.ones((h, w)), shift=(-shift[1], -shift[0]),
                              order=1, prefilter=False, mode="grid-constant")
            if bayer:
                rgb_add, rgb_w = cfa_accumulate(calibrated, shift, color)
                accum += score * rgb_add
                weight += score * rgb_w
                demo = bilinear_demosaic(calibrated, color)
                shifted = np.stack(
                    [
                        ndshift(demo[..., c], shift=(-shift[1], -shift[0]), order=1, prefilter=False, mode="grid-constant")
                        for c in range(3)
                    ],
                    axis=2,
                )
                demosaic_accum += score * shifted
                demosaic_weight += score * support[..., None]
            elif rgb:
                img = calibrated
                if img.ndim == 2:
                    raise ValueError("RGB source produced a 2-D frame")
                for c in range(3):
                    accum[..., c] += score * ndshift(
                        img[..., c], shift=(-shift[1], -shift[0]), order=1, prefilter=False, mode="grid-constant"
                    )
                    weight[..., c] += score * support
            else:
                accum += score * ndshift(
                    calibrated, shift=(-shift[1], -shift[0]), order=1, prefilter=False, mode="grid-constant"
                )
                weight += score * support
            n_used += 1
        emit("baseline", incomplete=True)
        if cancelled:
            break

    image = _normalise_stack(accum, weight)
    cov = weight
    provenance = {
        **snapshot_provenance,
        "device_report": report.__dict__,
        "color_mode": color,
        "baseline_operator_version": C.BASELINE_OPERATOR_VERSION,
        "source": meta.as_dict(),
        "n_captured": n,
        "reference_index": reference_index,
        "config": config.to_dict(),
        "calibration_mode": (calibration.mode if calibration else "approximate-noise"),
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
        coverage=cov,
        validity=cov > 0,
        units=units,
        channel_order=channel_order,
        backend=backend.name,
        precision=backend.precision,
        stage="final" if not cancelled else "baseline",
        incomplete=cancelled or n_used == 0,
        n_used=n_used,
        n_rejected=n_rejected,
        provenance=provenance,
        warnings=warnings,
    )
    emit(result.stage, incomplete=result.incomplete)
    return result


def _normalise_stack(accum: np.ndarray, weight: np.ndarray) -> np.ndarray:
    return np.divide(accum, weight, out=np.zeros_like(accum), where=weight > 0)
