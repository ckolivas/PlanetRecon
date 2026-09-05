"""Validated baseline: quality-weighted registration and stacking, plus CFA joint RGB."""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.ndimage import shift as ndshift

from planetrecon import constants as C
from planetrecon.backends.base import Backend, select_backend
from planetrecon.calibration import Calibration, apply_calibration
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


PreviewFn = Callable[[ReconstructionResult, dict], None]
CancelFn = Callable[[], bool]


def _alignment_plane(frame: np.ndarray, color_mode: str) -> np.ndarray:
    if frame.ndim == 3 and frame.shape[-1] == 3:
        return 0.5 * frame[..., 1] + 0.25 * frame[..., 0] + 0.25 * frame[..., 2]
    if is_bayer(color_mode):
        return extract_green_proxy(frame, color_mode)
    return np.asarray(frame, dtype=np.float64)


def _saturated(frame: np.ndarray, bit_depth: int) -> bool:
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
    backend, report = select_backend(config.device, threads=config.threads)
    meta = source.metadata()
    color = source.color_mode()
    n = source.n_frames()
    shape = source.frame_shape()
    h, w = int(shape[0]), int(shape[1])
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
    n_used = 0
    n_rejected = 0
    warnings = list(report.warnings)
    if report.fallback:
        warnings.append(report.reason)
    bit_depth = meta.bit_depth
    seq = 0

    def emit(stage: str, incomplete: bool) -> None:
        nonlocal seq
        if on_event is None:
            return
        cov = weight if weight.ndim == 2 else np.mean(weight, axis=2)
        image = _normalise_stack(accum, weight)
        result = ReconstructionResult(
            image=image,
            coverage=cov,
            validity=cov > 0,
            units="adu",
            channel_order=channel_order,
            backend=backend.name,
            precision=backend.precision,
            stage=stage,
            incomplete=incomplete,
            n_used=n_used,
            n_rejected=n_rejected,
            provenance={
                "device_report": report.__dict__,
                "color_mode": color,
                "baseline_operator_version": C.BASELINE_OPERATOR_VERSION,
                "source": meta.as_dict(),
            },
            warnings=list(warnings),
        )
        seq += 1
        on_event(result, {"seq": seq, "n_used": n_used, "n_total": n, "backend": backend.name})

    for indices, batch in source.iter_batches(config.batch_frames):
        if should_cancel is not None and should_cancel():
            break
        for local, index in enumerate(indices):
            if should_cancel is not None and should_cancel():
                break
            raw = batch[local]
            if config.reject_saturated and _saturated(raw, bit_depth):
                n_rejected += 1
                continue
            calibrated, cal_info = apply_calibration(raw, calibration)
            plane = _alignment_plane(calibrated, color)
            if reference is None:
                reference = plane
                shift = (0.0, 0.0)
            else:
                shift = backend.phase_correlation(reference, plane)
            if abs(shift[0]) > config.max_shift_px or abs(shift[1]) > config.max_shift_px:
                n_rejected += 1
                continue
            score = max(laplacian_score(plane), 1e-12)
            if bayer:
                rgb_add, rgb_w = cfa_accumulate(calibrated, shift, color)
                accum += score * rgb_add
                weight += score * rgb_w
                demo = bilinear_demosaic(calibrated, color)
                shifted = np.stack(
                    [
                        ndshift(demo[..., c], shift=(-shift[1], -shift[0]), order=1, prefilter=False)
                        for c in range(3)
                    ],
                    axis=2,
                )
                demosaic_accum += score * shifted
                demosaic_weight += score
            elif rgb:
                img = calibrated
                if img.ndim == 2:
                    raise ValueError("RGB source produced a 2-D frame")
                for c in range(3):
                    accum[..., c] += score * ndshift(
                        img[..., c], shift=(-shift[1], -shift[0]), order=1, prefilter=False
                    )
                    weight[..., c] += score
            else:
                accum += score * ndshift(
                    calibrated, shift=(-shift[1], -shift[0]), order=1, prefilter=False
                )
                weight += score
            n_used += 1
        emit("baseline", incomplete=True)

    image = _normalise_stack(accum, weight)
    cov = weight if weight.ndim == 2 else np.mean(weight, axis=2)
    provenance = {
        "device_report": report.__dict__,
        "color_mode": color,
        "baseline_operator_version": C.BASELINE_OPERATOR_VERSION,
        "source": meta.as_dict(),
        "n_captured": n,
        "calibration_mode": (calibration.mode if calibration else "approximate-noise"),
    }
    if bayer and demosaic_accum is not None:
        provenance["demosaic_first"] = {
            "label": "comparison",
            "mean": float(np.mean(demosaic_accum / np.clip(demosaic_weight, 1e-12, None))),
        }
    cancelled = should_cancel is not None and should_cancel()
    result = ReconstructionResult(
        image=image,
        coverage=cov,
        validity=cov > 0,
        units="adu",
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
    return accum / np.clip(weight, 1e-12, None)
