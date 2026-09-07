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
from planetrecon.pipeline.colour import complete_bayer_rgb


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
    resume_from=None,
    state_checkpoint=None,
) -> ReconstructionResult:
    from contextlib import nullcontext
    from planetrecon.backends.memory import cuda_allocation_limit
    from planetrecon.memory import cpu_memory_limit

    context = (cuda_allocation_limit(config.max_vram_bytes)
               if config.device != 'cpu' and config.geometry_mode == 'none' else nullcontext(None))
    with cpu_memory_limit(config.max_ram_bytes) as cpu_report, context as memory_report:
        def event(result, info):
            if cpu_report is not None:
                result.provenance['cpu_memory_budget'] = cpu_report
            if on_event is not None:
                on_event(result, info)
        result = _stack_source(source, config, calibration=calibration, on_event=event if on_event else None,
                             should_cancel=should_cancel, resume_from=resume_from,
                             state_checkpoint=state_checkpoint, memory_report=memory_report)
        if cpu_report is not None:
            result.provenance['cpu_memory_budget'] = cpu_report
        return result


def _stack_source(
    source: FrameSource,
    config: ReconstructionConfig,
    *,
    calibration: Calibration | None = None,
    on_event: PreviewFn | None = None,
    should_cancel: CancelFn | None = None,
    resume_from=None,
    state_checkpoint=None,
    memory_report=None,
) -> ReconstructionResult:
    if resume_from is not None or state_checkpoint is not None:
        from planetrecon import resume
        if state_checkpoint is not None:
            resume.validate_destination(state_checkpoint,source,config)
    if any(getattr(config, key) is not None for key in (
            "bias_path", "dark_path", "flat_path", "gain_e_per_adu", "read_noise_e", "saturate_adu")):
        if calibration is not None:
            raise ValueError("use either calibration settings or an explicit Calibration, not both")
        calibration = load_calibration(config, source.frame_shape())
    if source.color_mode() not in ('mono', 'RGB', 'BGR', 'RGGB', 'GRBG', 'GBRG', 'BGGR'):
        raise ValueError(f'unsupported reconstruction color mode {source.color_mode()!r}')
    if source.metadata().units == 'e-' and calibration and calibration.gain_e_per_adu is not None:
        raise ValueError('gain calibration cannot be applied to observations already in electrons')
    selection = None
    if config.frame_preselection:
        from planetrecon.pipeline.preprocess import screen_source
        n = source.n_frames()
        shape = source.frame_shape()
        if n < 1 or min(shape[:2]) < 5:
            raise ValueError('stack requires frames of at least 5 by 5 pixels')
        if config.reference_index >= n:
            raise ValueError('reference_index is outside the capture')
        rgb = is_bayer(source.color_mode()) or len(shape) == 3
        out_shape = (*shape[:2], 3) if rgb else shape[:2]
        empty = np.zeros(out_shape, dtype=np.float64)
        progress_result = ReconstructionResult(
            image=empty, coverage=empty.copy(), validity=empty > 0,
            units='e-' if calibration and calibration.gain_e_per_adu is not None else source.metadata().units,
            channel_order='RGB' if rgb else 'mono', backend='cpu', precision='float64',
            stage='preprocessing', incomplete=True,
            provenance=capture_provenance(source, config, calibration),
        )
        def progress(processed, total):
            if on_event is not None:
                on_event(progress_result, {'n_processed': processed, 'n_total': total,
                                          'n_used': 0, 'backend': 'cpu'})
        progress(0, n)
        selection = screen_source(source, config, calibration, should_cancel=should_cancel,
                                  on_progress=progress)
        progress_result.provenance['preprocessing'] = selection.summary
        if selection.cancelled:
            return progress_result
        if not selection.accepted.any():
            raise ValueError('no usable frames remain after preprocessing; a complete visible planet is required '
                             '(disable frame preselection for surface-detail crops)')
        if config.reference_index and not selection.accepted[config.reference_index]:
            raise ValueError('selected reference frame was rejected by preprocessing; choose an accepted frame or automatic reference 0')
        from planetrecon.geometry.discovery import discover_geometry
        if on_event is not None:
            on_event(progress_result, {'n_processed': n, 'n_total': n, 'n_used': 0,
                                      'backend': 'cpu', 'phase': 'orientation and rotation'})
        try:
            estimate = discover_geometry(source, config, selection, calibration, should_cancel)
        except InterruptedError:
            return progress_result
        selection.summary['geometry_estimate'] = estimate
        if on_event is not None:
            on_event(progress_result, {'n_processed': n, 'n_total': n, 'n_used': 0,
                                      'backend': 'cpu', 'phase': 'orientation and rotation',
                                      'geometry_estimate': estimate})
        del progress_result, empty
    if config.geometry_mode != "none":
        from planetrecon.pipeline.geometry_stack import stack_source_geometry

        return stack_source_geometry(
            source,
            config,
            calibration=calibration,
            on_event=on_event,
            should_cancel=should_cancel,
            resume_from=resume_from,
            state_checkpoint=state_checkpoint,
            selection=selection,
        )
    budget_error = memory_report and memory_report['error']
    backend, report = select_backend('cpu' if budget_error else config.device, threads=config.threads)
    if budget_error:
        report.requested, report.fallback, report.reason = config.device, True, budget_error
    report.execution_history = [backend.name]
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
    if "warnings" in meta.extras:
        warnings.extend(meta.extras["warnings"].value)
    if report.fallback:
        warnings.append(report.reason)
    bit_depth = meta.bit_depth
    seq = 0
    cancelled = False

    snapshot_provenance = capture_provenance(source, config, calibration)
    if selection is not None:
        snapshot_provenance['preprocessing'] = selection.summary
    snapshot_provenance["registration"] = "Gaussian 1.5px amplitude correlation with subpixel peak fit"
    if memory_report is not None:
        snapshot_provenance['cuda_allocation_budget'] = memory_report
    next_index = 0
    state_identity = None
    if resume_from is not None or state_checkpoint is not None:
        state_identity = resume.identity(source, config, calibration, should_cancel)
    if resume_from is not None:
        restored = resume.load(resume_from, state_identity, accum.shape, n, bayer)
        accum, weight = restored["accum"], restored["weight"]
        reference, reference_index = restored["reference"], restored["reference_index"]
        n_used, n_rejected = restored["n_used"], restored["n_rejected"]
        demosaic_accum, demosaic_weight = restored["demosaic_accum"], restored["demosaic_weight"]
        next_index = restored["next_index"]
        history = restored["execution_history"]
        report.execution_history = history + ([] if history[-1] == backend.name else [backend.name])
        warnings = list(dict.fromkeys(restored["warnings"] + warnings))
        snapshot_provenance["resumed_from_frame"] = next_index

    def cpu_fallback(exc):
        nonlocal backend
        if backend.name != "cuda":
            raise exc
        from planetrecon.backends.cpu import CPUBackend
        message = f"CUDA operation failed; continued on CPU with prior sums retained: {type(exc).__name__}: {exc}"
        backend = CPUBackend(threads=config.threads)
        report.selected, report.fallback, report.reason = "cpu", True, message
        report.execution_history.append("cpu")
        warnings.append(message)

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
        complete_bayer_rgb(result)
        seq += 1
        on_event(result, {"seq": seq, "n_used": n_used, "n_processed": n_used + n_rejected, "n_total": n, "backend": backend.name})

    for indices, batch in source.iter_batches(config.batch_frames, start=next_index, should_cancel=should_cancel):
        if should_cancel is not None and should_cancel():
            cancelled = True
            break
        for local, index in enumerate(indices):
            if should_cancel is not None and should_cancel():
                cancelled = True
                break
            raw = batch[local]
            if selection is not None and not selection.accepted[index]:
                n_rejected += 1
                continue
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
                try:
                    shift = backend.phase_correlation(reference, plane)
                except RuntimeError as exc:
                    cpu_fallback(exc)
                    shift = backend.phase_correlation(reference, plane)
            if not np.all(np.isfinite(shift)) or abs(shift[0]) > config.max_shift_px or abs(shift[1]) > config.max_shift_px:
                n_rejected += 1
                continue
            score = max(selection.measurements[index, 0] if selection is not None else laplacian_score(plane), 1e-12)
            if not np.isfinite(score):
                n_rejected += 1
                continue
            projected = None
            if backend.name == "cuda":
                try:
                    projected = backend.backproject(calibrated, shift, color)
                except RuntimeError as exc:
                    cpu_fallback(exc)
            if projected is not None:
                add, wt, demo, support = projected
                accum += score * add
                weight += score * wt
                if bayer:
                    demosaic_accum += score * demo
                    demosaic_weight += score * support[..., None]
                n_used += 1
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
        if state_checkpoint is not None:
            resume.save(state_checkpoint, state_identity, {
                "accum": accum, "weight": weight, "reference": reference,
                "demosaic_accum": demosaic_accum, "demosaic_weight": demosaic_weight,
                "reference_index": reference_index, "n_used": n_used, "n_rejected": n_rejected,
                "next_index": n_used + n_rejected,
                "execution_history": report.execution_history, "warnings": warnings})
        emit("baseline", incomplete=True)
        if cancelled:
            break

    cancelled = cancelled or bool(should_cancel is not None and should_cancel())
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
    complete_bayer_rgb(result)
    emit(result.stage, incomplete=result.incomplete)
    return result


def _normalise_stack(accum: np.ndarray, weight: np.ndarray) -> np.ndarray:
    return np.divide(accum, weight, out=np.zeros_like(accum), where=weight > 0)
