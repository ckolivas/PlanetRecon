"""Translation baseline: quality-weighted stacking and raw CFA RGB backprojection."""

from __future__ import annotations

from typing import Callable
from dataclasses import replace
from functools import partial

import numpy as np
from scipy.ndimage import shift as ndshift

from planetrecon import constants as C
from planetrecon.backends.base import Backend, select_backend
from planetrecon.calibration import Calibration, apply_calibration, load_calibration
from planetrecon.detector import (
    bilinear_demosaic,
    PreparedBilinearDemosaic,
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


class SetupProgress:
    """Lightweight event for setup phases that precede the first stacked frame."""
    stage = 'setup'


def _alignment_plane(frame: np.ndarray, color_mode: str) -> np.ndarray:
    if frame.ndim == 3 and frame.shape[-1] == 3:
        return 0.5 * frame[..., 1] + 0.25 * frame[..., 0] + 0.25 * frame[..., 2]
    if is_bayer(color_mode):
        return extract_green_proxy(frame, color_mode)
    return np.asarray(frame, dtype=np.float64)


def _colour_registration_plane(frame: np.ndarray, color_mode: str, *, demosaic=None) -> np.ndarray:
    """Use every colour for motion estimates; original samples form the output."""
    if is_bayer(color_mode):
        rgb = bilinear_demosaic(frame, color_mode) if demosaic is None else demosaic(frame)
        return rgb @ np.array([.25, .5, .25])
    return _alignment_plane(frame, color_mode)


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
    preprocessing=None,
    preprocessing_cache=None,
    cache_validation=None,
) -> ReconstructionResult:
    from contextlib import nullcontext
    from planetrecon.backends.memory import cuda_allocation_limit
    from planetrecon.memory import cpu_memory_limit
    from planetrecon.pipeline.brightness import detector_full_scale, normalise_result

    full_scale = detector_full_scale(source, config, calibration) if config.normalise_brightness else None

    context = (cuda_allocation_limit(config.max_vram_bytes)
               if config.device != 'cpu' and config.geometry_mode == 'none' else nullcontext(None))
    with cpu_memory_limit(config.max_ram_bytes) as cpu_report, context as memory_report:
        def event(result, info):
            if result.stage == 'setup':
                on_event(result, info)
                return
            if full_scale is not None:
                normalise_result(result, config.normalise_percent, full_scale)
            if cpu_report is not None:
                result.provenance['cpu_memory_budget'] = cpu_report
            if on_event is not None:
                on_event(result, info)
        result = _stack_source(source, config, calibration=calibration, on_event=event if on_event else None,
                             should_cancel=should_cancel, resume_from=resume_from,
                             state_checkpoint=state_checkpoint, memory_report=memory_report,
                             preprocessing=preprocessing, preprocessing_cache=preprocessing_cache,
                             cache_validation=cache_validation)
        if cpu_report is not None:
            result.provenance['cpu_memory_budget'] = cpu_report
        if full_scale is not None:
            normalise_result(result, config.normalise_percent, full_scale)
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
    preprocessing=None,
    preprocessing_cache=None,
    cache_validation=None,
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
    fallback_warnings = []  # Local alignment that cannot apply at all falls back to global, visibly.
    if config.local_alignment and config.alignment_method == 'square':
        h, w = source.frame_shape()[:2]
        required = config.local_patch_size + 6  # Complete +/-3 pixel search footprint.
        if min(h, w) < required:
            largest = min(255, min(h, w) - 6)
            largest -= (largest + 1) % 2
            if largest >= 15:
                raise ValueError(f'Local patch size {config.local_patch_size} needs frames at least '
                                 f'{required} by {required} pixels; this capture is {w} by {h}. '
                                 f'Choose an odd alignment patch size from 15 to {largest} pixels.')
            fallback_warnings.append(f'Capture is {w} by {h} pixels, smaller than the smallest local patch '
                                     f'search (21 by 21); using global alignment.')
            config = replace(config, local_alignment=False)
    if (config.local_alignment and config.alignment_method == 'alignment_points'
            and config.geometry_mode == 'field'):
        raise ValueError('Alignment points support Motion model None, Surface, Combined or Saturn. Choose one of '
                         'those, or circular/square alignment for Field.')
    selection = reference_selection = None
    cache_status = {'status': 'disabled', 'reason': 'Cached preprocessing is disabled for this run.'}
    if config.frame_preselection:
        from planetrecon.pipeline.preprocess_cache import load_cache, identity, cache_report, selection_digest
        if preprocessing is not None:
            from planetrecon.pipeline.capture_hash import LEGACY_HASH_METHOD
            expected = preprocessing.validation_identity or preprocessing.identity
            if (preprocessing.cancelled or expected != identity(source, config, calibration, should_cancel,
                    hash_method=expected.get('pixels_hash_method', LEGACY_HASH_METHOD))
                    or preprocessing.digest != selection_digest(preprocessing)):
                raise ValueError('preprocessing measurements do not match the input or configuration')
            selection = preprocessing
            cache_status = cache_report(selection, config=config, source=source)
        else:
            selection, cache_status = load_cache(source, config, calibration, path=preprocessing_cache,
                                                should_cancel=should_cancel, validation=cache_validation)
            if cache_status['status'] in ('stale', 'invalid'):
                raise ValueError(cache_status['reason'] + ' Run Preprocess again or disable cached preprocessing.')
        if config.local_alignment and selection is None:
            raise ValueError('Local alignment requires a matching cache. Run Preprocess first.')
        if config.stack_percent < 100 and selection is None:
            raise ValueError('Best-frame selection requires a matching preprocessing cache. Run Preprocess first, or set best frames to 100%.')
        reference_selection = selection
        if selection is not None and config.stack_percent < 100:
            from planetrecon.pipeline.preprocess import best_frame_mask, quality_range
            accepted = best_frame_mask(selection, config.stack_percent, config.frame_selection_mode)
            detail = {'percent': config.stack_percent, 'mode': config.frame_selection_mode,
                      'screened_frames': int(selection.accepted.sum()),
                      'selected_frames': int(accepted.sum()),
                      'additional_exclusions': int(selection.accepted.sum() - accepted.sum()),
                      'ranking': 'cached quality descending; equal scores use earliest frame'}
            if config.frame_selection_mode == 'quality_range':
                low, high = quality_range(selection)
                detail.update(quality_minimum=low, quality_maximum=high,
                              quality_cutoff=None if low is None else low+(high-low)*(1-config.stack_percent/100.),
                              ranking='strictly above capture quality-range cutoff; screening still applies')
            selection = replace(selection, accepted=accepted,
                                summary={**selection.summary, 'best_frame_selection': detail})
        if selection is not None:
            if not selection.accepted.any():
                raise ValueError('no usable frames remain after preprocessing')
            reference_mask = reference_selection if config.geometry_mode != 'none' else selection
            if config.local_alignment and np.count_nonzero(reference_mask.accepted) < 4:
                fallback_warnings.append('Fewer than four frames are selected, too few for a local alignment '
                                         'reference; using global alignment. Select more frames to enable it.')
                config = replace(config, local_alignment=False)
            if config.reference_index >= len(selection.accepted):
                raise ValueError('reference_index is outside the capture')
            if config.reference_index and not reference_mask.accepted[config.reference_index]:
                raise ValueError('selected reference frame was rejected by preprocessing or best-frame selection; choose a retained frame or automatic reference 0')
    if config.geometry_mode != "none":
        from planetrecon.pipeline.geometry_stack import stack_source_geometry

        result = stack_source_geometry(
            source,
            config,
            calibration=calibration,
            on_event=on_event,
            should_cancel=should_cancel,
            resume_from=resume_from,
            state_checkpoint=state_checkpoint,
            selection=selection, cache_status=cache_status, reference_selection=reference_selection,
        )
        result.warnings = fallback_warnings + list(result.warnings)
        return result
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
    demosaic = PreparedBilinearDemosaic((h, w), color) if bayer else None
    # Only the eligible experimental local path changes its registration proxy.
    # It requires cached screening, so scalar quality weights stay unchanged.
    local_window = config.local_patch_size
    local_step = local_window // 2
    centred_local_axis = False
    if config.local_alignment and config.alignment_method == 'square':
        from planetrecon.pipeline.local_align import patch_centres
        axes = [patch_centres(length, local_window, local_step) for length in (h, w)]
        centred_local_axis = any(len(axis) == 1 and axis[0] != local_window//2+3 for axis in axes)
    colour_registration = config.local_alignment and bayer
    alignment_plane = partial(_colour_registration_plane, demosaic=demosaic) if colour_registration else _alignment_plane
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
    chosen_reference = (config.reference_index if config.reference_index else
                        selection.best_reference_index if selection is not None else None)
    if chosen_reference is not None:
        raw_ref = source.read_raw(chosen_reference)
        calibrated_ref, ref_info = apply_calibration(raw_ref, calibration)
        if (not np.all(np.isfinite(calibrated_ref)) or
                (config.reject_saturated and (_saturated(raw_ref, meta.bit_depth) or ref_info["saturated"]))):
            raise ValueError("selected reference frame is invalid or saturated")
        reference = alignment_plane(calibrated_ref, color)
        reference_index = chosen_reference
    n_used = 0
    n_rejected = 0
    # Quality of each accumulated frame (NaN otherwise) for the stack summary;
    # None after resuming a checkpoint that predates the record.
    frame_quality = np.full(n, np.nan)
    pending_quality = {}
    warnings = fallback_warnings + list(report.warnings)
    if "warnings" in meta.extras:
        warnings.extend(meta.extras["warnings"].value)
    if report.fallback:
        warnings.append(report.reason)
    bit_depth = meta.bit_depth
    seq = 0
    cancelled = False

    snapshot_provenance = capture_provenance(source, config, calibration)
    snapshot_provenance["preprocessing_cache"] = cache_status
    snapshot_provenance["scalar_frame_weight"] = "linear quality" if config.quality_weighting else "equal"
    snapshot_provenance['frame_brightness'] = 'recorded detector values; explicit calibration only; no per-frame brightness normalisation'
    if selection is not None:
        snapshot_provenance['preprocessing'] = selection.summary
    snapshot_provenance["registration"] = "Gaussian 1.5px amplitude correlation with subpixel peak fit"
    if memory_report is not None:
        snapshot_provenance['cuda_allocation_budget'] = memory_report
    next_index = 0
    state_identity = None
    if resume_from is not None or state_checkpoint is not None:
        state_identity = resume.identity(source, config, calibration, should_cancel)
        from planetrecon.pipeline.preprocess_cache import reconstruction_digest
        state_identity["preprocessing_digest"] = reconstruction_digest(selection) if selection is not None else None
        if config.local_alignment:
            if config.alignment_method == 'circular_multiscale':
                state_identity['circular_registration_version'] = 1
            if config.alignment_method == 'alignment_points':
                state_identity['ap_registration_version'] = 1
            state_identity['local_registration_version'] = 2 if colour_registration else 1
            state_identity['local_patch_support'] = 'complete observed search footprint'
            state_identity['local_patch_boundary'] = 'one grid interval taper to global'
            state_identity['local_patch_peak'] = 'joint two-dimensional quadratic'
            state_identity['local_template_boundary'] = 'normalised support; best-frame fill'
            if centred_local_axis:
                state_identity['local_patch_grid'] = 'singleton axes centred'
    if resume_from is not None:
        restored = resume.load(resume_from, state_identity, accum.shape, n, bayer)
        accum, weight = restored["accum"], restored["weight"]
        reference, reference_index = restored["reference"], restored["reference_index"]
        n_used, n_rejected = restored["n_used"], restored["n_rejected"]
        demosaic_accum, demosaic_weight = restored["demosaic_accum"], restored["demosaic_weight"]
        next_index = restored["next_index"]
        frame_quality = restored[resume.FRAME_QUALITY]
        history = restored["execution_history"]
        report.execution_history = history + ([] if history[-1] == backend.name else [backend.name])
        warnings = list(dict.fromkeys(restored["warnings"] + warnings))
        snapshot_provenance["resumed_from_frame"] = next_index

    device_sums = None  # Alignment-point CUDA batch sums; CPU arrays stay authoritative.

    def cpu_fallback(exc):
        nonlocal backend, reference
        if backend.name != "cuda":
            raise exc
        from planetrecon.backends.cpu import CPUBackend
        message = f"CUDA operation failed; continued on CPU with prior sums retained: {type(exc).__name__}: {exc}"
        backend = CPUBackend(threads=config.threads)
        report.selected, report.fallback, report.reason = "cpu", True, message
        report.execution_history.append("cpu")
        warnings.append(message)
        if reference is not None:
            reference = backend.prepare_reference(reference)
        if device_sums is not None and device_sums.pending:
            try:
                device_sums.flush(accum, weight, *((demosaic_accum, demosaic_weight) if bayer else ()))
            except Exception:
                # Unmerged device sums are lost: recompute exactly those frames on CPU.
                for index in device_sums.discard():
                    accumulate(process_frame((index, source.read_raw(index))))

    def setup_progress(phase, done, total):
        if on_event is not None:
            on_event(SetupProgress, {'phase': phase, 'n_processed': done, 'n_total': total,
                                     'backend': backend.name})

    def prepare_reference():
        nonlocal reference
        if reference is not None:
            try:
                reference = backend.prepare_reference(reference)
            except RuntimeError as exc:
                cpu_fallback(exc)

    prepare_reference()

    local_matcher = None
    local_selection = None
    if config.local_alignment:
        from planetrecon.pipeline.local_align import LocalRegistration, build_template, cpu_backproject
        points = config.alignment_method == 'alignment_points'
        accepted = np.flatnonzero(selection.accepted)
        order = np.argsort(-selection.measurements[accepted, 0], kind='stable')
        if points:
            from planetrecon.pipeline.ap_align import REFERENCE_FRAMES, refined_reference
            candidates = accepted[order[:REFERENCE_FRAMES]]
        else:
            candidates = accepted[order[:64]]
        snapshot_provenance['local_alignment'] = {
            'version': 2 if colour_registration else 1, 'enabled': True, 'template_candidates': candidates.tolist(),
            'window_px': local_window, 'step_px': local_step, 'maximum_residual_px': 3,
            'anchor_index': reference_index,
            'patch_support': 'complete observed search footprint',
            'patch_boundary': 'one grid interval taper to global',
            'patch_peak': 'joint two-dimensional quadratic',
            'template': 'mean of valid aligned candidates, origin anchored to the selected best frame',
            'template_boundary': 'normalised support; best-frame fill',
        }
        if colour_registration:
            snapshot_provenance['local_alignment']['registration_proxy'] = 'bilinear RGB luminance (0.25 R + 0.5 G + 0.25 B)'
        if centred_local_axis:
            snapshot_provenance['local_alignment']['patch_grid'] = 'singleton axes centred'

        def read_plane(index):
            calibrated, _ = apply_calibration(source.read_raw(index), calibration)
            if points and backend.name == 'cuda':
                return backend.registration_plane(calibrated, color)
            return alignment_plane(calibrated, color)
        if resume_from is None:
            build = (partial(refined_reference, size=config.ap_size, use_cuda=backend.name == 'cuda',
                             progress=setup_progress) if points else build_template)
            try:
                reference = build(reference, candidates, read_plane,
                                  backend.phase_correlation, config.max_shift_px, should_cancel=should_cancel)
            except RuntimeError as exc:
                cpu_fallback(exc)
                build = (partial(refined_reference, size=config.ap_size, progress=setup_progress)
                         if points else build_template)
                reference = build(reference, candidates, read_plane,
                                  backend.phase_correlation, config.max_shift_px, should_cancel=should_cancel)
            prepare_reference()
        detail = snapshot_provenance['local_alignment']
        if config.alignment_method == 'circular_multiscale':
            from planetrecon.pipeline.circular_align import CircularMultiscaleRegistration
            local_matcher = CircularMultiscaleRegistration(reference,
                sampling_multiplier=config.sampling_multiplier,
                wavelength_nm=config.alignment_wavelength_nm, should_cancel=should_cancel)
            for key in ('window_px', 'step_px', 'patch_boundary', 'maximum_residual_px', 'patch_grid'):
                detail.pop(key, None)
            detail.update(local_matcher.provenance)
            if not local_matcher.layers:
                warnings.append('Capture is smaller than the sampling-derived alignment diameter plus search margin; using global alignment.')
            snapshot_provenance['registration'] += ' + circular multiscale local alignment'
        elif points:
            from planetrecon.pipeline.ap_align import AlignmentPoints, select_locally
            local_matcher = AlignmentPoints(reference, size=config.ap_size, should_cancel=should_cancel)
            for key in ('window_px', 'step_px', 'patch_boundary', 'maximum_residual_px', 'patch_grid',
                        'patch_support', 'patch_peak', 'template_boundary', 'template_candidates'):
                detail.pop(key, None)
            detail.update(local_matcher.provenance, template_frames=len(candidates),
                          template='best selected frames averaged after global alignment, then again after '
                                   'alignment-point registration to that mean; origin anchored to the best frame')
            if not local_matcher.layers:
                warnings.append('No bright, textured region fits an alignment point; using global alignment.')
            screened = np.flatnonzero(reference_selection.accepted)
            count = int(np.count_nonzero(selection.accepted))
            if local_matcher.layers and len(screened) <= count:
                # Every screened frame is selected (e.g. 100%): each point would
                # choose every frame, so ranking cannot change the result.
                detail['local_quality'] = {'frames_per_point': count, 'ranked_frames': 0,
                                           'skipped': 'every screened frame is selected'}
            elif local_matcher.layers:
                try:
                    local_selection = select_locally(local_matcher, screened, read_plane, backend.phase_correlation,
                                                     reference, count, selection.accepted.copy(), should_cancel,
                                                     use_cuda=backend.name == 'cuda', progress=setup_progress)
                except RuntimeError as exc:
                    cpu_fallback(exc)
                    local_selection = select_locally(local_matcher, screened, read_plane, backend.phase_correlation,
                                                     reference, count, selection.accepted.copy(), should_cancel,
                                                     progress=setup_progress)
                selection = replace(selection, accepted=local_selection.frames())
                detail['local_quality'] = {
                    'frames_per_point': count, 'ranked_frames': len(screened),
                    'candidates': 'every screened frame',
                    'contributing_frames': int(np.count_nonzero(selection.accepted)),
                    'measure': 'squared Laplacian of the 1.5 px smoothed plane, Gaussian-weighted around each '
                               'point, divided by local brightness squared',
                    'blending': 'normalised Gaussian footprints (sigma 0.6 x spacing); global selection outside points',
                }
            snapshot_provenance['registration'] += ' + alignment points'
            if backend.name == 'cuda':
                from planetrecon.backends.torch_ap import DeviceAccumulator
                device_sums = DeviceAccumulator()
        else:
            local_matcher = LocalRegistration(reference, window=local_window, step=local_step)
            snapshot_provenance['registration'] += ' + confidence-gated normalized local patches'

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

    def device_frame(index, calibrated):
        """Alignment points on CUDA: plane, registration and contributions stay on the device."""
        plane = backend.registration_plane(calibrated, color)
        # Local quality already registered this frame against the same reference.
        shift = local_selection.shifts.get(int(index)) if local_selection is not None else None
        if shift is None:
            shift = backend.phase_correlation(reference, plane)
        if not np.all(np.isfinite(shift)) or abs(shift[0]) > config.max_shift_px or abs(shift[1]) > config.max_shift_px:
            return None
        score = selection.measurements[index, 0]
        if not np.isfinite(score):
            return None
        pending_quality[int(index)] = max(score, 1e-12)
        score = max(score, 1e-12) if config.quality_weighting else 1.
        if local_matcher.use_cuda is False:
            local_matcher.use_cuda = True
        field = local_matcher.displacement_tensor(plane, shift)
        projected = backend.backproject_tensors(calibrated, field, color)
        if local_selection is not None:
            score = score * local_selection.weights_tensor(index)
        return (score, *projected)

    def process_frame(item):
        nonlocal reference, reference_index
        index, raw = item
        if selection is not None and not selection.accepted[index]:
            return None
        if not np.all(np.isfinite(raw)) or (config.reject_saturated and _saturated(raw, bit_depth)):
            return None
        calibrated, cal_info = apply_calibration(raw, calibration)
        if not np.all(np.isfinite(calibrated)) or (config.reject_saturated and cal_info["saturated"]):
            return None
        if device_sums is not None and backend.name == 'cuda':
            try:
                return device_frame(index, calibrated)
            except RuntimeError as exc:
                if hasattr(local_matcher, 'release_cuda'):
                    local_matcher.release_cuda()
                local_matcher.use_cuda = False
                cpu_fallback(exc)
        demosaiced = (demosaic(calibrated)
                      if colour_registration and backend.name != 'cuda' else None)
        plane = (_alignment_plane(demosaiced,'RGB') if demosaiced is not None
                 else alignment_plane(calibrated, color))
        if reference is None:
            reference = plane
            reference_index = int(index)
            prepare_reference()
            shift = (0.0, 0.0)
        else:
            try:
                shift = backend.phase_correlation(reference, plane)
            except RuntimeError as exc:
                cpu_fallback(exc)
                shift = backend.phase_correlation(reference, plane)
        if not np.all(np.isfinite(shift)) or abs(shift[0]) > config.max_shift_px or abs(shift[1]) > config.max_shift_px:
            return None
        score = max(selection.measurements[index, 0] if selection is not None else laplacian_score(plane), 1e-12)
        if not np.isfinite(score):
            return None
        pending_quality[int(index)] = score
        if not config.quality_weighting:
            score = 1.
        if local_matcher is not None:
            if local_matcher.use_cuda != (backend.name == 'cuda'):
                local_matcher.use_cuda = backend.name == 'cuda'
            try:
                global_shift = shift
                shift = (local_matcher.displacement_tensor(plane, shift)
                         if backend.name == 'cuda' and config.alignment_method != 'square'
                         else local_matcher.displacement(plane, shift))
            except RuntimeError as exc:
                if hasattr(local_matcher, 'release_cuda'):
                    local_matcher.release_cuda()
                cpu_fallback(exc)
                local_matcher.use_cuda = False
                shift = local_matcher.displacement(plane, shift)
        projected = None
        if backend.name == 'cuda':
            try:
                projected = backend.backproject(calibrated, shift, color)
            except RuntimeError as exc:
                cpu_fallback(exc)
                if hasattr(shift, 'detach'):
                    local_matcher.release_cuda()
                    local_matcher.use_cuda = False
                    shift = local_matcher.displacement(plane, global_shift)
        if projected is None and local_matcher is not None:
            projected = cpu_backproject(calibrated, shift, color, demosaiced=demosaiced)
        if projected is not None:
            if local_selection is not None:
                # Per-pixel blend of the alignment points that selected this frame.
                score = score * local_selection.weights(index)
            return (score, *projected)
        support = ndshift(np.ones((h, w)), shift=(-shift[1], -shift[0]),
                          order=1, prefilter=False, mode='grid-constant')
        if bayer:
            add, wt = cfa_accumulate(calibrated, shift, color)
            demo = demosaic(calibrated)
            shifted = np.stack([ndshift(demo[..., c], shift=(-shift[1], -shift[0]),
                                order=1, prefilter=False, mode='grid-constant') for c in range(3)], axis=2)
            return score, add, wt, shifted, support
        if rgb:
            if calibrated.ndim != 3:
                raise ValueError('RGB source produced a 2-D frame')
            add = np.stack([ndshift(calibrated[..., c], shift=(-shift[1], -shift[0]),
                           order=1, prefilter=False, mode='grid-constant') for c in range(3)], axis=2)
            return score, add, support[..., None], None, None
        add = ndshift(calibrated, shift=(-shift[1], -shift[0]),
                      order=1, prefilter=False, mode='grid-constant')
        return score, add, support, None, None

    def accumulate(projected):
        if projected is None:
            return
        score, add, wt, demo, support = projected
        scale = score[..., None] if np.ndim(score) == 2 and add.ndim == 3 else score
        accum[...] += scale * add
        weight[...] += scale * wt
        if bayer:
            demosaic_accum[...] += scale * demo
            demosaic_weight[...] += (score * support)[..., None]

    from planetrecon.pipeline.cpu_pool import CPUFramePool, selected_batches
    with CPUFramePool(config, shape, n-next_index, enabled=backend.name == 'cpu',
                      should_cancel=should_cancel) as pool:
        if backend.name == 'cpu':
            report.frame_workers = pool.workers
        snapshot_provenance['cpu_frame_workers'] = {'workers': pool.workers, 'limit': pool.reason,
            'scratch_budget_bytes': pool.scratch_budget_bytes,
            'estimated_frame_bytes': pool.estimated_frame_bytes,
            'accumulation': 'capture order; float64; no frame brightness normalisation'}
        if local_matcher is not None and config.alignment_method == 'circular_multiscale' and pool.workers > 1:
            # Only the owner polls the caller; workers observe a thread-safe flag.
            local_matcher.should_cancel = pool.stop.is_set
        if config.frame_preselection:
            emit('cache_ready', incomplete=True)
        def frame_results(items):
            items = iter(items)
            # Without screening, establish the first valid observation as the
            # fixed reference on the owner before any worker can read it.
            while reference is None:
                if pool.cancelled():
                    return
                try:
                    item = next(items)
                except StopIteration:
                    return
                yield process_frame(item)
            yield from pool.map(process_frame, items)

        batches = (selected_batches(source, config.batch_frames, start=next_index,
                                    accepted=selection.accepted, should_cancel=should_cancel)
                   if pool.workers > 1 and selection is not None else
                   source.iter_batches(config.batch_frames, start=next_index, should_cancel=should_cancel))
        for indices, batch in batches:
            if pool.cancelled():
                cancelled = True
                break
            try:
                for index, projected in zip(indices, frame_results(zip(indices, batch))):
                    if pool.cancelled():
                        cancelled = True
                        break
                    # Screening exclusions may have been skipped while filling
                    # the work batch. Account for exactly the consumed prefix.
                    n_rejected += int(index) - (n_used + n_rejected)
                    quality = pending_quality.pop(int(index), np.nan)
                    if projected is None:
                        n_rejected += 1
                        continue
                    if device_sums is not None and not isinstance(projected[1], np.ndarray):
                        device_sums.add(index, *projected)
                    else:
                        accumulate(projected)
                    if frame_quality is not None:
                        frame_quality[int(index)] = quality
                    n_used += 1
            except InterruptedError:
                if not pool.cancelled():
                    raise
                cancelled = True
            if device_sums is not None and device_sums.pending:
                try:
                    device_sums.flush(accum, weight, *((demosaic_accum, demosaic_weight) if bayer else ()))
                except RuntimeError as exc:
                    cpu_fallback(exc)
            cancelled = cancelled or pool.cancelled()
            if state_checkpoint is not None:
                resume.save(state_checkpoint, state_identity, {
                    "accum": accum, "weight": weight, "reference": reference,
                    "demosaic_accum": demosaic_accum, "demosaic_weight": demosaic_weight,
                    "reference_index": reference_index, "n_used": n_used, "n_rejected": n_rejected,
                    "next_index": n_used + n_rejected, resume.FRAME_QUALITY: frame_quality,
                    "execution_history": report.execution_history, "warnings": warnings})
            emit("baseline", incomplete=True)
            if cancelled:
                break
        if not cancelled and not pool.cancelled():
            n_rejected += n - (n_used + n_rejected)

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
    provenance["stack_quality"] = _stack_quality(source, calibration, config, frame_quality,
                                                 reference_selection, local_selection)
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


def _stack_quality(source, calibration, config, frame_quality, screened_selection, local_selection):
    """Frames, weighting, effective frame count and estimated SNR of a finished stack."""
    from planetrecon.pipeline.stack_quality import single_frame_snr, summarise
    if frame_quality is None:
        return {'unavailable': 'resumed from a checkpoint that predates per-frame quality records'}
    screened = None
    if screened_selection is not None:
        screened = np.where(screened_selection.accepted, screened_selection.measurements[:, 0], np.nan)
    used = np.flatnonzero(np.isfinite(frame_quality))
    return summarise(frame_quality, config.quality_weighting, screened=screened, local_selection=local_selection,
                     frame_snr=single_frame_snr(source, calibration, used))


def _normalise_stack(accum: np.ndarray, weight: np.ndarray) -> np.ndarray:
    return np.divide(accum, weight, out=np.zeros_like(accum), where=weight > 0)
