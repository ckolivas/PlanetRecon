"""Translation baseline: quality-weighted stacking and raw CFA RGB backprojection."""

from __future__ import annotations

import sys
import time
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
from planetrecon.result import DeferredResult, ReconstructionResult
from planetrecon.pipeline.provenance import capture_provenance
from planetrecon.pipeline.colour import complete_bayer_rgb


# A checkpoint holds every accumulator (with Fourier weighting, two full-frame sums
# per rank bin), so it is rewritten at most this often during a stack.
CHECKPOINT_INTERVAL_S = 60.

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
    from planetrecon.io.readahead import Readahead
    with cpu_memory_limit(config.max_ram_bytes) as cpu_report, context as memory_report, \
            Readahead(source) as reader:
        def event(result, info):
            if result.stage == 'setup':
                on_event(result, info)
                return

            def finish(result):
                if full_scale is not None:
                    normalise_result(result, config.normalise_percent, full_scale)
                if cpu_report is not None:
                    result.provenance['cpu_memory_budget'] = cpu_report
            if isinstance(result, DeferredResult):
                result.then(finish)
            else:
                finish(result)
            if on_event is not None:
                on_event(result, info)
        result = _stack_source(source, config, calibration=calibration, on_event=event if on_event else None,
                             should_cancel=should_cancel, resume_from=resume_from,
                             state_checkpoint=state_checkpoint, memory_report=memory_report, reader=reader,
                             preprocessing=preprocessing, preprocessing_cache=preprocessing_cache,
                             cache_validation=cache_validation)
        if cpu_report is not None:
            result.provenance['cpu_memory_budget'] = cpu_report
        if full_scale is not None:
            normalise_result(result, config.normalise_percent, full_scale)
        if config.geometry_mode != 'none':
            from planetrecon.geometry.pose import output_epoch
            # The time a motion-model stack represents; output names follow it.
            epoch = output_epoch(source, config.reference_epoch_s, cadence_s=config.cadence_s)
            if epoch is not None:
                result.provenance['output_epoch'] = epoch
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
    reader=None,
) -> ReconstructionResult:
    """``reader`` (``io.readahead.Readahead``) reads ahead of each pass; source reads go through it."""
    if reader is None:
        from planetrecon.io.readahead import Readahead
        reader = Readahead(source)
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
    selection = reference_selection = None
    cache_status = {'status': 'disabled', 'reason': 'Cached preprocessing is disabled for this run.'}
    if config.frame_preselection:
        from planetrecon.pipeline.preprocess_cache import (load_cache, identity, cache_report, selection_digest,
                                                           effective_selection)
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
        selection, cache_status = effective_selection(selection, cache_status, config)
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
        result.provenance['moons'] = ({'enabled': True, 'skipped': 'not available with planet rotation models'}
                                      if config.moon_stacking else {'enabled': False})
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
    # Alignment points register Bayer captures on bilinear RGB luminance. They
    # require cached screening, so scalar quality weights stay unchanged.
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
    # Fourier weighting keeps one accumulator pair per local-quality rank bin.
    from planetrecon.pipeline import fourier_weights
    rank_bins = (fourier_weights.rank_bins(int(np.count_nonzero(selection.accepted)))
                 if config.fourier_active else 0)
    n_bins = 2 * rank_bins
    bin_accum = np.zeros((n_bins,) + accum.shape) if n_bins else None
    bin_weight = np.zeros((n_bins,) + weight.shape) if n_bins else None

    reference = None
    reference_index = None
    chosen_reference = (config.reference_index if config.reference_index else
                        selection.best_reference_index if selection is not None else None)
    if chosen_reference is not None:
        raw_ref = reader.read_raw(chosen_reference)
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
    restored = None
    if resume_from is not None or state_checkpoint is not None:
        state_identity = resume.identity(source, config, calibration, should_cancel)
        from planetrecon.pipeline.preprocess_cache import reconstruction_digest
        state_identity["preprocessing_digest"] = reconstruction_digest(selection) if selection is not None else None
        if config.local_alignment:
            state_identity['ap_registration_version'] = 1
            state_identity['local_registration_version'] = 2 if colour_registration else 1
            state_identity['local_template_boundary'] = 'normalised support; best-frame fill'
        if n_bins:
            state_identity['fourier_bins'] = n_bins
        if config.moon_stacking:
            from planetrecon.pipeline.moons import VERSION as MOON_VERSION
            state_identity['moon_stacking_version'] = MOON_VERSION
    if resume_from is not None:
        restored = resume.load(resume_from, state_identity, accum.shape, n, bayer, bins=n_bins)
        bin_accum, bin_weight = restored['bin_accum'], restored['bin_weight']
        accum, weight = restored["accum"], restored["weight"]
        reference, reference_index = restored["reference"], restored["reference_index"]
        n_used, n_rejected = restored["n_used"], restored["n_rejected"]
        next_index = restored["next_index"]
        frame_quality = restored[resume.FRAME_QUALITY]
        history = restored["execution_history"]
        report.execution_history = history + ([] if history[-1] == backend.name else [backend.name])
        warnings = list(dict.fromkeys(restored["warnings"] + warnings))
        snapshot_provenance["resumed_from_frame"] = next_index

    device_sums = None  # Alignment-point CUDA batch sums; CPU arrays stay authoritative.

    def flush_device(everything=True):
        device_sums.flush(accum, weight, bins=(bin_accum, bin_weight) if n_bins else None, everything=everything)

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
        if device_sums is not None and (device_sums.pending or device_sums.bin_pending):
            try:
                flush_device()
            except Exception:
                # Unmerged device sums are lost: recompute exactly those frames' sums on CPU.
                pending, bin_only = device_sums.discard()
                for index in pending:
                    accumulate(process_frame((index, reader.read_raw(index))))
                for index in bin_only:
                    accumulate(process_frame((index, reader.read_raw(index))), main=False)

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
        from planetrecon.pipeline.ap_align import (REFERENCE_FRAMES, AlignmentPoints, calibration_samples,
                                                   refined_reference, select_locally)
        from planetrecon.pipeline.local_align import cpu_backproject
        accepted = np.flatnonzero(selection.accepted)
        order = np.argsort(-selection.measurements[accepted, 0], kind='stable')
        candidates = accepted[order[:REFERENCE_FRAMES]]
        snapshot_provenance['local_alignment'] = {
            'version': 2 if colour_registration else 1, 'enabled': True, 'anchor_index': reference_index,
        }
        if colour_registration:
            snapshot_provenance['local_alignment']['registration_proxy'] = 'bilinear RGB luminance (0.25 R + 0.5 G + 0.25 B)'

        def read_plane(index):
            raw = reader.read_raw(index)
            if backend.name == 'cuda' and calibration is None:
                return backend.registration_plane(raw, color)  # Uploads the raw samples; float64 on the device.
            calibrated, _ = apply_calibration(raw, calibration)
            if backend.name == 'cuda':
                return backend.registration_plane(calibrated, color)
            return alignment_plane(calibrated, color)
        if resume_from is None:
            try:
                reference = refined_reference(reference, candidates, read_plane, backend.phase_correlation,
                                              config.max_shift_px, size=config.ap_size,
                                              use_cuda=backend.name == 'cuda', should_cancel=should_cancel,
                                              progress=setup_progress, plan=reader.expect)
            except RuntimeError as exc:
                cpu_fallback(exc)
                reference = refined_reference(reference, candidates, read_plane, backend.phase_correlation,
                                              config.max_shift_px, size=config.ap_size,
                                              should_cancel=should_cancel, progress=setup_progress,
                                              plan=reader.expect)
            prepare_reference()
        detail = snapshot_provenance['local_alignment']
        local_matcher = AlignmentPoints(reference, size=config.ap_size, should_cancel=should_cancel)
        if local_matcher.layers:
            def calibrate():
                local_matcher.calibrate(calibration_samples(np.flatnonzero(selection.accepted), read_plane,
                                                            backend.phase_correlation, reference,
                                                            config.max_shift_px, plan=reader.expect))
            local_matcher.use_cuda = backend.name == 'cuda'
            try:
                calibrate()
            except RuntimeError as exc:
                local_matcher.release_cuda()
                local_matcher.use_cuda = False
                cpu_fallback(exc)
                calibrate()
        detail.update(local_matcher.provenance, template_frames=len(candidates),
                      template='best selected frames averaged after global alignment, then again after '
                               'alignment-point registration to that mean; origin anchored to the best frame')
        if not local_matcher.layers:
            warnings.append('No bright, textured region fits an alignment point; using global alignment.')
        screened = np.flatnonzero(reference_selection.accepted)
        count = int(np.count_nonzero(selection.accepted))
        if not local_matcher.layers and n_bins:
            warnings.append('Fourier weighting needs alignment points; using ordinary weights.')
            n_bins, bin_accum, bin_weight = 0, None, None
        if local_matcher.layers and len(screened) <= count and not n_bins:
            # Every screened frame is selected (e.g. 100%): each point would
            # choose every frame, so ranking cannot change the result.
            detail['local_quality'] = {'frames_per_point': count, 'ranked_frames': 0,
                                       'skipped': 'every screened frame is selected'}
        elif local_matcher.layers:
            try:
                from planetrecon.backends.torch_ap import PinnedStage
                rank_stage, rank_groups = PinnedStage(), [0]

                def rank_group(indices):
                    """Global shifts and point scores of frames registered together on CUDA."""
                    from planetrecon.backends import precision
                    from planetrecon.backends.torch_ap import local_quality_batch
                    rank_groups[0] += 1
                    parity = rank_groups[0] % 2
                    raws = [reader.read_raw(index) for index in indices]
                    # Non-blocking uploads; uncalibrated integer frames go up as they are read.
                    images = rank_stage.upload_stack(('frames', parity), raws if calibration is None else
                                                     [apply_calibration(raw, calibration)[0] for raw in raws])
                    planes = backend.registration_planes(images, color)
                    found = backend.phase_correlations(reference, planes)
                    centres = rank_stage.upload(('centres', parity), local_matcher.centres.astype(float)[None]
                                                + np.array([[float(dy), float(dx)] for dx, dy in found])[:, None, :])
                    return found, local_quality_batch(planes.to(precision.REGISTRATION), centres, local_matcher.size / 4)
                cuda_ranking = backend.name == 'cuda'
                local_selection = select_locally(local_matcher, screened, read_plane, backend.phase_correlation,
                                                 reference, count, selection.accepted.copy(), should_cancel,
                                                 use_cuda=cuda_ranking, progress=setup_progress,
                                                 n_bins=rank_bins if n_bins else 0,
                                                 global_scores=selection.measurements[:, 0], plan=reader.expect,
                                                 measure_group=rank_group if cuda_ranking else None,
                                                 group=_group_frames(h * w, 6 if rgb else 2, memory_report))
            except RuntimeError as exc:
                cpu_fallback(exc)
                local_selection = select_locally(local_matcher, screened, read_plane, backend.phase_correlation,
                                                 reference, count, selection.accepted.copy(), should_cancel,
                                                 progress=setup_progress,
                                                 n_bins=rank_bins if n_bins else 0,
                                                 global_scores=selection.measurements[:, 0], plan=reader.expect)
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
            from planetrecon.backends.torch_ap import DeviceSums
            if local_selection is not None:
                blend = local_selection.blend_tensors()
            else:
                # Every screened frame is selected: each frame weighs its score everywhere.
                import torch
                zeros = lambda *size: torch.zeros(size, dtype=torch.float64, device='cuda:0')
                blend = zeros(h, 1), zeros(1, w), zeros(h * w), torch.ones(h * w, dtype=torch.float64, device='cuda:0')
            device_sums = DeviceSums((h, w), 3 if rgb else 1, n_bins, blend)
            group_frames = _group_frames(h * w, 6 if rgb else 2, memory_report)
            detail['cuda_execution'].update(
                frame_group=group_frames,
                accumulation=('fused Triton backprojection and slot sums (footprints computed per pixel)'
                              if device_sums.kernels is not None else 'torch backprojection and per-term sums'))

    moon_stack = None
    moon_state = None  # Checkpointed moon definitions; None when moons were not looked for.
    pending_moons = {}
    if not config.moon_stacking:
        snapshot_provenance['moons'] = {'enabled': False}
    elif selection is None or reference is None:
        snapshot_provenance['moons'] = {'enabled': True, 'skipped': 'needs preprocessing to choose a reference'}
    else:
        from planetrecon.pipeline import moons as moon_module
        times, real_time = moon_module.frame_times(source)
        t_out = .5 * (times[0] + times[-1])
        channels = 3 if rgb else 1
        if restored is not None and restored.get('moons') is not None:
            moon_stack = moon_module.MoonStack(
                [moon_module.Moon.from_dict(m) for m in restored['moons']['moons']], (h, w), channels, times,
                t_out, bayer=color if bayer else None, accum=restored['moon_accum'], weight=restored['moon_weight'],
                counts=restored['moons'].get('counts'))
            moon_summary = restored['moons'].get('summary', {})
        else:
            def read_moon_plane(index):
                raw = reader.read_raw(index)
                calibrated, _ = apply_calibration(raw, calibration)
                if backend.name == 'cuda' and (colour_registration or not bayer):
                    # The device registration plane is the stack's alignment plane here.
                    registration = backend.registration_plane(raw if calibration is None else calibrated, color)
                    return registration.double().cpu().numpy(), registration
                plane = np.asarray(alignment_plane(calibrated, color), dtype=np.float64)
                return plane, plane

            def find_moons():
                points = local_matcher.centres if local_matcher is not None and local_matcher.layers else ()
                return moon_module.setup(
                    np.flatnonzero(local_selection.global_mask if local_selection is not None
                                   else selection.accepted), read_moon_plane, backend.phase_correlation, reference,
                    config.max_shift_px, times, t_out, real_time=real_time, points=points,
                    point_reach=.75 * config.ap_size, progress=setup_progress,
                    check_cancel=check_cancel, plan=reader.expect)

            def check_cancel():
                if should_cancel is not None and should_cancel():
                    raise InterruptedError('stack cancelled while finding moons')
            try:
                found, moon_summary = find_moons()
            except RuntimeError as exc:
                cpu_fallback(exc)
                found, moon_summary = find_moons()
            moon_stack = moon_module.MoonStack(found, (h, w), channels, times, t_out,
                                               bayer=color if bayer else None)
        moon_state = {**moon_stack.state(), 'summary': moon_summary}
        if not moon_stack.moons:
            moon_stack = None
        snapshot_provenance['moons'] = {
            'enabled': True, 'version': moon_module.VERSION, 'method': moon_module.METHOD,
            'time_unit': 's' if real_time else 'frame', **moon_summary,
            'stacked': len(moon_stack.moons) if moon_stack is not None else 0}

    def moon_contributions(index, calibrated, plane, global_shift, score):
        if moon_stack is None:
            return
        if local_selection is not None:
            # Outside the alignment points the global frame selection applies.
            score = score * float(local_selection.global_mask[index])
        if score > 0:
            pending_moons[int(index)] = (moon_stack.contribute(int(index), calibrated, plane, global_shift), score)

    def emit(stage: str, incomplete: bool) -> None:
        """Report progress. The result's image, coverage and provenance are formed only if the
        consumer reads them (or keeps the result), so progress alone costs no image."""
        nonlocal seq
        if on_event is None:
            return

        def form(result):
            if device_sums is not None and device_sums.pending:
                try:
                    device_sums.flush_main(accum, weight)
                except RuntimeError as exc:
                    cpu_fallback(exc)
            cov = weight.copy()
            result.image = _normalise_stack(accum, weight) if moon_stack is None else moon_stack.composite(
                _normalise_stack(accum, weight))
            result.coverage, result.validity = cov, cov > 0
        result = DeferredResult(
            form,
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
        result.then(complete_bayer_rgb)
        seq += 1
        held = sys.getrefcount(result)
        on_event(result, {"seq": seq, "n_used": n_used, "n_processed": n_used + n_rejected, "n_total": n,
                          "backend": backend.name, "device_report": dict(report.__dict__)})
        if not result.formed and sys.getrefcount(result) > held:
            result.form()  # The consumer kept it: form it while the sums are those it reports.

    def device_group(chunk):
        """Alignment points on CUDA for a group of (index, raw): planes, matching, backprojection and
        weights stay on the device. Returns (indices used, (indices, contributions, pairs) or None).

        The frames go up in one non-blocking upload; their saturation counts, correlation surfaces
        and moon centroid windows come back in one copy, so the CPU prepares the next group while the
        GPU resamples and accumulates this one.
        """
        import torch
        kept = []
        for index, raw in chunk:
            index = int(index)
            if not selection.accepted[index] or not np.isfinite(selection.measurements[index, 0]):
                continue
            if raw.dtype.kind not in 'ui' and not np.all(np.isfinite(raw)):
                continue
            if calibration is None:
                calibrated = raw  # The raw samples are the measurements; saturation is counted on the device.
            else:
                if config.reject_saturated and _saturated(raw, bit_depth):
                    continue
                calibrated, cal_info = apply_calibration(raw, calibration)
                if not np.all(np.isfinite(calibrated)) or (config.reject_saturated and cal_info["saturated"]):
                    continue
            # Local quality already registered this frame against the same reference.
            shift = local_selection.shifts.get(index) if local_selection is not None else None
            kept.append([index, calibrated, shift])
        if not kept:
            return set(), None
        local_matcher._check_cancel()
        device_sums.groups += 1
        # Raw integer samples stay integers on the device; the kernels read them as float64.
        images = device_sums.stage.upload_stack(('frames', device_sums.groups % 2),
                                                [calibrated for _, calibrated, _ in kept])
        if device_sums.kernels is None:
            images = images.to(torch.float64)
        planes = backend.registration_planes(images, color)
        for k, item in enumerate(kept):
            if item[2] is None:
                item[2] = backend.phase_correlation(reference, planes[k])
        good = [k for k, (_, _, shift) in enumerate(kept) if np.all(np.isfinite(shift))
                and abs(shift[0]) <= config.max_shift_px and abs(shift[1]) <= config.max_shift_px]
        if not good:
            return set(), None
        if len(good) < len(kept):
            kept = [kept[k] for k in good]
            select = torch.as_tensor(good, device='cuda:0')
            images, planes = images[select], planes[select]
        shifts = [(float(shift[0]), float(shift[1])) for _, _, shift in kept]
        # Everything the CPU needs from the device, in one copy.
        parts, sizes = [], []
        count_saturated = calibration is None and config.reject_saturated and bit_depth <= 16
        if count_saturated:
            peak = (1 << int(bit_depth)) - 1
            samples = images if images.dtype.is_floating_point else images.to(torch.int32)  # No uint16 compare.
            parts.append((samples >= peak).flatten(1).sum(1).to(torch.float64))
        matcher = local_matcher._device()
        points = len(local_matcher.centres) > 0
        if points:
            ncc = matcher.correlations(planes, shifts)
            parts.append(ncc.reshape(-1).to(torch.float64))
        plan = None
        if moon_stack is not None:
            plan = moon_stack.plan([(index, shift) for (index, _, _), shift in zip(kept, shifts)], (h, w))
            parts.append(moon_stack.device_windows(
                planes, plan, upload=lambda a: device_sums.stage.upload(('moons', device_sums.groups % 2), a)))
        sizes = [len(part) for part in parts]
        host = torch.cat(parts).cpu().numpy() if parts else np.zeros(0)
        pieces = iter(np.split(host, np.cumsum(sizes)[:-1]) if parts else [])
        usable = np.ones(len(kept), bool)
        if count_saturated:
            usable &= next(pieces) / images[0].numel() <= .05
        ncc_host = next(pieces).reshape(ncc.shape) if points else None
        windows = next(pieces) if moon_stack is not None else None
        if not usable.any():
            return set(), None
        rows = np.flatnonzero(usable)
        if len(rows) < len(kept):
            select = torch.as_tensor(rows, device='cuda:0')
            images = images[select]
            ncc_host = ncc_host[rows] if points else None
        fields = matcher.solve(ncc_host, [shifts[k] for k in rows])
        if device_sums.kernels is not None:
            from planetrecon.detector import BAYER_PATTERNS
            labels = ([["RGB".index(c) for c in row] for row in BAYER_PATTERNS[color]] if bayer else None)
            contributions = device_sums.kernels.pull(images, fields, labels)
        else:
            contributions = backend.backproject_group(images, fields, color)
        indices, scores, pairs = [], [], []
        for f, k in enumerate(rows):
            index = kept[k][0]
            score = max(selection.measurements[index, 0], 1e-12)
            pending_quality[index] = score
            score = score if config.quality_weighting else 1.
            indices.append(index)
            scores.append(score)
            terms = (local_selection.pairs(index, device_sums.main) if local_selection is not None
                     else [(device_sums.main, np.zeros(0, np.int64), 1.)])
            pairs.extend((f, slot, chosen, outside, score) for slot, chosen, outside in terms)
        if moon_stack is not None:
            # While the GPU resamples: the moons' centroids on their windows of the registration planes.
            followed = moon_stack.follow([calibrated for _, calibrated, _ in kept], plan, windows)
            for k, index, score in zip(rows, indices, scores):
                if local_selection is not None:
                    # Outside the alignment points the global frame selection applies.
                    score = score * float(local_selection.global_mask[index])
                if score > 0:
                    pending_moons[index] = (followed[k], score)
        return set(indices), (indices, contributions.reshape(len(rows), h * w, -1), pairs)

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
        moon_contributions(index, calibrated, plane, shift, score)
        if local_matcher is not None:
            if local_matcher.use_cuda != (backend.name == 'cuda'):
                local_matcher.use_cuda = backend.name == 'cuda'
            try:
                global_shift = shift
                shift = (local_matcher.displacement_tensor(plane, shift) if backend.name == 'cuda'
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
            projected = cpu_backproject(calibrated, shift, color)
        if projected is not None:
            bins = [(2*b + fourier_weights.half(index), score * m) for b, m in local_selection.bin_weights(index)] if n_bins else None
            if local_selection is not None:
                # Per-pixel blend of the alignment points that selected this frame.
                score = score * local_selection.weights(index)
            return (score, *projected, bins)
        if bayer:
            add, wt = cfa_accumulate(calibrated, shift, color)
            return score, add, wt, None, None
        support = ndshift(np.ones((h, w)), shift=(-shift[1], -shift[0]),
                          order=1, prefilter=False, mode='grid-constant')
        if rgb:
            if calibrated.ndim != 3:
                raise ValueError('RGB source produced a 2-D frame')
            add = np.stack([ndshift(calibrated[..., c], shift=(-shift[1], -shift[0]),
                           order=1, prefilter=False, mode='grid-constant') for c in range(3)], axis=2)
            return score, add, support[..., None], None, None
        add = ndshift(calibrated, shift=(-shift[1], -shift[0]),
                      order=1, prefilter=False, mode='grid-constant')
        return score, add, support, None, None

    def accumulate(projected, main=True):
        """Add one frame's contributions; ``main=False`` adds only its Fourier rank-bin sums."""
        if projected is None:
            return
        score, add, wt, demo, support, *bins = projected
        scale = score[..., None] if np.ndim(score) == 2 and add.ndim == 3 else score
        if main:
            accum[...] += scale * add
            weight[...] += scale * wt
        for b, m in (bins[0] or []) if bins else []:
            m = m[..., None] if add.ndim == 3 else m
            bin_accum[b] += m * add
            bin_weight[b] += m * wt

    from planetrecon.pipeline.cpu_pool import CPUFramePool
    with CPUFramePool(config, shape, n-next_index, enabled=backend.name == 'cpu',
                      should_cancel=should_cancel) as pool:
        if backend.name == 'cpu':
            report.frame_workers = pool.workers
        snapshot_provenance['cpu_frame_workers'] = {'workers': pool.workers, 'limit': pool.reason,
            'scratch_budget_bytes': pool.scratch_budget_bytes,
            'estimated_frame_bytes': pool.estimated_frame_bytes,
            'accumulation': 'capture order; float64; no frame brightness normalisation'}
        if local_matcher is not None and pool.workers > 1:
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

        # Selected frames in capture order, read ahead of the work; unselected frames are
        # skipped (counted as rejected from the gaps between indices).
        order = np.arange(next_index, n)
        if selection is not None:
            order = order[selection.accepted[next_index:]]
        raw_bytes = 2 * int(np.prod(shape))  # Raw samples are 16-bit integers.
        batches = reader.batches(order, min(config.batch_frames, max(1, (64 << 20) // raw_bytes)),
                                 should_cancel=should_cancel)
        last_checkpoint, dirty = [time.monotonic()], [False]

        def record(index, projected):
            """Count one consumed frame in capture order; ``projected`` is its CPU contributions,
            True when the device sums already hold it, or None when it was rejected."""
            nonlocal n_used, n_rejected
            # Screening exclusions may have been skipped while filling
            # the work batch. Account for exactly the consumed prefix.
            n_rejected += int(index) - (n_used + n_rejected)
            quality = pending_quality.pop(int(index), np.nan)
            moon_frame = pending_moons.pop(int(index), None)
            if projected is None or projected is False:
                n_rejected += 1
                return
            if projected is not True:
                accumulate(projected)
            if frame_quality is not None:
                frame_quality[int(index)] = quality
            if moon_frame is not None:
                moon_stack.add(*moon_frame)
            n_used += 1

        def save_checkpoint():
            resume.save(state_checkpoint, state_identity, {
                "accum": accum, "weight": weight, "reference": reference,
                "bin_accum": bin_accum, "bin_weight": bin_weight,
                "reference_index": reference_index, "n_used": n_used, "n_rejected": n_rejected,
                "next_index": n_used + n_rejected, resume.FRAME_QUALITY: frame_quality,
                "execution_history": report.execution_history, "warnings": warnings,
                "moons": moon_state if moon_stack is None else {**moon_state, **moon_stack.state()},
                "moon_accum": None if moon_stack is None else moon_stack.accum,
                "moon_weight": None if moon_stack is None else moon_stack.weight})
            last_checkpoint[0], dirty[0] = time.monotonic(), False

        for indices, batch in batches:
            if pool.cancelled():
                cancelled = True
                break
            dirty[0] = True
            try:
                if device_sums is not None and backend.name == 'cuda':
                    items = list(zip(indices, batch))
                    for start in range(0, len(items), group_frames):
                        if pool.cancelled():
                            cancelled = True
                            break
                        chunk = items[start:start + group_frames]
                        if backend.name != 'cuda':  # A CUDA failure earlier in this batch.
                            for index, raw in chunk:
                                record(index, process_frame((index, raw)))
                            continue
                        try:
                            used, payload = device_group(chunk)
                            if payload is not None:
                                device_sums.add(*payload)
                        except RuntimeError as exc:
                            for index, _ in chunk:
                                pending_quality.pop(int(index), None)
                                pending_moons.pop(int(index), None)
                            local_matcher.release_cuda()
                            local_matcher.use_cuda = False
                            cpu_fallback(exc)
                            for index, raw in chunk:
                                record(index, process_frame((index, raw)))
                            continue
                        for index, _ in chunk:
                            record(index, int(index) in used)
                else:
                    for index, projected in zip(indices, frame_results(zip(indices, batch))):
                        if pool.cancelled():
                            cancelled = True
                            break
                        record(index, projected)
            except InterruptedError:
                if not pool.cancelled():
                    raise
                cancelled = True
            cancelled = cancelled or pool.cancelled()
            # Checkpoints are written at most every CHECKPOINT_INTERVAL_S, on cancellation and at the end.
            save = state_checkpoint is not None and (
                cancelled or time.monotonic() - last_checkpoint[0] >= CHECKPOINT_INTERVAL_S)
            if device_sums is not None and (device_sums.pending or (save and device_sums.bin_pending)):
                try:
                    flush_device(everything=save)
                except RuntimeError as exc:
                    cpu_fallback(exc)
            if save:
                save_checkpoint()
            emit("baseline", incomplete=True)
            if cancelled:
                break
        if device_sums is not None and (device_sums.pending or device_sums.bin_pending):
            try:
                flush_device()
            except RuntimeError as exc:
                cpu_fallback(exc)
        if state_checkpoint is not None and dirty[0]:
            save_checkpoint()
        if not cancelled and not pool.cancelled():
            n_rejected += n - (n_used + n_rejected)

    cancelled = cancelled or bool(should_cancel is not None and should_cancel())
    image = _normalise_stack(accum, weight)
    fourier_detail = None
    if n_bins and n_used:
        image, fourier_detail = fourier_weights.apply(bin_accum, bin_weight, config.fourier_power)
    if moon_stack is not None:
        image = moon_stack.composite(image)
        snapshot_provenance['moons']['moons'] = moon_stack.provenance()
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
    if n_used == 0 and not cancelled:
        raise ValueError("no usable frames remain after rejection")
    if fourier_detail is not None:
        provenance['fourier_weighting'] = fourier_detail
    provenance["stack_quality"] = _stack_quality(source, calibration, config, frame_quality,
                                                 reference_selection, local_selection, fourier_detail is not None)
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


GROUP_FRAMES = 8  # Most frames matched, resampled and accumulated together on CUDA.
GROUP_BUDGET_BYTES = 1 << 30


def _group_frames(pixels, contribution_channels, memory_report):
    """Frames per CUDA group: their working set (raw frame, registration plane, proxy, fields,
    contributions and resampling terms) within GROUP_BUDGET_BYTES, or an eighth of a CUDA
    allocation budget."""
    per_frame = pixels * (8 + 8 + 12 + 16 + 8 * contribution_channels + 32)
    budget = GROUP_BUDGET_BYTES
    limit = (memory_report or {}).get('effective_bytes')
    if limit:
        budget = min(budget, int(limit) // 8)
    return int(max(1, min(GROUP_FRAMES, budget // per_frame)))


def _stack_quality(source, calibration, config, frame_quality, screened_selection, local_selection, fourier=False):
    """Frames, weighting, effective frame count and estimated SNR of a finished stack."""
    from planetrecon.pipeline.stack_quality import single_frame_snr, summarise
    if frame_quality is None:
        return {'unavailable': 'resumed from a checkpoint that predates per-frame quality records'}
    screened = None
    if screened_selection is not None:
        screened = np.where(screened_selection.accepted, screened_selection.measurements[:, 0], np.nan)
    used = np.flatnonzero(np.isfinite(frame_quality))
    return summarise(frame_quality, config.quality_weighting, screened=screened, local_selection=local_selection,
                     frame_snr=single_frame_snr(source, calibration, used),
                     fourier_power=config.fourier_power if fourier else None)


def _normalise_stack(accum: np.ndarray, weight: np.ndarray) -> np.ndarray:
    return np.divide(accum, weight, out=np.zeros_like(accum), where=weight > 0)
