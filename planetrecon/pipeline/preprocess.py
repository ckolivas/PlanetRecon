"""Streaming capture screening before any reconstruction samples are accumulated.

The noise-robust squared-Laplacian is an adaptation of Kraaikamp's published
AutoStakkert estimator family, not a claim of bitwise AutoStakkert equivalence.
See docs/preprocessing.md for the exact reproducible algorithm and limitations.
"""
from dataclasses import dataclass
from contextlib import contextmanager
from functools import partial

import numpy as np
from scipy.ndimage import binary_dilation, gaussian_filter, label, laplace

from planetrecon.calibration import apply_calibration
from planetrecon.detector import is_bayer


def measurement_plane(frame, color):
    """2x2 area means suppress noise and remove the Bayer colour lattice.

    Each Bayer cell contains R, G, G, B, giving the same luminance weights as
    RGB (1/4, 1/2, 1/4), without interpolating detector samples for stacking.
    """
    plane = np.asarray(frame, dtype=np.float64)
    if plane.ndim == 3:
        plane = .25 * plane[..., 0] + .5 * plane[..., 1] + .25 * plane[..., 2]
    h, w = plane.shape
    if min(h, w) >= 10 or is_bayer(color):
        return plane[:h//2*2, :w//2*2].reshape(h//2, 2, w//2, 2).mean(axis=(1, 3)), 2
    return plane, 1


def measure_frame(frame, color):
    """Return quality, silhouette width/height in detector px, angle and status.

    The largest connected illuminated silhouette excludes detached moons and
    hot pixels. Principal axes follow rings/phase/roll; these are apparent
    dimensions, not an inferred physical equator or circular globe diameter.
    """
    plane, scale = measurement_plane(frame, color)
    smooth = gaussian_filter(plane, 1.0, mode='nearest')
    border = np.concatenate((smooth[0], smooth[-1], smooth[1:-1, 0], smooth[1:-1, -1]))
    sky = float(np.median(border))
    noise = 1.4826 * float(np.median(np.abs(border - sky)))
    peak = float(smooth.max())
    contrast = peak - sky
    if contrast <= max(6 * noise, np.finfo(float).eps * max(abs(peak), 1)):
        return (np.nan, np.nan, np.nan, np.nan, 'no_target')
    labels, count = label(smooth > sky + max(.15 * contrast, 6 * noise))
    sizes = np.bincount(labels.ravel(), minlength=count+1)
    sizes[0] = 0
    mask = labels == int(sizes.argmax())
    if sizes.max() < 9:
        return (np.nan, np.nan, np.nan, np.nan, 'no_target')
    yy, xx = np.nonzero(mask)
    if xx.min() == 0 or yy.min() == 0 or xx.max() == plane.shape[1]-1 or yy.max() == plane.shape[0]-1:
        return (np.nan, np.nan, np.nan, np.nan, 'clipped_target')
    points = np.column_stack((xx - xx.mean(), yy - yy.mean()))
    _, axes = np.linalg.eigh(points.T @ points / len(points))
    # The major apparent axis becomes horizontal, including a ring system or
    # the long axis of a crescent. Do not force width == height.
    axis = axes[:, 1]
    angle = float(np.arctan2(axis[1], axis[0]) % np.pi)
    projected = points @ axes[:, ::-1]
    # Pixel squares have a projected extent, too; include it at both limbs.
    extent = np.ptp(projected, axis=0) + np.abs(axes[:, ::-1]).sum(axis=0)
    width, height = extent * scale
    support = binary_dilation(mask, iterations=2)
    support[[0, -1], :] = False
    support[:, [0, -1]] = False
    quality = float(np.mean(laplace(smooth, mode='nearest')[support] ** 2))
    return quality, float(width), float(height), angle, 'ok'


def sigma_selection(measurements, valid):
    """One simultaneous population-2σ decision; equality is retained."""
    measurements = np.asarray(measurements, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(measurements[:, :3]).all(axis=1)
    rejected = np.zeros((len(valid), 3), dtype=bool)
    stats = {}
    for column, name in enumerate(('quality', 'width_px', 'height_px')):
        values = measurements[valid, column]
        if not len(values):
            stats[name] = None
            continue
        mean, std = float(values.mean()), float(values.std(ddof=0))
        low, high = mean - 2 * std, mean + 2 * std
        # Identical measurements must survive tiny floating-point roundoff.
        tolerance = 32 * np.finfo(float).eps * max(abs(mean), 1)
        rejected[:, column] = valid & (measurements[:, column] < low - tolerance)
        if column:
            rejected[:, column] |= valid & (measurements[:, column] > high + tolerance)
        stats[name] = {'mean': mean, 'std': std, 'lower': low,
                       'upper': high if column else None}
    return valid & ~rejected.any(axis=1), rejected, stats


@dataclass
class FrameSelection:
    accepted: np.ndarray
    measurements: np.ndarray
    summary: dict
    cancelled: bool = False
    identity: dict | None = None
    digest: str | None = None

    @property
    def best_reference_index(self) -> int | None:
        """Sharpest retained observation; equal scores choose the earliest."""
        indices = np.flatnonzero(self.accepted & np.isfinite(self.measurements[:, 0]))
        return int(indices[np.argmax(self.measurements[indices, 0])]) if indices.size else None


def _measure_observation(raw, *, color, bit_depth, reject_saturated, calibration):
    from planetrecon.pipeline.baseline import _saturated
    values, status = (np.nan,)*4, 'invalid'
    if np.isfinite(raw).all():
        if reject_saturated and _saturated(raw, bit_depth):
            status = 'saturated'
        else:
            calibrated, info = apply_calibration(raw, calibration)
            if np.isfinite(calibrated).all():
                if reject_saturated and info['saturated']:
                    status = 'saturated'
                else:
                    *values, status = measure_frame(calibrated, color)
                    if status == 'ok' and not np.isfinite(values).all():
                        status = 'invalid'
    return values, status


@contextmanager
def _measurement_pool(source, config):
    # At most four active frames, bounded by both the requested thread count
    # and a conservative 128 MiB estimate for concurrent measurement scratch.
    pixels = max(1, int(np.prod(source.frame_shape())))
    workers = max(1, min(4, config.threads, config.batch_frames, source.n_frames(),
                         (128*1024**2)//(128*pixels)))
    if workers > 1:
        try:
            from threadpoolctl import threadpool_limits
        except ImportError:
            pass  # Optional runtime: keep serial execution without nested-pool control.
        else:
            from concurrent.futures import ThreadPoolExecutor
            with threadpool_limits(limits=1), ThreadPoolExecutor(max_workers=workers) as pool:
                yield pool
            return
    yield None


def screen_source(source, config, calibration=None, *, should_cancel=None, on_progress=None):
    """Read serial bounded batches; measure independent frames in a bounded pool."""

    n = source.n_frames()
    metrics = np.full((n, 4), np.nan)
    statuses = np.full(n, 'unmeasured', dtype='<U20')
    processed = 0
    stopped = False
    bit_depth = None
    with _measurement_pool(source, config) as pool:
        for indices, batch in source.iter_batches(config.batch_frames, should_cancel=should_cancel):
            if should_cancel is not None and should_cancel():
                stopped = True
                break
            if config.reject_saturated and bit_depth is None and any(np.isfinite(raw).all() for raw in batch):
                # Metadata can seek/read the SER trailer. Keep all source I/O
                # on this thread and reuse its fixed bit depth for this pass.
                bit_depth = source.metadata().bit_depth
            measure = partial(_measure_observation, color=source.color_mode(), bit_depth=bit_depth,
                              reject_saturated=config.reject_saturated, calibration=calibration)
            results = pool.map(measure, batch) if pool is not None else map(measure, batch)
            try:
                for index in indices:
                    if should_cancel is not None and should_cancel():
                        stopped = True
                        break
                    values, status = next(results)
                    metrics[index], statuses[index] = values, status
                    processed += 1
            finally:
                if pool is not None:
                    results.close()  # Cancel queued work on cancellation or an exception.
            if on_progress is not None:
                on_progress(processed, n)
            if stopped:
                break
    stopped = stopped or processed < n or bool(should_cancel and should_cancel())
    accepted, cuts, stats = sigma_selection(metrics, statuses == 'ok')
    reasons = {name: np.flatnonzero(statuses == name).tolist()
               for name in ('invalid', 'saturated', 'no_target', 'clipped_target')}
    reasons.update({name: np.flatnonzero(cuts[:, i]).tolist()
                    for i, name in enumerate(('low_quality', 'width_outlier', 'height_outlier'))})
    summary = {
        'method': 'Kraaikamp-style noise-robust squared Laplacian adaptation v1',
        'noise_filter': '2x2 area bin then Gaussian sigma=1 binned pixel',
        'size_method': '15% contrast largest silhouette; principal-axis projected extents',
        'sigma': 2.0, 'std_ddof': 0, 'statistics': stats,
        'n_measured': processed, 'n_total': n, 'n_accepted': int(accepted.sum()),
        'n_rejected': int(processed - accepted.sum()), 'complete': not stopped,
        'rejected_indices_by_reason': reasons,
        'axis_note': 'Apparent major/minor axes; phase and rings included, physical equator not inferred.',
    }
    return FrameSelection(accepted, metrics, summary, stopped)


def quality_range(selection):
    """Worst/best finite quality measured across the capture, before screening."""
    scores = selection.measurements[:, 0]
    scores = scores[np.isfinite(scores)]
    return (float(scores.min()), float(scores.max())) if scores.size else (None, None)


def quality_range_counts(selection):
    low, high = quality_range(selection)
    scores = np.sort(selection.measurements[selection.accepted, 0])
    if low is None or high == low:
        counts = [len(scores)] * 100
    else:
        cutoffs = low + (high-low)*(1-np.arange(1, 101)/100.)
        counts = (len(scores)-np.searchsorted(scores, cutoffs, side='right')).tolist()
        counts[-1] = len(scores)  # 100% retains all screened frames, including the minimum.
    return {'minimum': low, 'maximum': high, 'retained_counts': counts,
            'range_scope': 'all finite measured capture qualities; screening still applies'}


def best_frame_mask(selection, percent, mode='frame_count'):
    """Rank screened frames without overwriting cached measurements or decisions.

    Count mode rounds up and breaks ties by frame index. Quality-range mode keeps
    scores strictly above low + (high-low)*(1-percent/100), then applies screening.
    A flat quality range and 100% retain all screened frames.
    """
    if type(percent) is not int or not 1 <= percent <= 100:
        raise ValueError('stack_percent must be an integer from 1 to 100')
    if mode not in ('frame_count', 'quality_range'):
        raise ValueError('unknown frame_selection_mode')
    indices = np.flatnonzero(selection.accepted)
    scores = selection.measurements[indices, 0]
    if not np.isfinite(scores).all():
        raise ValueError('retained preprocessing quality scores must be finite')
    if mode == 'quality_range':
        low, high = quality_range(selection)
        if low is None or low == high or percent == 100:
            return selection.accepted.copy()
        cutoff = low + (high-low)*(1-percent/100.)
        return selection.accepted & (selection.measurements[:, 0] > cutoff)
    count = (len(indices) * percent + 99) // 100
    order = np.argsort(-scores, kind='stable')
    mask = np.zeros_like(selection.accepted)
    mask[indices[order[:count]]] = True
    return mask
