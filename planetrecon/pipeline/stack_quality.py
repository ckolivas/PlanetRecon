"""Stack quality summary: frames, weighting, effective frame count and SNR.

Each accumulated frame contributes with its scalar weight (linear quality or
equal). The effective number of frames, (sum w)^2 / sum w^2, is the count of
equally weighted frames giving the same noise reduction, so stack noise falls
by sqrt(N_eff) relative to one frame for independent noise. With alignment
points every point stacks its own best frames, so the count and quality are
reported per point as well.

Single-frame SNR is measured on calibrated raw frames, where pixel-scale noise
dominates real structure in seeing-limited captures: the noise is a robust
median of a second-difference operator over the bright target, the signal is
the target's mean above the background. The stack estimate multiplies it by
sqrt(N_eff); interpolation smoothing during registration is not counted, so
it describes the uncorrelated noise per detector pixel.
"""
import numpy as np
from scipy.ndimage import convolve, uniform_filter

from planetrecon.calibration import apply_calibration
from planetrecon.detector import is_bayer

# Immerkaer's operator: zero response to planes and linear ramps; unit-variance
# noise gives a response of standard deviation 6.
NOISE_KERNEL = np.array([[1., -2., 1.], [-2., 4., -2.], [1., -2., 1.]])
NOISE_SAMPLES = 8
METHOD = ('effective frames (sum w)^2/sum w^2 of the accumulated frame weights; single-frame noise from a '
          'robust second-difference estimate on calibrated raw frames over the bright target; estimated stack '
          'SNR = single-frame SNR x sqrt(effective frames), excluding registration interpolation smoothing')


def _planes(frame, color_mode):
    """Same-colour sample planes, so the operator never mixes CFA colours."""
    if is_bayer(color_mode):
        return [frame[dy::2, dx::2] for dy in (0, 1) for dx in (0, 1)]
    if frame.ndim == 3:
        return [frame[..., c] for c in range(frame.shape[-1])]
    return [frame]


def frame_noise(frame, color_mode):
    """(signal above background, noise) of one calibrated frame, or None."""
    signals, variances = [], []
    for plane in _planes(np.asarray(frame, dtype=np.float64), color_mode):
        if min(plane.shape) < 8:
            continue
        response = convolve(plane, NOISE_KERNEL, mode='nearest')[1:-1, 1:-1] / 6.
        level = uniform_filter(plane, 3, mode='nearest')[1:-1, 1:-1]
        background = np.percentile(level, 5)
        top = np.percentile(level, 99.5)
        if not top > background:
            continue
        target = level > background + .5 * (top - background)
        if np.count_nonzero(target) < 64:
            continue
        noise = 1.4826 * np.median(np.abs(response[target]))
        signals.append(float(np.mean(level[target]) - background))
        variances.append(noise * noise)
    if not signals or not np.mean(variances) > 0:
        return None
    return float(np.mean(signals)), float(np.sqrt(np.mean(variances)))


def single_frame_snr(source, calibration, indices):
    """Median signal, noise and SNR over up to NOISE_SAMPLES evenly spaced frames."""
    indices = np.asarray(indices, dtype=int)
    if not len(indices):
        return None
    chosen = indices[np.linspace(0, len(indices) - 1, min(NOISE_SAMPLES, len(indices))).round().astype(int)]
    measured = []
    for index in np.unique(chosen):
        raw = source.read_raw(int(index))
        frame, _ = apply_calibration(raw, calibration)
        if not np.isfinite(frame).all():
            continue
        result = frame_noise(frame, source.color_mode())
        if result is not None:
            measured.append(result)
    if not measured:
        return None
    signal = float(np.median([s for s, _ in measured]))
    noise = float(np.median([n for _, n in measured]))
    return {'signal': signal, 'noise': noise, 'snr': signal / noise, 'frames_sampled': len(measured)}


def _effective(weights):
    weights = np.asarray(weights, dtype=np.float64)
    total = weights.sum()
    return float(total * total / np.sum(weights * weights)) if len(weights) and total > 0 else 0.


def _spread(values):
    values = np.asarray(values, dtype=np.float64)
    return {'median': float(np.median(values)), 'minimum': float(values.min()), 'maximum': float(values.max())}


def summarise(frame_quality, quality_weighting, *, screened=None, local_selection=None, frame_snr=None,
              fourier_power=None):
    """Summary of the frames accumulated so far.

    ``frame_quality``: capture-length scores, NaN for frames that did not
    contribute. ``screened``: optional capture-length quality of all frames
    that passed screening, for comparison with the stacked subset.
    ``fourier_power``: set when Fourier weighting re-weighted the stack per
    spatial frequency; the frame counts and SNR then describe its in-bin
    weights, which every scale starts from.
    """
    frame_quality = np.asarray(frame_quality, dtype=np.float64)
    used = np.flatnonzero(np.isfinite(frame_quality))
    quality = frame_quality[used]
    weights = np.maximum(quality, 1e-12) if quality_weighting else np.ones(len(used))
    summary = {'frames_stacked': int(len(used)),
               'weighting': (f'Fourier power {fourier_power:g}' if fourier_power is not None else
                             'linear quality' if quality_weighting else 'equal'),
               'method': METHOD}
    if not len(used):
        return summary
    effective = _effective(weights)
    summary['quality'] = {'mean': float(quality.mean()), 'weighted_mean': float(np.sum(weights * quality) / weights.sum()),
                          'minimum': float(quality.min()), 'maximum': float(quality.max())}
    if screened is not None:
        screened = np.asarray(screened, dtype=np.float64)
        screened = screened[np.isfinite(screened)]
        if len(screened):
            summary['quality']['screened_mean'] = float(screened.mean())
            summary['quality']['screened_frames'] = int(len(screened))
    summary['effective_frames'] = effective
    if local_selection is not None:
        rows = np.array([local_selection.row.get(int(i), -1) for i in used])
        known = rows >= 0
        chosen = np.zeros((len(used), local_selection.chosen.shape[1]), dtype=bool)
        chosen[known] = local_selection.chosen[rows[known]]
        counts = chosen.sum(0)
        if counts.any():
            live = np.flatnonzero(counts)
            per_point = [_effective(weights[chosen[:, a]]) for a in live]
            point_quality = [float(np.sum(weights[chosen[:, a]] * quality[chosen[:, a]]) / weights[chosen[:, a]].sum())
                             for a in live]
            summary['alignment_points'] = {'points': int(chosen.shape[1]), 'frames_per_point': _spread(counts[live]),
                                           'effective_frames': _spread(per_point),
                                           'weighted_quality': _spread(point_quality)}
            effective = summary['alignment_points']['effective_frames']['median']
    summary['snr_gain'] = float(np.sqrt(effective))
    if frame_snr is not None:
        summary['single_frame'] = dict(frame_snr)
        summary['estimated_snr'] = float(frame_snr['snr'] * np.sqrt(effective))
    return summary


def describe(summary):
    """One line for the desktop and command line."""
    n = summary.get('frames_stacked', 0)
    if not n or 'quality' not in summary:
        return f'{n} frames stacked'
    q = summary['quality']
    points = summary.get('alignment_points')
    of = f" of {q['screened_frames']} screened" if 'screened_frames' in q else ''
    if points:
        text = (f"each alignment point stacks {points['frames_per_point']['median']:.0f}{of} frames "
                f"({summary['weighting']} weights; effective {points['effective_frames']['median']:.0f}) · "
                f"quality weighted mean {points['weighted_quality']['median']:.4g}")
    else:
        text = (f"{n}{of} frames stacked ({summary['weighting']} weights; effective "
                f"{summary['effective_frames']:.0f}) · quality mean {q['mean']:.4g}, weighted {q['weighted_mean']:.4g}")
    if 'screened_mean' in q:
        text += f" (all screened {q['screened_mean']:.4g})"
    if 'estimated_snr' in summary:
        text += (f" · est. SNR {summary['estimated_snr']:.0f} "
                 f"(single frame {summary['single_frame']['snr']:.1f} × {summary['snr_gain']:.1f})")
    else:
        text += f" · SNR gain ×{summary['snr_gain']:.1f}"
    return text
