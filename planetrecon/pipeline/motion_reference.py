"""Deterministic motion anchors and bounded reference pools in capture time."""
import numpy as np

from planetrecon.geometry.pose import source_times_s


def motion_reference_plan(source, config, selection=None):
    """Use screened frames before the optional stack-count/quality cutoff.

    Keeping this pool independent of that cutoff gives geometry discovery and
    stacking the same detector origin, even when the sharpest frames occur late.
    Only the template uses this pool; accumulation still uses the chosen mask.
    """
    times, origin = source_times_s(source, cadence_s=config.cadence_s)
    eligible = (np.arange(len(times)) if selection is None
                else np.flatnonzero(selection.accepted))
    if not len(eligible):
        raise ValueError('no usable frames remain for the motion reference')
    quality = (np.ones(len(times)) if selection is None
               else np.asarray(selection.measurements[:, 0]))
    if not np.isfinite(quality[eligible]).all():
        raise ValueError('motion reference quality must be finite')
    target = float((times[0] + times[-1]) / 2.)
    explicit = config.reference_index != 0
    if explicit:
        if config.reference_index not in eligible:
            raise ValueError('selected reference frame was rejected by preprocessing')
        target = float(times[config.reference_index])
    distance = np.abs(times[eligible] - target)
    nearest = eligible[np.lexsort((eligible, -quality[eligible], distance))]
    if config.motion_reference == 'best' and not explicit:
        pool = eligible
    else:
        pool = eligible[distance <= .1 * (times[-1] - times[0])]
        # Tiny clips and sparse screening still need four template observations.
        if len(pool) < min(4, len(eligible)):
            pool = nearest[:min(4, len(eligible))]
    if explicit:
        anchor = int(config.reference_index)
    elif config.motion_reference == 'best':
        anchor = int(pool[np.lexsort((pool, -quality[pool]))[0]])
    else:
        good = pool[quality[pool] >= np.median(quality[pool])]
        anchor = int(good[np.lexsort((good, -quality[good], np.abs(times[good]-target)))[0]])
    ranked = pool[np.lexsort((pool, np.abs(times[pool]-target), -quality[pool]))]
    return {
        'version': 1, 'policy': 'manual' if explicit else config.motion_reference,
        'anchor_index': anchor, 'anchor_time': float(times[anchor]),
        'target_time': target, 'time_basis': 'frame index' if origin == 'inferred' else 'seconds from start',
        'template_candidates': ranked[:64].tolist(),
        'pool': 'screened frames before optional stacking cutoff',
        'window': ('whole capture' if config.motion_reference == 'best' and not explicit else
                   'middle 20% of capture duration, expanded to four nearest screened frames if needed'
                   if not explicit else '20% of capture duration around manual anchor, expanded to four if needed'),
    }
