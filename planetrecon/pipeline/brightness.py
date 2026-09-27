"""One output gain after stacking; observations and accumulators stay unchanged."""
import math

import numpy as np


def detector_full_scale(source, config, calibration=None):
    meta = source.metadata()
    if meta.units != 'adu' or not 1 <= meta.bit_depth <= 16:
        raise ValueError('Brightness normalisation requires a capture with a known integer ADU range.')
    gain = config.gain_e_per_adu
    if gain is None and calibration is not None:
        gain = calibration.gain_e_per_adu
    return float((1 << meta.bit_depth) - 1) * (gain if gain is not None else 1.)


def normalise_result(result, percent, full_scale):
    """Map the supported stack peak to the requested detector-range fraction."""
    if result.provenance.get('brightness_normalisation', {}).get('applied'):
        return result
    supported = result.validity & np.isfinite(result.image)
    peak = float(np.max(result.image, where=supported, initial=0.))
    target = full_scale * percent / 100.
    gain = target / peak if peak > 0 else 1.
    if not math.isfinite(gain):
        raise ValueError('Brightness normalisation gain is not finite.')
    applied = peak > 0 and result.n_used > 0
    if applied:
        result.image = result.image * gain
    result.provenance['brightness_normalisation'] = {
        'applied': applied, 'percent': percent, 'full_scale': full_scale,
        'original_peak': peak, 'target_peak': target, 'gain': gain if applied else 1.,
        'scope': 'single shared output gain after stacking; input frames and accumulator weights unchanged',
        'peak_scope': 'finite supported samples across all channels',
    }
    return result
