"""Independent preprocessing and atomic, input-validated reusable measurements."""
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from zipfile import BadZipFile

import numpy as np

from planetrecon.calibration import load_calibration
from planetrecon.pipeline.preprocess import FrameSelection, screen_source
from planetrecon.pipeline.provenance import capture_provenance

SCHEMA = 'planetrecon-preprocessing-1'
GEOMETRY_KEYS = ('cadence_s', 'exposure_s', 'field_angle0_rad', 'field_rate_rad_s',
                 'reference_epoch_s', 'equatorial_radius_px', 'sub_obs_lat_rad', 'pole_pa_rad')


def default_cache_path(source):
    path = Path(source.metadata().path)
    return path.with_name(path.name + '.planetrecon-preprocess.npz') if path.is_file() else None


def calibration_for(source, config, calibration=None):
    configured = any(getattr(config, key) is not None for key in
                     ('bias_path', 'dark_path', 'flat_path', 'gain_e_per_adu', 'read_noise_e', 'saturate_adu'))
    if configured:
        if calibration is not None:
            raise ValueError('use either calibration settings or an explicit Calibration, not both')
        return load_calibration(config, source.frame_shape())
    return calibration


def identity(source, config, calibration, should_cancel=None):
    # Hash observed pixels, not just file timestamps or a few samples. This also
    # distinguishes crops/indexed sources backed by the very same capture file.
    digest = hashlib.sha256()
    for i in range(source.n_frames()):
        if should_cancel and should_cancel():
            raise InterruptedError('preprocessing cache verification cancelled')
        frame = np.ascontiguousarray(source.read_raw(i))
        digest.update(frame.dtype.str.encode())
        digest.update(memoryview(frame).cast('B'))
    times = source.timestamps()
    time_hash = None if times is None else hashlib.sha256(np.ascontiguousarray(times).tobytes()).hexdigest()
    return {'schema': SCHEMA, 'pixels_sha256': digest.hexdigest(),
            'shape': list(source.frame_shape()), 'n_frames': source.n_frames(),
            'color_mode': source.color_mode(), 'bit_depth': source.metadata().bit_depth,
            'units': source.metadata().units, 'timestamps_sha256': time_hash,
            'timestamp_scale_s': source.timestamp_scale_s(),
            'reject_saturated': config.reject_saturated,
            'calibration': capture_provenance(source, config, calibration)['calibration']}


def selection_digest(selection):
    h = hashlib.sha256(json.dumps({'identity': selection.identity, 'summary': selection.summary},
                                sort_keys=True, allow_nan=False).encode())
    h.update(selection.accepted.tobytes())
    h.update(selection.measurements.tobytes())
    return h.hexdigest()


def reconstruction_digest(selection):
    """Only accepted indices/weights and input identity affect accumulator reuse."""
    h = hashlib.sha256(json.dumps(selection.identity, sort_keys=True, allow_nan=False).encode())
    h.update(selection.accepted.tobytes())
    h.update(selection.measurements[selection.accepted, 0].tobytes())
    return h.hexdigest()


def exclusion_counts(selection):
    reasons = selection.summary['rejected_indices_by_reason']
    quality = set(reasons['low_quality'])
    shape = set(reasons['width_outlier']) | set(reasons['height_outlier']) | set(reasons['clipped_target']) | set(reasons['no_target'])
    total = int((~selection.accepted).sum())
    return {'n_total': len(selection.accepted), 'accepted': int(selection.accepted.sum()),
            'quality': len(quality), 'shape': len(shape), 'quality_shape_overlap': len(quality & shape),
            'other': len(set(reasons['invalid']) | set(reasons['saturated'])), 'excluded': total}


def cache_report(selection, path=None, config=None):
    estimate = selection.summary.get('geometry_estimate', {})
    if config is not None:
        original = selection.summary.get('geometry_analysis_config', {})
        for key, value in original.items():
            current = getattr(config, key)
            suggested = estimate.get('suggestions', {}).get(key, value)
            def same(a, b):
                return (math.isclose(a, b, rel_tol=1e-8, abs_tol=1e-10)
                        if isinstance(a, (int, float)) and isinstance(b, (int, float)) else a == b)
            if not same(current, value) and not same(current, suggested):
                estimate = {'suggestions': {}, 'applicable': False, 'notes': ['Geometry settings changed; run Preprocess to refresh geometry estimates. Quality/shape exclusions remain valid.']}
                break
    return {'status': 'ready', 'path': None if path is None else str(path),
            'digest': selection.digest, **exclusion_counts(selection),
            'geometry_estimate': estimate}


def save_cache(path, selection, source, config):
    path = Path(path)
    for value in (source.metadata().path, config.bias_path, config.dark_path, config.flat_path):
        if value and (path.resolve() == Path(value).resolve() or
                      (path.exists() and Path(value).exists() and path.samefile(value))):
            raise ValueError('preprocessing cache cannot replace a capture or calibration input')
    if path.exists():
        try:
            with np.load(path, allow_pickle=False) as data:
                owned = json.loads(str(data['metadata'])).get('schema') == SCHEMA
        except (ValueError, OSError, KeyError, TypeError, AttributeError, BadZipFile):
            owned = False
        if not owned and path != default_cache_path(source):
            raise FileExistsError('cache destination contains another file; choose a different cache path')
    meta = {'schema': SCHEMA, 'identity': selection.identity, 'summary': selection.summary,
            'digest': selection.digest}
    fd, tmp = tempfile.mkstemp(prefix='.'+path.name, suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            np.savez_compressed(stream, accepted=selection.accepted, measurements=selection.measurements,
                                metadata=json.dumps(meta, sort_keys=True, allow_nan=False))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def load_cache(source, config, calibration=None, *, path=None, should_cancel=None):
    path = Path(path) if path is not None else default_cache_path(source)
    if path is None or not path.is_file():
        return None, {'status': 'missing', 'reason': 'No preprocessing cache; run Preprocess to measure exclusions.'}
    if calibration is None:
        calibration = calibration_for(source, config)
    try:
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data['metadata']))
            if meta['schema'] != SCHEMA:
                raise ValueError('unsupported cache schema')
            selection = FrameSelection(data['accepted'].copy(), data['measurements'].copy(),
                                       meta['summary'], identity=meta['identity'], digest=meta['digest'])
        n = source.n_frames()
        if (selection.accepted.dtype != np.bool_ or selection.accepted.shape != (n,)
                or selection.measurements.dtype != np.float64 or selection.measurements.shape != (n, 4)
                or not selection.summary['complete'] or selection.summary['n_measured'] != n
                or not np.isfinite(selection.measurements[selection.accepted]).all()
                or selection_digest(selection) != selection.digest):
            raise ValueError('invalid or incomplete cache measurements')
        if selection.identity != identity(source, config, calibration, should_cancel):
            return None, {'status': 'stale', 'reason': 'Capture interpretation or calibration changed; run Preprocess again.'}
        return selection, cache_report(selection, path, config)
    except (ValueError, OSError, KeyError, TypeError, BadZipFile) as exc:
        return None, {'status': 'invalid', 'reason': f'Preprocessing cache unavailable: {exc}'}


def preprocess_source(source, config, *, calibration=None, cache_path=None,
                      should_cancel=None, on_progress=None):
    """Compute fresh screening + geometry, cache if file-backed, and return it.

    Callable independently of reconstruction, even when cache use is disabled.
    Cancellation never replaces an existing cache. Array sources return a
    reusable in-memory selection; callers can also choose an explicit path.
    """
    from planetrecon.geometry.discovery import discover_geometry
    from planetrecon.memory import cpu_memory_limit
    from planetrecon.runtime import apply_thread_limits
    apply_thread_limits(config.threads)
    with cpu_memory_limit(config.max_ram_bytes):
        if source.n_frames() < 1 or min(source.frame_shape()[:2]) < 5:
            raise ValueError('preprocessing requires frames of at least 5 by 5 pixels')
        if source.color_mode() not in ('mono', 'RGB', 'BGR', 'RGGB', 'BGGR', 'GRBG', 'GBRG'):
            raise ValueError('unsupported preprocessing color mode')
        calibration = calibration_for(source, config, calibration)
        if source.metadata().units == 'e-' and calibration and calibration.gain_e_per_adu is not None:
            raise ValueError('gain calibration cannot be applied to observations already in electrons')
        before = identity(source, config, calibration, should_cancel)
        selection = screen_source(source, config, calibration, should_cancel=should_cancel, on_progress=on_progress)
        if selection.cancelled:
            raise InterruptedError('preprocessing cancelled')
        selection.summary['geometry_estimate'] = discover_geometry(source, config, selection, calibration, should_cancel)
        selection.summary['geometry_analysis_config'] = {key: getattr(config, key) for key in GEOMETRY_KEYS}
        selection.identity = identity(source, config, calibration, should_cancel)
        if selection.identity != before:
            raise ValueError('capture changed during preprocessing; no cache was saved')
        selection.digest = selection_digest(selection)
        path = cache_path if cache_path is not None else default_cache_path(source)
        if should_cancel and should_cancel():
            raise InterruptedError('preprocessing cancelled')
        if path is not None:
            save_cache(path, selection, source, config)
        return selection
