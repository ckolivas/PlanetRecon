"""Independent preprocessing and atomic, input-validated reusable measurements."""
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from dataclasses import dataclass
from zipfile import BadZipFile

import numpy as np

from planetrecon.calibration import load_calibration
from planetrecon.pipeline.preprocess import FrameSelection, screen_source, quality_range_counts
from planetrecon.pipeline.provenance import capture_provenance
from planetrecon.pipeline.capture_hash import HASH_METHOD, LEGACY_HASH_METHOD, pixel_hash

SCHEMA = 'planetrecon-preprocessing-1'
GEOMETRY_KEYS = ('cadence_s', 'exposure_s', 'field_angle0_rad', 'field_rate_rad_s',
                 'reference_epoch_s', 'equatorial_radius_px', 'sub_obs_lat_rad', 'pole_pa_rad', 'rotation_planet',
                 'motion_reference', 'reference_index')


def _file_stamp(path):
    try:
        path = Path(path).resolve()
        stat = path.stat()
        return (str(path), stat.st_dev, stat.st_ino, stat.st_size,
                stat.st_mtime_ns, stat.st_ctime_ns)
    except OSError:
        return None


def _validation_key(source, config, calibration):
    # Only concrete file adapters qualify. Array/indexed/custom sources can
    # change independently of the path in their metadata and must be hashed.
    from planetrecon.io.ser import SERSource
    from planetrecon.io.avi import AVISource
    from planetrecon.io.hdf5_source import HDF5ObservedSource
    if type(source) not in (SERSource, AVISource, HDF5ObservedSource):
        return None
    stamp = _file_stamp(source.metadata().path)
    if stamp is None:
        return None
    return (stamp, json.dumps({
        'adapter': type(source).__name__, 'metadata': source.metadata().as_dict(),
        'reject_saturated': config.reject_saturated,
        'calibration': capture_provenance(source, config, calibration)['calibration'],
    }, sort_keys=True))


@dataclass
class CacheValidation:
    """Session-only proof of a full check, passed between desktop workers.

    Never store this object in settings, preprocessing files or result provenance.
    A separate cache-bound digest permits file-metadata reuse across sessions.
    Both the cache digest and file stamps are still checked on every use.
    """
    source_key: object = None
    cache_stamp: object = None
    digest: str | None = None

    def clear(self):
        self.source_key = self.cache_stamp = self.digest = None

    def remember(self, key, path, selection):
        self.source_key = key
        self.cache_stamp = _file_stamp(path) if path is not None else None
        self.digest = selection.digest


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


def identity(source, config, calibration, should_cancel=None, on_progress=None, *,
             hash_method=HASH_METHOD, legacy_identity=None):
    # Hash observed pixels, not just file timestamps or a few samples. This also
    # distinguishes crops/indexed sources backed by the very same capture file.
    pixels, legacy = pixel_hash(source, config, should_cancel, on_progress,
                                method=hash_method, include_legacy=legacy_identity is not None)
    times = source.timestamps()
    time_hash = None if times is None else hashlib.sha256(np.ascontiguousarray(times).tobytes()).hexdigest()
    result = {'schema': SCHEMA, 'pixels_sha256': pixels,
            'shape': list(source.frame_shape()), 'n_frames': source.n_frames(),
            'color_mode': source.color_mode(), 'bit_depth': source.metadata().bit_depth,
            'units': source.metadata().units, 'timestamps_sha256': time_hash,
            'timestamp_scale_s': source.timestamp_scale_s(),
            'reject_saturated': config.reject_saturated,
            'calibration': capture_provenance(source, config, calibration)['calibration']}
    if legacy_identity is not None:
        legacy_identity.update(result, pixels_sha256=legacy)
    if hash_method != LEGACY_HASH_METHOD:
        result['pixels_hash_method'] = hash_method
    return result


def _validation_digest(selection):
    """Bind a migrated verification identity to the unchanged cached selection."""
    def without_pixels(data):
        return {k: v for k, v in data.items() if k not in ('pixels_sha256', 'pixels_hash_method')}
    if (selection.validation_identity.get('pixels_hash_method') != HASH_METHOD
            or without_pixels(selection.validation_identity) != without_pixels(selection.identity)):
        raise ValueError('cache verification interpretation does not match its measurements')
    return hashlib.sha256(json.dumps([selection.digest, selection.validation_identity],
                                    sort_keys=True, allow_nan=False).encode()).hexdigest()


def _source_validation_digest(selection, key):
    """Bind an established full check to file metadata and cached measurements."""
    return hashlib.sha256(json.dumps(['source-validation-v1', selection.digest, key],
                                    sort_keys=True, allow_nan=False).encode()).hexdigest()


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


def quality_plot_data(selection):
    """Per-frame display data, sent once by workers, never added to provenance."""
    return {'scores': selection.measurements[:, 0].copy(),
            'accepted': selection.accepted.copy(),
            'reasons': selection.summary['rejected_indices_by_reason']}


def cache_report(selection, path=None, config=None, source=None):
    estimate = selection.summary.get('geometry_estimate', {})
    if config is not None:
        if config.geometry_mode == 'saturn':
            from planetrecon.geometry.discovery import without_assumed_saturn_view
            estimate = without_assumed_saturn_view(estimate)
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
        if (config.geometry_mode == 'saturn' and 'flattening' in estimate.get('suggestions', {})
                and estimate.get('saturn_geometry', {}).get('status') != 'estimated'):
            estimate = {**estimate, 'suggestions': {k: v for k, v in estimate['suggestions'].items() if k != 'flattening'},
                        'notes': [*estimate.get('notes', []), 'Cached whole-silhouette flattening is not used for Saturn globe geometry.']}
    if (config is not None and 'reference_index' in estimate
            and 'motion_reference' not in selection.summary.get('geometry_analysis_config', {})):
        estimate = {'suggestions': {}, 'applicable': False,
                    'notes': ['Motion reference selection changed; run Preprocess to refresh geometry. Quality/shape exclusions remain valid.']}
    if 'saturn_geometry' in estimate:
        from planetrecon.geometry.saturn_fit import SATURN_FIT_VERSION
        if estimate['saturn_geometry'].get('estimator_version') != SATURN_FIT_VERSION:
            estimate = {'suggestions': {}, 'applicable': False,
                        'notes': ['Saturn boundary estimator changed; run Preprocess to refresh geometry. Quality/shape exclusions remain valid.']}
    if source is not None and config is not None:
        view = estimate.get('viewing_geometry', {})
        if view.get('origin') == 'jpl_horizons_geocentric':
            from planetrecon.geometry.viewing import capture_epoch_jd
            planet = 'saturn' if config.geometry_mode == 'saturn' else config.rotation_planet
            try:
                epoch = capture_epoch_jd(source, config.cadence_s)
            except ValueError:
                epoch = None
            if planet != view['planet'] or epoch != view['epoch_jd_utc']:
                estimate = {'suggestions': {}, 'applicable': False,
                            'notes': ['Planet or capture UTC changed; run Preprocess to refresh viewing geometry.']}
        from planetrecon.geometry.pose import capture_exposure
        recorded = selection.summary.get('geometry_estimate', {}).get('exposure')
        if recorded is not None and recorded != capture_exposure(source, config.exposure_s):
            estimate = {'suggestions': {}, 'applicable': False,
                        'notes': ['Exposure changed; run Preprocess to refresh geometry estimates. Quality/shape exclusions remain valid.']}
    timing = selection.summary.get('timing', {})
    if source is not None:
        from planetrecon.geometry.pose import capture_timing
        timing = capture_timing(source, cadence_s=config.cadence_s if config is not None else None)
    return {'status': 'ready', 'path': None if path is None else str(path),
            'digest': selection.digest, **exclusion_counts(selection),
            'best_reference_index': selection.best_reference_index,
            'quality_range': quality_range_counts(selection),
            'execution': selection.summary.get('execution'),
            'timing': timing,
            'geometry_estimate': estimate}


def save_cache(path, selection, source, config, *, validation_key=None):
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
    if selection.validation_identity is not None:
        meta['validation_identity'] = selection.validation_identity
        meta['validation_digest'] = _validation_digest(selection)
    if validation_key is not None:
        meta['source_validation_digest'] = _source_validation_digest(selection, validation_key)
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


def load_cache(source, config, calibration=None, *, path=None, should_cancel=None, on_progress=None,
               validation=None, force_full_validation=False):
    """Use verified file metadata by default; hash fully on changes or request."""
    previous = (validation.source_key, validation.cache_stamp, validation.digest) if validation else None
    if validation is not None:
        validation.clear()
    path = Path(path) if path is not None else default_cache_path(source)
    if path is None or not path.is_file():
        return None, {'status': 'missing', 'reason': 'No preprocessing cache; run Preprocess to measure exclusions.'}
    if should_cancel and should_cancel():
        raise InterruptedError('preprocessing cache verification cancelled')
    if calibration is None:
        calibration = calibration_for(source, config)
    try:
        key = _validation_key(source, config, calibration)
        stamp = _file_stamp(path)
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data['metadata']))
            if meta['schema'] != SCHEMA:
                raise ValueError('unsupported cache schema')
            selection = FrameSelection(data['accepted'].copy(), data['measurements'].copy(),
                meta['summary'], identity=meta['identity'], digest=meta['digest'],
                validation_identity=meta.get('validation_identity'))
        n = source.n_frames()
        if (not isinstance(selection.identity, dict)
                or selection.accepted.dtype != np.bool_ or selection.accepted.shape != (n,)
                or selection.measurements.dtype != np.float64 or selection.measurements.shape != (n, 4)
                or not selection.summary['complete'] or selection.summary['n_measured'] != n
                or not np.isfinite(selection.measurements[selection.accepted]).all()
                or selection_digest(selection) != selection.digest):
            raise ValueError('invalid or incomplete cache measurements')
        if (selection.validation_identity is not None
                and (not isinstance(selection.validation_identity, dict)
                     or meta.get('validation_digest') != _validation_digest(selection))):
            raise ValueError('invalid cache verification identity')
        persisted = (key is not None and meta.get('source_validation_digest')
                     == _source_validation_digest(selection, key))
        reused = not force_full_validation and (persisted or
                    key is not None and previous == (key, stamp, selection.digest))
        upgraded = False
        if not reused:
            expected = selection.validation_identity or selection.identity
            old = {} if 'pixels_hash_method' not in expected else None
            current = identity(source, config, calibration, should_cancel, on_progress,
                               hash_method=expected.get('pixels_hash_method', HASH_METHOD), legacy_identity=old)
            if expected != (old if old is not None else current):
                return None, {'status': 'stale', 'reason': 'Capture interpretation or calibration changed; run Preprocess again.'}
            if old is not None:
                selection.validation_identity = current
                upgraded = True
        if key != _validation_key(source, config, calibration) or stamp != _file_stamp(path):
            raise ValueError('capture or cache changed during validation; retry')
        if should_cancel and should_cancel():
            raise InterruptedError('preprocessing cache verification cancelled')
        if upgraded or (key is not None and not persisted):
            # One complete legacy check upgrades verification, without changing
            # quality/geometry measurements, their digest, or checkpoint identity.
            try:
                save_cache(path, selection, source, config, validation_key=key)
            except OSError:
                pass  # Read-only caches remain usable; retry migration next load.
        if validation is not None and key is not None:
            validation.remember(key, path, selection)
        return selection, cache_report(selection, path, config, source)
    except InterruptedError:
        raise
    except (ValueError, OSError, KeyError, TypeError, BadZipFile) as exc:
        return None, {'status': 'invalid', 'reason': f'Preprocessing cache unavailable: {exc}'}


def preprocess_source(source, config, *, calibration=None, cache_path=None,
                      should_cancel=None, on_progress=None, validation=None, on_device=None,
                      measure_geometry=True):
    """Compute fresh screening + geometry, cache if file-backed, and return it.

    Callable independently of reconstruction, even when cache use is disabled.
    Cancellation never replaces an existing cache. Array sources return a
    reusable in-memory selection; callers can also choose an explicit path.
    Without measure_geometry only quality and shape are measured; the cache
    records an inapplicable estimate, so motion runs preprocess again.
    """
    from planetrecon.geometry.discovery import discover_geometry
    from planetrecon.memory import cpu_memory_limit
    from planetrecon.runtime import apply_thread_limits
    if validation is not None:
        validation.clear()
    apply_thread_limits(config.threads)
    with cpu_memory_limit(config.max_ram_bytes):
        if source.n_frames() < 1 or min(source.frame_shape()[:2]) < 5:
            raise ValueError('preprocessing requires frames of at least 5 by 5 pixels')
        if source.color_mode() not in ('mono', 'RGB', 'BGR', 'RGGB', 'BGGR', 'GRBG', 'GBRG'):
            raise ValueError('unsupported preprocessing color mode')
        calibration = calibration_for(source, config, calibration)
        key = _validation_key(source, config, calibration)
        if source.metadata().units == 'e-' and calibration and calibration.gain_e_per_adu is not None:
            raise ValueError('gain calibration cannot be applied to observations already in electrons')
        before = identity(source, config, calibration, should_cancel)
        selection = screen_source(source, config, calibration, should_cancel=should_cancel,
                                  on_progress=on_progress, on_device=on_device)
        if selection.cancelled:
            raise InterruptedError('preprocessing cancelled')
        from planetrecon.geometry.pose import capture_timing
        selection.summary['timing'] = capture_timing(source, cadence_s=config.cadence_s)
        if measure_geometry:
            if on_device:
                on_device({'backend': 'cpu', 'reason': 'Geometry estimation', 'phase': 'geometry'})
            selection.summary['geometry_estimate'] = discover_geometry(source, config, selection, calibration, should_cancel)
            selection.summary['geometry_analysis_config'] = {key: getattr(config, key) for key in GEOMETRY_KEYS}
            selection.summary['geometry_analysis_mode'] = config.geometry_mode
        else:
            selection.summary['geometry_estimate'] = {'suggestions': {}, 'applicable': False, 'status': 'not_measured',
                'notes': ['Geometry was not measured; run Preprocess to estimate it. Quality/shape exclusions remain valid.']}
        selection.identity = identity(source, config, calibration, should_cancel)
        if selection.identity != before:
            raise ValueError('capture changed during preprocessing; no cache was saved')
        if key != _validation_key(source, config, calibration):
            raise ValueError('capture or calibration changed during preprocessing; no cache was saved')
        selection.digest = selection_digest(selection)
        path = cache_path if cache_path is not None else default_cache_path(source)
        if should_cancel and should_cancel():
            raise InterruptedError('preprocessing cancelled')
        if path is not None:
            save_cache(path, selection, source, config, validation_key=key)
            if validation is not None and key is not None and key == _validation_key(source, config, calibration):
                validation.remember(key, path, selection)
        return selection
