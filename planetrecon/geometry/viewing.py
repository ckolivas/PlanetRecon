"""Geocentric viewing latitude from SER UTC and JPL Horizons quantity 14.

No terrestrial site coordinates are needed. Cache only validated responses,
keyed by target and epoch; keep blank GUI inputs automatic for later captures.
"""
from dataclasses import replace
from datetime import datetime, timedelta
import csv
import json
import math
import os
from pathlib import Path
import re
import tempfile
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np

API_URL = 'https://ssd.jpl.nasa.gov/api/horizons.api'
SOURCE_URL = 'https://ssd.jpl.nasa.gov/horizons/manual.html#observer-table'
BODY_IDS = dict(mercury=199, venus=299, mars=499, jupiter=599,
                saturn=699, uranus=799, neptune=899)
SCHEMA = 'geocentric-viewing-1'
MAX_RESPONSE = 128 * 1024


def capture_epoch_jd(source, cadence_s=None):
    """Absolute midpoint, preferring SER trailer UTC to the header start time.

    Relative/synthetic ticks and missing dates cannot authorize an ephemeris.
    Restrict automatic lookup to the modern imaging era (1900--2100).
    """
    meta = source.metadata()
    field = meta.extras.get('datetime_utc_ticks')
    if field is None or source.timestamp_scale_s() != 1e-7:
        raise ValueError('SER absolute UTC capture date is unavailable')
    lo = (datetime(1900, 1, 1) - datetime(1, 1, 1)).days * 864000000000
    hi = (datetime(2101, 1, 1) - datetime(1, 1, 1)).days * 864000000000
    times = source.timestamps()
    if times is not None and len(times):
        if np.any(times[1:] < times[:-1]):
            raise ValueError('SER timestamps must be nondecreasing')
        first, last = int(times[0]), int(times[-1])
        if lo <= first <= last < hi:
            ticks = (first + last) // 2
            return 1721425.5 + ticks / 864000000000
    start = field.value
    if not isinstance(start, (int, np.integer)) or not lo <= start < hi:
        raise ValueError('SER absolute UTC capture date is unavailable (supported years 1900–2100)')
    from planetrecon.geometry.pose import source_times_s
    relative, origin = source_times_s(source, cadence_s=cadence_s)
    duration = float(relative[-1]) if len(relative) and origin != 'inferred' else 0.
    jd = 1721425.5 + int(start) / 864000000000 + duration / 172800
    if not math.isfinite(jd) or jd >= 1721425.5 + hi / 864000000000:
        raise ValueError('SER capture midpoint is outside supported years 1900–2100')
    return jd


def parse_response(payload, planet, jd):
    """Check target, geocentre, epoch and ellipsoid before converting latitude."""
    if not isinstance(payload, dict) or payload.get('error'):
        raise ValueError('JPL Horizons did not return an ephemeris')
    result = payload.get('result', '')
    if not isinstance(result, str) or len(result) > MAX_RESPONSE:
        raise ValueError('Invalid JPL Horizons response')
    body = re.search(r'Target body name:\s*\w+\s*\((\d+)\)', result)
    if (not body or int(body[1]) != BODY_IDS[planet]
            or not re.search(r'Center body name:\s*Earth\s*\(399\)', result)
            or not re.search(r'Center-site name:\s*GEOCENTRIC\s*\n', result)):
        raise ValueError('JPL Horizons target or observer does not match the request')
    radii = re.search(r'Target radii\s*:\s*([\d.eE+-]+),\s*([\d.eE+-]+),\s*([\d.eE+-]+)\s*km', result)
    if not radii:
        raise ValueError('JPL Horizons reference ellipsoid is missing')
    a, b, c = map(float, radii.groups())
    if not all(math.isfinite(v) and v > 0 for v in (a, b, c)) or not math.isclose(a, b) or c > a:
        raise ValueError('JPL Horizons reference ellipsoid is unsupported')
    try:
        before, table = result.split('$$SOE')
        table, _ = table.split('$$EOE')
        header = next(line for line in before.splitlines() if 'ObsSub-LAT' in line and ',' in line)
        names = [v.strip() for v in next(csv.reader([header]))]
        if 'Date__(UT)' not in names[0]:
            raise ValueError('Ephemeris time scale is not UT')
        rows = list(csv.reader(line for line in table.splitlines() if line.strip()))
        if len(rows) != 1 or len(rows[0]) != len(names):
            raise ValueError('Expected one ephemeris epoch')
        row = rows[0]
        # Explicit Gregorian FRACSEC output, independent of the machine locale.
        date = row[0].strip()
        match = re.fullmatch(r'(\d{4})-([A-Z][a-z]{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2}\.\d+)', date)
        if not match:
            raise ValueError('Invalid ephemeris date')
        year, month, day, hour, minute, second = match.groups()
        month = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'].index(month) + 1
        epoch = datetime(int(year), month, int(day), int(hour), int(minute)) + timedelta(seconds=float(second))
        returned_jd = 1721425.5 + (epoch - datetime(1, 1, 1)).total_seconds() / 86400
        if abs(returned_jd - jd) * 86400 > .01:
            raise ValueError('Ephemeris epoch differs from the capture midpoint')
        latitude = float(row[names.index('ObsSub-LAT')])
        if not math.isfinite(latitude) or abs(latitude) > 90:
            raise ValueError('Invalid ephemeris latitude')
    except (ValueError, StopIteration, IndexError) as exc:
        raise ValueError('Invalid JPL Horizons latitude table: ' + str(exc)) from exc
    # Horizons gives the surface normal angle. Our globe needs the viewing
    # direction angle through the centre of the reference ellipsoid.
    lat = math.radians(latitude)
    centric = math.atan2((c/a)**2 * math.sin(lat), math.cos(lat))
    return {'schema': SCHEMA, 'origin': 'jpl_horizons_geocentric', 'planet': planet,
            'epoch_jd_utc': jd, 'source_url': SOURCE_URL,
            'planetodetic_latitude_deg': latitude, 'reference_radii_km': [a, b, c],
            'sub_obs_lat_rad': centric}


def fetch_response(planet, jd):
    values = dict(COMMAND=str(BODY_IDS[planet]), CENTER='500@399', EPHEM_TYPE='OBSERVER',
                  TLIST=format(jd, '.9f'), TLIST_TYPE='JD', TIME_TYPE='UT',
                  TIME_DIGITS='FRACSEC', CAL_TYPE='GREGORIAN', QUANTITIES='14',
                  CSV_FORMAT='YES', EXTRA_PREC='YES', OBJ_DATA='YES')
    query = urlencode({'format': 'json', **{k: "'" + v + "'" for k, v in values.items()}})
    with urlopen(API_URL + '?' + query, timeout=15) as response:
        data = response.read(MAX_RESPONSE + 1)
    if len(data) > MAX_RESPONSE:
        raise ValueError('JPL Horizons response exceeds the size limit')
    return json.loads(data)


def viewing_geometry(source, planet, cadence_s=None):
    jd = capture_epoch_jd(source, cadence_s)
    path = Path(source.metadata().path)
    cache = path.with_name(path.name + '.planetrecon-viewing.json') if path.is_file() else None
    key = {'schema': SCHEMA, 'planet': planet, 'epoch_jd_utc': jd}
    if cache is not None:
        try:
            if cache.stat().st_size <= MAX_RESPONSE:
                saved = json.loads(cache.read_text())
                if saved['key'] == key:
                    return parse_response(saved['response'], planet, jd)
        except (OSError, ValueError, KeyError, TypeError):
            pass
    payload = fetch_response(planet, jd)
    record = parse_response(payload, planet, jd)
    if cache is not None:
        tmp = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', dir=cache.parent, prefix=cache.name, delete=False) as handle:
                tmp = Path(handle.name)
                json.dump({'key': key, 'response': payload}, handle)
            os.replace(tmp, cache)
        except OSError:
            pass  # Read-only captures can still use the result in this run.
        finally:
            if tmp is not None:
                tmp.unlink(missing_ok=True)
    return record


def resolve_viewing(source, config, *, strict=True):
    """Return an effective config and provenance; manual inputs always win."""
    if config.sub_obs_lat_rad is not None or config.geometry_mode not in ('surface', 'combined', 'saturn'):
        return config, None
    planet = 'saturn' if config.geometry_mode == 'saturn' else config.rotation_planet
    if planet not in BODY_IDS:
        return config, None
    try:
        # Preserve existing unknown-date surface workflows; Saturn cannot
        # assume edge-on rings. A dated, identified target must use its view.
        try:
            capture_epoch_jd(source, config.cadence_s)
        except ValueError:
            if config.geometry_mode != 'saturn':
                return config, None
            raise
        record = viewing_geometry(source, planet, config.cadence_s)
    except (OSError, ValueError, TypeError) as exc:
        message = ('Automatic planetary viewing latitude unavailable: ' + str(exc)
                   + '. Supply the planet-facing latitude override, or retry with valid SER UTC and an internet connection (cached results work offline).')
        if strict:
            raise ValueError(message) from exc
        return config, {'status': 'unavailable', 'notes': [message]}
    return replace(config, sub_obs_lat_rad=record['sub_obs_lat_rad']), record
