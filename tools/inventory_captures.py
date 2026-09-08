"""Private SER intake: export metadata without paths, pixels or observer fields."""
import argparse
import json
from pathlib import Path
import re
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from planetrecon.io.ser import SERSource
from planetrecon.geometry.pose import capture_timing
from tools.study_io import file_hash, write_json


def inventory(entries, *, metadata_only=False):
    records = []; identifiers = set()
    for entry in entries:
        identifier = entry['id']
        if not re.fullmatch(r'[a-z0-9][a-z0-9_-]*', identifier) or identifier in identifiers:
            raise ValueError('unique anonymous capture identifiers required')
        identifiers.add(identifier)
        permissions = entry.get('permissions', {})
        permissions = {key: permissions.get(key, 'unknown')
                       for key in ('report_pixels', 'redistribute_fixture')}
        if any(v not in ('unknown', 'allowed', 'denied') for v in permissions.values()):
            raise ValueError('permissions must be unknown, allowed or denied')
        evidence = entry.get('permission_evidence')
        if 'allowed' in permissions.values() and not evidence:
            raise ValueError('explicit permission evidence required for an allowed use')
        override = entry.get('bayer_override')
        if override is not None and not entry.get('interpretation_evidence'):
            raise ValueError('colour override requires interpretation evidence')
        path = Path(entry['path']); before = path.stat()
        source = SERSource(path, bayer_override=override)
        try:
            header = source.header
            timestamps = source.timestamps()
            timestamp_diagnostics = None if timestamps is None else {
                'duplicate_intervals': int(np.count_nonzero(timestamps[1:] == timestamps[:-1])),
                'reversed_intervals': int(np.count_nonzero(timestamps[1:] < timestamps[:-1])),
                'nonpositive_ticks': int(np.count_nonzero(timestamps <= 0))}
            record = {'id': identifier, 'target': entry.get('target'),
                      'night_group': entry.get('night_group'), 'camera_group': entry.get('camera_group'),
                      'frames': source.n_frames(), 'width': header.width, 'height': header.height,
                      'bit_depth': header.pixel_depth, 'header_colour': header.color_mode,
                      'effective_colour': source.color_mode(),
                      'interpretation_evidence': entry.get('interpretation_evidence', 'SER header only'),
                      'timing': capture_timing(source), 'timestamp_diagnostics': timestamp_diagnostics,
                      'file_bytes': before.st_size,
                      'capture_sha256': None if metadata_only else file_hash(path),
                      'permissions': permissions, 'permission_evidence': evidence}
        finally:
            source.close()
        after = path.stat()
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            raise ValueError('capture changed during intake')
        records.append(record)
    return {'status': 'metadata_inventory' if metadata_only else 'hashed_inventory',
            'records': records, 'scientific_qualification': False,
            'scope': 'Intake metadata only; no reconstruction, independence, geometry or scientific qualification. Unknown permissions remain unknown. Metadata-only records do not identify image contents.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--private-spec', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--metadata-only', action='store_true')
    a = p.parse_args()
    if a.out.exists(): raise ValueError('use a new inventory output')
    report = inventory(json.loads(a.private_spec.read_text()), metadata_only=a.metadata_only)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(a.out, report)
