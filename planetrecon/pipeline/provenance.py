"""Bounded input and calibration identities for full-resolution result snapshots."""
from dataclasses import fields
import hashlib
from pathlib import Path

import numpy as np


def capture_provenance(source, config, calibration):
    path = Path(source.metadata().path)
    identity = {"method": "unavailable", "reason": "source has no accessible backing file"}
    if path.is_file():
        # A bounded identity, explicitly not a full-file checksum for huge SERs.
        with path.open("rb") as stream:
            stat = path.stat()
            first = stream.read(65536)
            stream.seek(max(0, stat.st_size - 65536))
            last = stream.read(65536)
        identity = {"method": "size-mtime-first-last-64KiB-sha256",
                    "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                    "sample_sha256": hashlib.sha256(first + last).hexdigest(),
                    "full_file_checksum": False}
    cal = {}
    if calibration is not None:
        for field in fields(calibration):
            value = getattr(calibration, field.name)
            if isinstance(value, np.ndarray):
                arr = np.ascontiguousarray(value)
                cal[field.name] = {"shape": list(arr.shape), "dtype": arr.dtype.str,
                                   "sha256": hashlib.sha256(arr.tobytes()).hexdigest()}
            else:
                cal[field.name] = value
    return {"config": config.to_dict(), "input_identity": identity,
            "calibration": cal, "calibration_mode": calibration.mode if calibration else "approximate-noise"}
