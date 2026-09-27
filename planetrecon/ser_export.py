"""Lossless, ordered SER filtering using validated preprocessing selections."""
from dataclasses import dataclass
import os
from pathlib import Path
import struct
import tempfile

import numpy as np

from planetrecon.export import ExportCancelled
from planetrecon.io import open_source
from planetrecon.io.ser import SERSource, SER_HEADER_SIZE, NAME_TO_COLOR
from planetrecon.pipeline.preprocess import best_frame_mask
from planetrecon.pipeline.preprocess_cache import load_cache


@dataclass(frozen=True)
class SERExportReport:
    path: Path
    n_input: int
    n_written: int
    has_timestamps: bool


def export_filtered_ser(source_path, destination, config, *, cache_path=None,
                        expected_digest=None, validation=None, overwrite=False,
                        should_cancel=None, on_progress=None):
    """Copy selected full detector frames verbatim, never quality-sort them.

    Calibration settings are used to validate the measurements only.
    Output pixels remain uncalibrated, uncropped and unnormalised. Explicit
    Bayer/endian interpretation is recorded in the output's ecosystem header.
    """
    source_path, destination = Path(source_path), Path(destination)
    if source_path.suffix.lower() != '.ser' or destination.suffix.lower() != '.ser':
        raise ValueError('Filtered SER export requires a SER input and .ser output')
    if not config.frame_preselection:
        raise ValueError('Enable cached preprocessing to export the selected quality/size range')
    if (destination.resolve() == source_path.resolve()
            or destination.exists() and destination.samefile(source_path)):
        raise ValueError('Filtered export cannot replace the input capture')
    if (destination.exists() or destination.is_symlink()) and not overwrite:
        raise FileExistsError(destination)

    def cancelled():
        if should_cancel is not None and should_cancel():
            raise ExportCancelled('SER export cancelled')

    def progress(stage, done, total):
        cancelled()
        if on_progress is not None:
            on_progress(stage, done, total)

    def stamp():
        stat = source_path.stat()
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns

    temporary = None
    cancelled()
    before = stamp()
    try:
        options = dict(bayer_override=config.bayer_override, endian_override=config.endian_override,
                       endian_convention=config.endian_convention,
                       recover_complete_frames=config.recover_complete_frames)
        with open_source(source_path, crop=config.crop, **options) as measured, SERSource(source_path, **options) as raw:
            progress('Validating capture', 0, raw.n_frames())
            selection, report = load_cache(measured, config, path=cache_path, validation=validation,
                should_cancel=should_cancel, on_progress=lambda done, total: progress('Validating capture', done, total))
            if selection is None:
                raise ValueError(report.get('reason', 'Preprocess this capture before exporting'))
            if expected_digest is not None and selection.digest != expected_digest:
                raise ValueError('Preprocessing changed; reload the quality graph before exporting')
            indices = np.flatnonzero(best_frame_mask(selection, config.stack_percent, config.frame_selection_mode))
            if not indices.size:
                raise ValueError('No frames remain inside the selected quality and size limits')
            timestamps = raw.timestamps()
            # Preserve opaque header text bytes rather than decoding/re-encoding.
            with source_path.open('rb') as stream:
                header = bytearray(stream.read(SER_HEADER_SIZE))
            struct.pack_into('<i', header, 18, NAME_TO_COLOR[raw.color_mode()])
            struct.pack_into('<i', header, 22, 0 if raw._little() else 1)
            struct.pack_into('<i', header, 38, len(indices))
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode='wb', dir=destination.parent,
                    prefix='.'+destination.name+'.', suffix='.tmp', delete=False) as output:
                temporary = Path(output.name)
                output.write(header)
                progress('Exporting SER', 0, len(indices))
                for count, index in enumerate(indices, 1):
                    cancelled()
                    output.write(raw.read_frame_bytes(int(index)))
                    if count == len(indices) or count % max(1, len(indices)//100) == 0:
                        progress('Exporting SER', count, len(indices))
                if timestamps is not None:
                    output.write(np.asarray(timestamps[indices], dtype='<i8').tobytes())
                output.flush()
                os.fsync(output.fileno())
            cancelled()
            raw._check_input()
            if stamp() != before:
                raise OSError('Input capture changed during export; output was not published')
            if overwrite:
                os.replace(temporary, destination)
            else:
                # Atomic no-clobber publication also handles another writer
                # creating the destination after the initial existence check.
                os.link(temporary, destination)
                temporary.unlink()
            temporary = None
            return SERExportReport(destination, raw.n_frames(), len(indices), timestamps is not None)
    except InterruptedError as exc:
        raise ExportCancelled('SER export cancelled') from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
