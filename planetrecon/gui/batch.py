"""Sequential capture queue with distinct, non-overwriting output paths."""
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

# WinJUPOS / FireCapture capture time: YYYY-MM-DD-HHMM_T, T in tenths of a minute (truncated).
WINJUPOS_TIME = re.compile(r'(\d{4}-\d{2}-\d{2})-(\d{2})(\d{2})_(\d)(?!\d)')


@dataclass
class BatchItem:
    source: Path
    output: Path
    status: str = 'queued'
    detail: str = ''
    quality: str = ''  # Stack quality summary of the completed result.


def make_queue(paths, directory, encoding):
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError('Choose an existing batch output directory')
    extension = '.png' if encoding == 'png16' else '.tif'
    reserved = set()
    items = []
    for source in dict.fromkeys(Path(p).resolve() for p in paths):
        base = source.stem + '-planetrecon'
        output = directory / (base + extension)
        count = 2
        while output in reserved or output.exists() or output.is_symlink():
            output = directory / f'{base}-{count}{extension}'
            count += 1
        reserved.add(output)
        items.append(BatchItem(source, output))
    if not items:
        raise ValueError('Choose at least one capture')
    return items


def quality_tag(summary):
    """Filename tag from a stack quality summary: frames (per alignment point when
    used), weighted mean quality and estimated SNR; '' without a summary."""
    if not summary or 'quality' not in summary:
        return ''
    points = summary.get('alignment_points')
    frames = points['frames_per_point']['median'] if points else summary['frames_stacked']
    quality = points['weighted_quality']['median'] if points else summary['quality']['weighted_mean']
    tag = f'-n{frames:.0f}-q{quality:.4g}'
    if 'estimated_snr' in summary:
        tag += f"-snr{summary['estimated_snr']:.0f}"
    return tag


def epoch_stem(stem, epoch):
    """``stem`` with its WinJUPOS time moved to a stack's output epoch.

    Capture names carry the capture midpoint. ``epoch`` is the result's
    ``output_epoch`` provenance: an epoch away from the midpoint is named by its
    UTC when the SER records it, otherwise by shifting the named time.
    """
    match = WINJUPOS_TIME.search(stem)
    if not epoch or match is None or abs(epoch['offset_from_midpoint_s']) < .05:
        return stem
    if epoch.get('utc'):
        time = datetime.fromisoformat(epoch['utc'])
    else:
        named = datetime.strptime(f'{match[1]} {match[2]}{match[3]}', '%Y-%m-%d %H%M')
        # The named tenth spans 6 s; take its centre.
        time = named + timedelta(seconds=6*int(match[4]) + 3 + epoch['offset_from_midpoint_s'])
    tenth = time.second // 6
    return stem[:match.start()] + f'{time:%Y-%m-%d-%H%M}_{tenth}' + stem[match.end():]


def result_output(output, provenance, taken=()):
    """``output`` named for its result: WinJUPOS time at the output epoch and the
    quality tag before the extension; never an existing or taken path."""
    output = Path(output)
    provenance = provenance or {}
    base = epoch_stem(output.stem, provenance.get('output_epoch')) + quality_tag(provenance.get('stack_quality'))
    if base == output.stem:
        return output
    candidate = output.with_name(base + output.suffix)
    count = 2
    while candidate in taken or candidate.exists() or candidate.is_symlink():
        candidate = output.with_name(f'{base}-{count}{output.suffix}')
        count += 1
    return candidate
