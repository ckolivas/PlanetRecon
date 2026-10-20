"""Sequential capture queue with distinct, non-overwriting output paths."""
from dataclasses import dataclass
from pathlib import Path


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


def with_quality(output, summary, taken=()):
    """``output`` with the quality tag before its extension, never an existing or taken path."""
    output = Path(output)
    tag = quality_tag(summary)
    if not tag:
        return output
    base = output.stem + tag
    candidate = output.with_name(base + output.suffix)
    count = 2
    while candidate in taken or candidate.exists() or candidate.is_symlink():
        candidate = output.with_name(f'{base}-{count}{output.suffix}')
        count += 1
    return candidate
