"""Sequential capture queue with distinct, non-overwriting output paths."""
from dataclasses import dataclass
from pathlib import Path


@dataclass
class BatchItem:
    source: Path
    output: Path
    status: str = 'queued'
    detail: str = ''


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
