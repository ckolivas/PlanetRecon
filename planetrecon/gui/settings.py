"""Versioned, atomic desktop preferences, separate from reconstruction checkpoints."""
import json
import os
from pathlib import Path
import tempfile

from PySide6.QtCore import QStandardPaths


def default_path():
    override = os.environ.get('PLANETRECON_SETTINGS_PATH')
    if override:
        return Path(override)
    return Path(QStandardPaths.writableLocation(
        QStandardPaths.StandardLocation.GenericConfigLocation)) / 'PlanetRecon' / 'settings.json'


def load(path):
    try:
        data = json.loads(Path(path).read_text())
        if not isinstance(data, dict) or data.get('version') != 1:
            raise ValueError('unsupported settings version')
        return data, None
    except FileNotFoundError:
        return {}, None
    except (OSError, ValueError) as exc:
        return {}, f'Could not restore settings: {exc}'


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False,
                                         prefix=path.name + '.', suffix='.tmp') as stream:
            name = stream.name
            json.dump({'version': 1, **data}, stream, indent=2, allow_nan=False)
            stream.write('\n')
        os.replace(name, path)
    finally:
        if name is not None:
            Path(name).unlink(missing_ok=True)
