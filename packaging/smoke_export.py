"""Exercise all six scientific encodings through a frozen executable.

Run from the source checkout: python3 packaging/smoke_export.py /path/to/planetrecon
The host's NumPy, tifffile and Qt independently inspect the bundled writer's output.
"""
from pathlib import Path
import json
import subprocess
import sys
import tempfile

import numpy as np
import tifffile
from PySide6.QtGui import QImage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from planetrecon.io.ser import write_ser, COLOR_MONO, COLOR_RGB
from planetrecon.result import load_snapshot


def main():
    executable = str(Path(sys.argv[1]).resolve())
    with tempfile.TemporaryDirectory(prefix='planetrecon-export-smoke-') as directory:
        root = Path(directory)
        for color in ('mono', 'RGB'):
            frame = np.arange(80, dtype=np.uint16).reshape(8, 10) * 500 + 300
            if color == 'RGB':
                frame = np.stack([frame, frame[:, ::-1], frame // 2], axis=-1)
            capture, out = root / f'{color}.ser', root / color
            write_ser(capture, np.stack([frame] * 2), color_id=COLOR_MONO if color == 'mono' else COLOR_RGB)
            subprocess.run([executable, '--threads', '2', 'stack', '--path', str(capture),
                            '--out', str(out), '--device', 'cpu'], check=True, timeout=30)
            result = load_snapshot(out / 'stack.npz')
            for encoding in ('png16', 'tiff16', 'tiff32'):
                dest = out / (encoding + ('.png' if encoding == 'png16' else '.tif'))
                args = [executable, 'export', '--path', str(out / 'stack.npz'),
                        '--out', str(dest), '--encoding', encoding]
                if encoding != 'tiff32':
                    args += ['--black', '0', '--white', '65535']
                subprocess.run(args, check=True, timeout=30)
                expected = (result.image.astype(np.float32) if encoding == 'tiff32'
                            else np.floor(result.image + .5).astype(np.uint16))
                if encoding == 'png16':
                    q = QImage(str(dest)).convertToFormat(QImage.Format.Format_RGBA64)
                    assert not q.isNull()
                    rgba = np.frombuffer(q.constBits(), dtype=np.uint16).reshape(8, 10, 4)
                    actual = rgba[..., 0] if color == 'mono' else rgba[..., :3]
                else:
                    with tifffile.TiffFile(dest) as tf:
                        actual = tf.pages[0].asarray()
                        assert actual.dtype == expected.dtype
                        assert not json.loads(tf.pages[0].description)['result']['incomplete']
                np.testing.assert_array_equal(actual, expected)
    print('PASS: frozen mono/RGB PNG16, TIFF16, TIFF32 encoders and CPU SER stack')


if __name__ == '__main__':
    main()
