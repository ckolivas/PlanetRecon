"""PNG reading via Qt (no PIL in the venv) and small comparison helpers."""
import numpy as np


def read_png(path):
    from PySide6.QtGui import QImage
    image = QImage(str(path))
    if image.isNull():
        raise ValueError(f'cannot read {path}')
    depth = image.depth()
    rgba = image.convertToFormat(QImage.Format.Format_RGBA64)
    a = np.frombuffer(rgba.constBits(), dtype=np.uint16).reshape(
        rgba.height(), rgba.bytesPerLine() // 2)[:, :rgba.width() * 4].reshape(rgba.height(), rgba.width(), 4)
    rgb = a[..., :3].astype(float)
    grey = image.isGrayscale() or np.array_equal(rgb[..., 0], rgb[..., 1]) and np.array_equal(rgb[..., 1], rgb[..., 2])
    return (rgb[..., 0] if grey else rgb), depth, image.format()
