"""Cancellation-aware compressed NPZ writes without changing array encoding."""
import numpy as np


class _CheckedWriter:
    # NumPy feeds ZIP compression bounded buffers (normally 16 MiB). Check even
    # empty compressed writes, and split output to avoid one large disk write.
    def __init__(self, stream, should_cancel):
        self.stream = stream
        self.should_cancel = should_cancel

    def __getattr__(self, name):
        return getattr(self.stream, name)

    def write(self, data):
        if self.should_cancel():
            raise InterruptedError('checkpoint cancelled during compression/write')
        view = memoryview(data).cast('B')
        total = 0
        while total < len(view):
            if self.should_cancel():
                raise InterruptedError('checkpoint cancelled during compression/write')
            written = self.stream.write(view[total:total+1024*1024])
            if written is None or written <= 0:
                raise OSError('checkpoint write made no progress')
            total += written
        return total


def save_compressed(stream, *, should_cancel=None, **arrays):
    """Keep NumPy's NPZ format and check cancellation throughout ZIP output.

    The caller owns atomic replacement and cleanup of the unpublished file.
    """
    target = stream if should_cancel is None else _CheckedWriter(stream, should_cancel)
    np.savez_compressed(target, **arrays)
