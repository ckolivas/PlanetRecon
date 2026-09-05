"""Bounded native AVI 1.0 decoder: uncompressed DIB RGB24 and gray palette8.

No colour conversion, gamma correction, external process or codec installation.
Compressed, YUV, audio/multiple-stream and OpenDML inputs fail explicitly.
"""
from __future__ import annotations

import os
from pathlib import Path
import struct
import tempfile

import numpy as np

from planetrecon.io.source import FrameSource, FieldValue, ObservationMetadata


class AVISource(FrameSource):
    def __init__(self, path, *, bayer_override=None, endian_override=None, **_ignored):
        if bayer_override or endian_override:
            raise ValueError('AVI does not support Bayer/endian overrides; use raw SER')
        self.path = Path(path)
        self._file = self.path.open('rb')
        self._index = tempfile.TemporaryFile()
        self._identity = self._stat()
        try:
            self._parse()
        except BaseException:
            self.close()
            raise

    def _stat(self):
        s = self.path.stat()
        return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns

    def _read(self, offset, size):
        self._file.seek(offset)
        data = self._file.read(size)
        if len(data) != size:
            raise ValueError('truncated AVI chunk or frame')
        return data

    def _chunks(self, start, end):
        while start < end:
            if end - start < 8:
                raise ValueError('truncated AVI chunk header')
            tag, size = struct.unpack('<4sI', self._read(start, 8))
            stop = start + 8 + size
            padded = stop + (size & 1)
            if padded > end:
                raise ValueError('AVI chunk extends beyond its parent')
            yield tag, start + 8, size
            start = padded

    def _parse(self):
        riff, size, kind = struct.unpack('<4sI4s', self._read(0, 12))
        if riff != b'RIFF' or kind != b'AVI ' or size + 8 != self._identity[2]:
            raise ValueError('requires a complete single-RIFF AVI 1.0 file; OpenDML unsupported')
        headers = []
        movies = []
        for tag, pos, length in self._chunks(12, size + 8):
            if tag == b'LIST':
                kind = self._read(pos, 4)
                if kind == b'hdrl':
                    headers.append((pos + 4, pos + length))
                elif kind == b'movi':
                    movies.append((pos + 4, pos + length))
        if len(headers) != 1 or len(movies) != 1:
            raise ValueError('AVI requires one header list and one movie list')
        streams = []
        for tag, pos, length in self._chunks(*headers[0]):
            if tag == b'LIST' and self._read(pos, 4) == b'strl':
                streams.append((pos + 4, pos + length))
        if len(streams) != 1:
            raise ValueError('AVI supports exactly one video stream, without audio')
        fields = {}
        for tag, pos, length in self._chunks(*streams[0]):
            if tag in (b'strh', b'strf'):
                if tag in fields or length > 4096:
                    raise ValueError('invalid AVI stream header')
                fields[tag] = self._read(pos, length)
        sh, fmt = fields.get(b'strh', b''), fields.get(b'strf', b'')
        if len(sh) < 56 or sh[:4] != b'vids' or len(fmt) < 40:
            raise ValueError('missing AVI video stream or bitmap header')
        self._scale, self._rate, start, declared = struct.unpack_from('<4I', sh, 20)
        if not self._scale or not self._rate or start:
            raise ValueError('invalid or nonzero-start AVI timing')
        header_size, self._w, signed_h, planes, bits, compression = struct.unpack_from('<IiiHHI', fmt)
        if header_size != 40 or self._w <= 0 or signed_h == 0 or planes != 1 or compression != 0 or bits not in (8, 24):
            raise ValueError('supported AVI pixels: BI_RGB RGB24 or identity grayscale palette8 only')
        self._h = abs(signed_h)
        self._bottom_up = signed_h > 0
        self._channels = 3 if bits == 24 else 1
        self._stride = ((self._w * self._channels + 3) // 4) * 4
        if bits == 8:
            palette = fmt[40:1064]
            expected = b''.join(bytes((i, i, i, 0)) for i in range(256))
            if palette != expected:
                raise ValueError('AVI palette must be complete identity grayscale')
        self._n = 0
        def scan(lo, hi, depth=0):
            if depth > 8:
                raise ValueError('AVI nesting is too deep')
            for tag, pos, length in self._chunks(lo, hi):
                if tag == b'LIST':
                    if self._read(pos, 4) != b'rec ':
                        raise ValueError('unsupported AVI movie list')
                    scan(pos + 4, pos + length, depth + 1)
                elif tag in (b'00db', b'00dc'):
                    if length != self._stride * self._h:
                        raise ValueError('AVI frame size disagrees with uncompressed pixel format')
                    self._index.write(struct.pack('<Q', pos))
                    self._n += 1
                elif tag != b'JUNK':
                    raise ValueError(f'unsupported AVI movie chunk {tag!r}')
        scan(*movies[0])
        if self._n == 0 or self._n != declared:
            raise ValueError('AVI frame count disagrees with header (dropped/empty frames unsupported)')
        self._index.flush()

    def metadata(self):
        return ObservationMetadata(str(self.path), self._n, self._w, self._h,
            self.color_mode(), 8, 'little', units='decoded-code-value', extras={
                'decoder': FieldValue('native DIB 1.0', 'inferred'),
                'cadence_s': FieldValue(self._scale / self._rate, 'header', 'nominal; no measured timestamps'),
                'warnings': FieldValue(['AVI transfer curve and sensor linearity are unknown; values are not calibrated ADU.',
                    'Header cadence is nominal; supply measured timing externally for physical rotation.'], 'inferred'),
            })

    def n_frames(self):
        return self._n

    def color_mode(self):
        return 'RGB' if self._channels == 3 else 'mono'

    def frame_shape(self):
        return (self._h, self._w, 3) if self._channels == 3 else (self._h, self._w)

    def read_raw(self, index):
        if not 0 <= index < self._n:
            raise IndexError(index)
        if self._stat() != self._identity:
            raise OSError('AVI input changed while open')
        self._index.seek(index * 8)
        offset, = struct.unpack('<Q', self._index.read(8))
        rows = np.frombuffer(self._read(offset, self._h * self._stride), dtype=np.uint8).reshape(self._h, self._stride)
        rows = rows[:, :self._w * self._channels]
        if self._bottom_up:
            rows = rows[::-1]
        if self._channels == 3:
            rows = rows.reshape(self._h, self._w, 3)[..., ::-1]
        return rows.copy()

    def close(self):
        self._file.close()
        self._index.close()
