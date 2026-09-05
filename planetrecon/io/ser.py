"""SER v3 reader/writer. Ecosystem endianness: header 0 means little-endian."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from planetrecon import constants as C
from planetrecon.io.source import FieldValue, FrameSource, ObservationMetadata


SER_HEADER_SIZE = 178
SER_FILE_ID = b"LUCAM-RECORDER"
_HEADER_STRUCT = struct.Struct("<14siiiiiii40s40s40sqq")

COLOR_MONO = 0
COLOR_RGGB = 8
COLOR_GRBG = 9
COLOR_GBRG = 10
COLOR_BGGR = 11
COLOR_CYYM = 16
COLOR_YCMY = 17
COLOR_YMCY = 18
COLOR_MYYC = 19
COLOR_RGB = 100
COLOR_BGR = 101

COLOR_NAMES = {
    COLOR_MONO: "mono",
    COLOR_RGGB: "RGGB",
    COLOR_GRBG: "GRBG",
    COLOR_GBRG: "GBRG",
    COLOR_BGGR: "BGGR",
    COLOR_CYYM: "CYYM",
    COLOR_YCMY: "YCMY",
    COLOR_YMCY: "YMCY",
    COLOR_MYYC: "MYYC",
    COLOR_RGB: "RGB",
    COLOR_BGR: "BGR",
}
BAYER_IDS = {COLOR_RGGB, COLOR_GRBG, COLOR_GBRG, COLOR_BGGR}
NAME_TO_COLOR = {name: cid for cid, name in COLOR_NAMES.items()}


def _decode_ascii(raw: bytes) -> str:
    return raw.split(b"\x00", 1)[0].decode("utf-8", errors="replace").strip()


def _pad_ascii(text: str, n: int) -> bytes:
    blob = text.encode("utf-8", errors="replace")[:n]
    return blob + b"\x00" * (n - len(blob))


@dataclass(frozen=True)
class SERHeader:
    file_id: bytes
    lu_id: int
    color_id: int
    little_endian_flag: int
    width: int
    height: int
    pixel_depth: int
    frame_count: int
    observer: str
    instrument: str
    telescope: str
    datetime: int
    datetime_utc: int
    endian_convention: str = "ecosystem"

    @property
    def color_mode(self) -> str:
        return COLOR_NAMES.get(int(self.color_id), f"unknown-{self.color_id}")

    @property
    def planes(self) -> int:
        if int(self.color_id) in (COLOR_RGB, COLOR_BGR):
            return 3
        return 1

    @property
    def bytes_per_sample(self) -> int:
        return 1 if int(self.pixel_depth) <= 8 else 2

    @property
    def frame_bytes(self) -> int:
        return int(self.width) * int(self.height) * self.planes * self.bytes_per_sample

    @property
    def samples_are_little_endian(self) -> bool:
        # Spec boolean is LittleEndian; first writers treated 0 as little-endian.
        if self.endian_convention == "spec":
            return bool(self.little_endian_flag)
        return int(self.little_endian_flag) == 0

    def numpy_dtype(self) -> np.dtype:
        if self.bytes_per_sample == 1:
            return np.dtype(np.uint8)
        return np.dtype("<u2" if self.samples_are_little_endian else ">u2")


def parse_header(blob: bytes, endian_convention: str = "ecosystem") -> SERHeader:
    if len(blob) < SER_HEADER_SIZE:
        raise ValueError(f"SER header is {len(blob)} bytes, expected {SER_HEADER_SIZE}")
    fields = _HEADER_STRUCT.unpack(blob[:SER_HEADER_SIZE])
    file_id = fields[0]
    if file_id != SER_FILE_ID:
        raise ValueError(f"not a SER file: FileID={file_id!r}")
    return SERHeader(
        file_id=file_id,
        lu_id=int(fields[1]),
        color_id=int(fields[2]),
        little_endian_flag=int(fields[3]),
        width=int(fields[4]),
        height=int(fields[5]),
        pixel_depth=int(fields[6]),
        frame_count=int(fields[7]),
        observer=_decode_ascii(fields[8]),
        instrument=_decode_ascii(fields[9]),
        telescope=_decode_ascii(fields[10]),
        datetime=int(fields[11]),
        datetime_utc=int(fields[12]),
        endian_convention=endian_convention,
    )


def pack_header(header: SERHeader) -> bytes:
    return _HEADER_STRUCT.pack(
        header.file_id,
        header.lu_id,
        header.color_id,
        header.little_endian_flag,
        header.width,
        header.height,
        header.pixel_depth,
        header.frame_count,
        _pad_ascii(header.observer, 40),
        _pad_ascii(header.instrument, 40),
        _pad_ascii(header.telescope, 40),
        header.datetime,
        header.datetime_utc,
    )


def write_ser(
    path: str | Path,
    frames: np.ndarray,
    *,
    color_id: int = COLOR_MONO,
    pixel_depth: int | None = None,
    little_endian_flag: int = 0,
    timestamps: np.ndarray | None = None,
    observer: str = "",
    instrument: str = "",
    telescope: str = "",
    datetime: int = 0,
    datetime_utc: int = 0,
    endian_convention: str = "ecosystem",
) -> Path:
    """Write a tiny SER used as a deterministic golden file."""
    path = Path(path)
    arr = np.asarray(frames)
    if arr.ndim == 3:
        n, height, width = arr.shape
        planes = 1
    elif arr.ndim == 4 and arr.shape[-1] == 3:
        n, height, width, planes = arr.shape
    else:
        raise ValueError("frames must be (N,H,W) or (N,H,W,3)")
    if int(color_id) not in COLOR_NAMES:
        raise ValueError(f"SER ColorID {color_id} is not a recognised mono/Bayer/RGB id")
    if pixel_depth is None:
        pixel_depth = 8 if arr.dtype == np.uint8 else 16
    header = SERHeader(
        file_id=SER_FILE_ID,
        lu_id=0,
        color_id=int(color_id),
        little_endian_flag=int(little_endian_flag),
        width=int(width),
        height=int(height),
        pixel_depth=int(pixel_depth),
        frame_count=int(n),
        observer=observer,
        instrument=instrument,
        telescope=telescope,
        datetime=int(datetime),
        datetime_utc=int(datetime_utc),
        endian_convention=endian_convention,
    )
    if header.planes != planes:
        raise ValueError(f"color_id {color_id} expects {header.planes} planes, got {planes}")
    dtype = header.numpy_dtype()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = np.asarray(arr, dtype=dtype)
    with path.open("wb") as fh:
        fh.write(pack_header(header))
        fh.write(payload.tobytes(order="C"))
        if timestamps is not None:
            ts = np.asarray(timestamps, dtype="<i8")
            if ts.size != n:
                raise ValueError("timestamp count must match frame count")
            fh.write(ts.tobytes(order="C"))
    return path


class SERSource(FrameSource):
    def __init__(
        self,
        path: str | Path,
        *,
        endian_convention: str = "ecosystem",
        endian_override: str | None = None,
        bayer_override: str | None = None,
        recover_complete_frames: bool = False,
        **_ignored,
    ):
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self.endian_convention = endian_convention
        self.endian_override = endian_override
        self.bayer_override = bayer_override
        self.recover_complete_frames = bool(recover_complete_frames)
        self._fh = self.path.open("rb")
        header_blob = self._fh.read(SER_HEADER_SIZE)
        self.header = parse_header(header_blob, endian_convention=endian_convention)
        if self.header.color_id not in COLOR_NAMES:
            raise ValueError(
                f"SER ColorID {self.header.color_id} is not a recognised mono/Bayer/RGB id"
            )
        self._file_size = self.path.stat().st_size
        self._n_declared = self.header.frame_count
        complete, remainder = divmod(
            max(0, self._file_size - SER_HEADER_SIZE), max(self.header.frame_bytes, 1)
        )
        trailer_bytes = complete * 8
        # timestamps occupy 8 bytes/frame after image data if present
        image_bytes = self._n_declared * self.header.frame_bytes
        available = self._file_size - SER_HEADER_SIZE
        if available < image_bytes:
            n_complete = available // self.header.frame_bytes
            if not self.recover_complete_frames:
                raise ValueError(
                    f"{self.path} is truncated: declared {self._n_declared} frames, "
                    f"only {n_complete} complete frames present"
                )
            self._n = int(n_complete)
            self._has_trailer = False
        else:
            self._n = int(self._n_declared)
            extra = available - image_bytes
            self._has_trailer = extra >= 8 * self._n
        self._mmap = np.memmap(
            self.path,
            dtype=np.uint8,
            mode="r",
            offset=SER_HEADER_SIZE,
            shape=(self._file_size - SER_HEADER_SIZE,),
        )
        self._timestamps = None
        if self._has_trailer:
            raw = np.frombuffer(
                self._mmap[image_bytes : image_bytes + 8 * self._n], dtype="<i8"
            ).copy()
            self._timestamps = raw

    def metadata(self) -> ObservationMetadata:
        color = self.color_mode()
        endian = "little" if self._little() else "big"
        extras = {
            "color_id": FieldValue(self.header.color_id, "header"),
            "little_endian_flag": FieldValue(self.header.little_endian_flag, "header"),
            "endian_convention": FieldValue(self.endian_convention, "user"),
            "observer": FieldValue(self.header.observer, "header"),
            "instrument": FieldValue(self.header.instrument, "header"),
            "telescope": FieldValue(self.header.telescope, "header"),
            "datetime_utc_ticks": FieldValue(self.header.datetime_utc, "header"),
            "ser_operator_version": FieldValue(C.SER_OPERATOR_VERSION, "inferred"),
        }
        if self.bayer_override:
            extras["bayer_override"] = FieldValue(self.bayer_override, "user")
        if self.endian_override:
            extras["endian_override"] = FieldValue(self.endian_override, "user")
        if self.header.color_id == COLOR_MONO and self.bayer_override is None:
            extras["mono_or_bayer_choice"] = FieldValue(
                "mono",
                "header",
                "writer labelled mono; user may select a Bayer pattern",
            )
        return ObservationMetadata(
            path=str(self.path),
            n_frames=self._n,
            width=self.header.width,
            height=self.header.height,
            color_mode=color,
            bit_depth=self.header.pixel_depth,
            endian=endian,
            extras=extras,
        )

    def n_frames(self) -> int:
        return self._n

    def frame_shape(self) -> tuple[int, ...]:
        if self.header.planes == 3:
            return (self.header.height, self.header.width, 3)
        return (self.header.height, self.header.width)

    def color_mode(self) -> str:
        if self.bayer_override:
            return self.bayer_override
        return self.header.color_mode

    def timestamps(self) -> np.ndarray | None:
        return None if self._timestamps is None else self._timestamps.copy()

    def _little(self) -> bool:
        if self.endian_override == "little":
            return True
        if self.endian_override == "big":
            return False
        return self.header.samples_are_little_endian

    def read_raw(self, index: int) -> np.ndarray:
        if index < 0 or index >= self._n:
            raise IndexError(index)
        start = index * self.header.frame_bytes
        stop = start + self.header.frame_bytes
        blob = np.frombuffer(self._mmap[start:stop].tobytes(), dtype=np.uint8)
        if self.header.bytes_per_sample == 1:
            samples = blob.astype(np.uint16, copy=False)
        else:
            dt = np.dtype("<u2" if self._little() else ">u2")
            samples = np.frombuffer(blob.tobytes(), dtype=dt).astype(np.uint16, copy=False)
        h, w = self.header.height, self.header.width
        if self.header.planes == 3:
            packed = samples.reshape(h, w, 3)
            if self.color_mode() == "BGR":
                packed = packed[..., ::-1]
            return packed
        return samples.reshape(h, w)

    def close(self) -> None:
        self._mmap = None
        if getattr(self, "_fh", None) is not None:
            self._fh.close()
            self._fh = None
