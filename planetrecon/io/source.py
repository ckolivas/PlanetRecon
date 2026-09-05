"""Production FrameSource contract. Truth arrays never travel this interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Literal

import numpy as np

from planetrecon import constants as C


Origin = Literal["measured", "header", "inferred", "user"]


@dataclass(frozen=True)
class FieldValue:
    value: Any
    origin: Origin
    note: str = ""


@dataclass
class ObservationMetadata:
    path: str
    n_frames: int
    width: int
    height: int
    color_mode: str
    bit_depth: int
    endian: str
    schema_name: str = C.FRAME_SOURCE_SCHEMA
    schema_version: str = C.FRAME_SOURCE_SCHEMA_VERSION
    extras: dict[str, FieldValue] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        extra = {
            key: {"value": fv.value, "origin": fv.origin, "note": fv.note}
            for key, fv in self.extras.items()
        }
        return {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "path": self.path,
            "n_frames": int(self.n_frames),
            "width": int(self.width),
            "height": int(self.height),
            "color_mode": self.color_mode,
            "bit_depth": int(self.bit_depth),
            "endian": self.endian,
            "extras": extra,
        }


class FrameSource(ABC):
    """Indexed raw reads with bounded batches. Production sources expose no truth."""

    @abstractmethod
    def metadata(self) -> ObservationMetadata:
        raise NotImplementedError

    @abstractmethod
    def n_frames(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def frame_shape(self) -> tuple[int, ...]:
        raise NotImplementedError

    @abstractmethod
    def color_mode(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def read_raw(self, index: int) -> np.ndarray:
        raise NotImplementedError

    def timestamps(self) -> np.ndarray | None:
        return None

    def iter_batches(
        self,
        batch: int,
        start: int = 0,
        stop: int | None = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        n = self.n_frames()
        stop = n if stop is None else min(stop, n)
        start = max(0, int(start))
        batch = max(1, int(batch))
        for origin in range(start, stop, batch):
            end = min(origin + batch, stop)
            frames = np.stack([self.read_raw(i) for i in range(origin, end)], axis=0)
            yield np.arange(origin, end, dtype=np.int64), frames

    def close(self) -> None:
        return None

    def __enter__(self) -> FrameSource:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def has_truth(self) -> bool:
        return False


class ArraySource(FrameSource):
    """In-memory observed frames for tests. Still no truth channel."""

    def __init__(
        self,
        frames: np.ndarray,
        *,
        color_mode: str = "mono",
        bit_depth: int = 16,
        path: str = "memory://array",
        extras: dict[str, FieldValue] | None = None,
    ):
        self._frames = np.asarray(frames)
        if self._frames.ndim not in (3, 4):
            raise ValueError("frames must be (N,H,W) or (N,H,W,C)")
        self._color_mode = color_mode
        self._bit_depth = int(bit_depth)
        self._path = path
        self._extras = extras or {}

    def metadata(self) -> ObservationMetadata:
        shape = self._frames.shape
        return ObservationMetadata(
            path=self._path,
            n_frames=int(shape[0]),
            width=int(shape[2]),
            height=int(shape[1]),
            color_mode=self._color_mode,
            bit_depth=self._bit_depth,
            endian="little",
            extras=dict(self._extras),
        )

    def n_frames(self) -> int:
        return int(self._frames.shape[0])

    def frame_shape(self) -> tuple[int, ...]:
        return tuple(int(v) for v in self._frames.shape[1:])

    def color_mode(self) -> str:
        return self._color_mode

    def read_raw(self, index: int) -> np.ndarray:
        if index < 0 or index >= self.n_frames():
            raise IndexError(index)
        return np.array(self._frames[index], copy=True)


def open_source(path: str | Path, **kwargs) -> FrameSource:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".ser":
        from planetrecon.io.ser import SERSource

        return SERSource(path, **kwargs)
    if suffix in {".h5", ".hdf5"}:
        from planetrecon.io.hdf5_source import HDF5ObservedSource

        return HDF5ObservedSource(path, **kwargs)
    raise ValueError(f"unsupported capture type: {path.suffix}")
