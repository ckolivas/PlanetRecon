"""Observed-only adapter for Gate-1 HDF5 files. Truth arrays are not exposed."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from planetrecon.io.source import FieldValue, FrameSource, ObservationMetadata


FORBIDDEN_TRUTH = (
    "/object/feature_truth",
    "/object/bland_truth",
    "/object/full_latent_4x",
    "/object/full_latent_detector",
    "/truth_transfer/feature/otf_bar",
    "/truth_transfer/feature/psf_bar",
)


class HDF5ObservedSource(FrameSource):
    """Read noisy observed crops only. Passing truth=True is rejected."""

    def __init__(self, path: str | Path, crop: str = "feature", **kwargs):
        if kwargs.get("truth"):
            raise ValueError("production FrameSource cannot expose synthetic truth")
        if crop not in ("feature", "bland"):
            raise ValueError(f"unknown crop {crop!r}")
        self.path = Path(path)
        self.crop = crop
        self._file = h5py.File(self.path, "r")
        ds_name = f"/frames/{crop}_observed_e"
        if ds_name not in self._file:
            self._file.close()
            raise KeyError(ds_name)
        self._ds = self._file[ds_name]
        if self._ds.ndim != 3 or any(v == 0 for v in self._ds.shape):
            self._file.close()
            raise ValueError("observed frames must have nonempty shape (N,H,W)")
        self._n = int(self._ds.shape[0])
        self._h = int(self._ds.shape[1])
        self._w = int(self._ds.shape[2])

    def metadata(self) -> ObservationMetadata:
        cfg = self._file["/config"].attrs if "/config" in self._file else {}
        extras = {
            "seed": FieldValue(int(self._file.attrs.get("seed", -1)), "header"),
            "Dr0": FieldValue(float(cfg.get("Dr0", float("nan"))), "header"),
            "crop": FieldValue(self.crop, "user"),
            "truth_available": FieldValue(False, "inferred", "stripped from production source"),
        }
        return ObservationMetadata(
            path=str(self.path),
            n_frames=self._n,
            width=self._w,
            height=self._h,
            color_mode="mono",
            bit_depth=32,
            endian="little",
            units="e-",
            extras=extras,
        )

    def n_frames(self) -> int:
        return self._n

    def frame_shape(self) -> tuple[int, ...]:
        return (self._h, self._w)

    def color_mode(self) -> str:
        return "mono"

    def read_raw(self, index: int) -> np.ndarray:
        if index < 0 or index >= self._n:
            raise IndexError(index)
        return np.asarray(self._ds[index], dtype=np.float64)

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
