"""Detector-space calibration. Missing tables yield labelled approximate-noise."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Calibration:
    bias: np.ndarray | None = None
    dark: np.ndarray | None = None
    flat: np.ndarray | None = None
    gain_e_per_adu: float | None = None
    read_noise_e: float | None = None
    saturate_adu: float | None = None

    @property
    def mode(self) -> str:
        if self.gain_e_per_adu is None:
            return "approximate-noise"
        return "calibrated"


def load_calibration(config, shape: tuple[int, ...]) -> Calibration | None:
    """Load detector-lattice NPY tables, without broadcasting or pickle objects."""
    values = {name: getattr(config, name) for name in ("gain_e_per_adu", "read_noise_e", "saturate_adu")}
    for name in ("bias", "dark", "flat"):
        path = getattr(config, name + "_path")
        if path is None:
            values[name] = None
            continue
        array = np.load(path, allow_pickle=False)
        if not isinstance(array, np.ndarray):
            array.close()
            raise ValueError(f"{name} must be an NPY array")
        if array.shape != shape or array.dtype.kind not in "fiu" or not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must be finite and match the raw detector frame shape {shape}")
        if name == "flat" and np.any(array <= 0):
            raise ValueError("flat values must be positive")
        values[name] = np.asarray(array, dtype=np.float64)
    return Calibration(**values) if any(v is not None for v in values.values()) else None


def apply_calibration(frame: np.ndarray, cal: Calibration | None) -> tuple[np.ndarray, dict]:
    work = np.asarray(frame, dtype=np.float64)
    info = {"mode": "approximate-noise", "saturated": False}
    if cal is None:
        return work, info
    if cal.bias is not None:
        work = work - cal.bias
    if cal.dark is not None:
        work = work - cal.dark
    if cal.flat is not None:
        flat = np.asarray(cal.flat, dtype=np.float64)
        if flat.shape != work.shape or not np.isfinite(flat).all() or np.any(flat <= 0):
            raise ValueError('flat must be finite, positive and match the raw detector frame shape')
        # Flat illumination has arbitrary units. Form relative sensitivity
        # before correcting observations; an absolute floor changes the image
        # when the same flat is saved on a smaller numeric scale.
        relative = flat / flat.max()
        work = work * (float(np.median(relative)) / relative)
    sat = False
    if cal.saturate_adu is not None:
        sat = bool(np.any(np.asarray(frame) >= cal.saturate_adu))
    if cal.gain_e_per_adu is not None:
        work = work * float(cal.gain_e_per_adu)
        info["mode"] = "calibrated"
        info["units"] = "e-"
    else:
        info["units"] = "adu"
    info["saturated"] = sat
    info["mode"] = cal.mode
    return work, info
