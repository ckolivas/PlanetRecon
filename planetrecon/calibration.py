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
        work = work / np.clip(cal.flat, 1e-6, None)
        work *= float(np.median(cal.flat))
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
