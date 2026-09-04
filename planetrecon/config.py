"""Simulation configuration derived from locked R8 parameters."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass

from planetrecon import constants as C


@dataclass(frozen=True)
class SimConfig:
    seed: int
    dr0: float
    n_frames: int = C.N_FRAMES
    n_diam: int = C.DEFAULT_N_DIAM
    pupil_pad_factor: float = C.DEFAULT_PUPIL_PAD_FACTOR
    subharmonic_levels: int = C.DEFAULT_SUBHARMONICS
    exposure_samples_j: int = C.DEFAULT_J
    padding_detector_px: int = C.DEFAULT_PADDING_DET_PX
    eval_size: int = C.EVAL_SIZE

    @property
    def d_m(self) -> float:
        return C.D_M

    @property
    def obstruction_ratio(self) -> float:
        return C.OBSTRUCTION_RATIO

    @property
    def wavelength_m(self) -> float:
        return C.WAVELENGTH_M

    @property
    def wind_m_s(self) -> float:
        return C.WIND_M_S

    @property
    def r0_m(self) -> float:
        return C.D_M / self.dr0

    @property
    def tau0_s(self) -> float:
        return C.TAU0_FACTOR * self.r0_m / C.WIND_M_S

    @property
    def texp_s(self) -> float:
        return C.TEXP_OVER_TAU0 * self.tau0_s

    @property
    def dt_s(self) -> float:
        return C.DT_OVER_TAU0 * self.tau0_s

    @property
    def detector_pixel_scale_rad(self) -> float:
        return C.DETECTOR_SAMP_FACTOR * C.WAVELENGTH_M / C.D_M

    @property
    def detector_pixel_scale_arcsec(self) -> float:
        return self.detector_pixel_scale_rad * (180.0 / math.pi) * 3600.0

    @property
    def f_c(self) -> float:
        return C.D_M / C.WAVELENGTH_M

    @property
    def pupil_dx_m(self) -> float:
        return C.D_M / self.n_diam

    @property
    def pupil_grid_size(self) -> int:
        n = int(round(self.pupil_pad_factor * self.n_diam))
        if n % 2:
            n += 1
        return n

    @property
    def optical_pixel_scale_rad(self) -> float:
        return C.WAVELENGTH_M / (self.pupil_grid_size * self.pupil_dx_m)

    @property
    def bin_factor(self) -> int:
        ratio = self.detector_pixel_scale_rad / self.optical_pixel_scale_rad
        n = int(round(ratio))
        if abs(ratio - n) > 1e-8:
            raise ValueError(
                f"bin factor is not integer: optical/detector = {ratio}"
            )
        if n < 1:
            raise ValueError("bin factor < 1")
        return n

    @property
    def screen_dx_m(self) -> float:
        return self.pupil_dx_m

    @property
    def interp_margin_m(self) -> float:
        return C.SCREEN_MARGIN_PUPIL_PX * self.pupil_dx_m

    def travel_m(self, n_frames: int | None = None) -> float:
        n = self.n_frames if n_frames is None else n_frames
        return C.WIND_M_S * ((n - 1) * self.dt_s + self.texp_s)

    @property
    def screen_length_x_m(self) -> float:
        # Always size for the locked N=500, longer mandatory regime so paired
        # seeds share one unit field and short test runs cannot wrap.
        dt_long = C.TAU0_FACTOR * (C.D_M / 4.0) / C.WIND_M_S
        texp_long = C.TEXP_OVER_TAU0 * dt_long
        travel = C.WIND_M_S * ((C.N_FRAMES - 1) * dt_long + texp_long)
        return travel + C.D_M + 2.0 * self.interp_margin_m

    @property
    def screen_length_y_m(self) -> float:
        return max(
            C.D_M + 2.0 * self.interp_margin_m,
            C.SCREEN_CROSSWIND_DIAMS * C.D_M,
        )

    @property
    def screen_nx(self) -> int:
        n = int(math.ceil(self.screen_length_x_m / self.screen_dx_m)) + 2
        return n + (n % 2)

    @property
    def screen_ny(self) -> int:
        n = int(math.ceil(self.screen_length_y_m / self.screen_dx_m)) + 2
        return n + (n % 2)

    @property
    def object_n4(self) -> int:
        n_det = int(2 * C.PLANET_RADIUS_DET_PX + 2 * self.padding_detector_px)
        return n_det * C.OBJECT_OVERSAMPLE

    def to_json_dict(self) -> dict:
        d = asdict(self)
        d.update(
            {
                "D_m": self.d_m,
                "obstruction_ratio": self.obstruction_ratio,
                "wavelength_m": self.wavelength_m,
                "wind_m_s": self.wind_m_s,
                "r0_m": self.r0_m,
                "tau0_s": self.tau0_s,
                "texp_s": self.texp_s,
                "dt_s": self.dt_s,
                "detector_pixel_scale_rad": self.detector_pixel_scale_rad,
                "detector_pixel_scale_arcsec": self.detector_pixel_scale_arcsec,
                "pupil_grid_size": self.pupil_grid_size,
                "screen_dx_m": self.screen_dx_m,
                "screen_shape": [self.screen_ny, self.screen_nx],
                "bin_factor": self.bin_factor,
                "paired_unit_screen": True,
                "r0_ref_m": C.R0_REF_M,
                "revision": C.REVISION,
            }
        )
        return d

    def json_utf8(self) -> bytes:
        return json.dumps(self.to_json_dict(), sort_keys=True, indent=2).encode(
            "utf-8"
        )


def make_config(seed: int, dr0: float, **kwargs) -> SimConfig:
    dr0 = float(dr0)
    if dr0 <= 0:
        raise ValueError("D/r0 must be positive")
    return SimConfig(seed=int(seed), dr0=dr0, **kwargs)
