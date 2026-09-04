"""Locked R8 physical constants and object geometry."""

from __future__ import annotations

D_M = 0.250
OBSTRUCTION_RATIO = 0.30
WAVELENGTH_M = 610e-9
WIND_M_S = 5.0
TAU0_FACTOR = 0.314
TEXP_OVER_TAU0 = 0.30
DT_OVER_TAU0 = 1.0
N_FRAMES = 500
READ_NOISE_E = 2.0
DETECTOR_SAMP_FACTOR = 0.5  # pixels per λ/D; 0.5 λ/D rad/pixel
EVAL_SIZE = 128
OBJECT_OVERSAMPLE = 4
PLANET_RADIUS_DET_PX = 80.0
LIMB_DARKENING_U = 0.45
T0_S = 0.58875e-3
REF_MEAN_E_AT_T0 = 800.0
LIMB_EXCLUSION_PX = 4.0
CROP_LIMB_MARGIN_PX = 16
TUKEY_ALPHA = 0.125
HIGH_BAND_RHO_MIN = 0.5
HIGH_BAND_MTF_MIN = 0.05
SCHEMA_NAME = "planetary-mfbd-gate1"
SCHEMA_VERSION = "1.0"
REVISION = "R8"
DEV_SEEDS = (1001, 1002, 1003)
EVAL_SEEDS = tuple(range(2001, 2013))
EXT_SEEDS = tuple(range(2013, 2025))
MANDATORY_DR0 = (8.0, 4.0)
R0_REF_M = 1.0
DEFAULT_N_DIAM = 64
DEFAULT_PUPIL_PAD_FACTOR = 8.0
DEFAULT_SUBHARMONICS = 3
DEFAULT_J = 8
DEFAULT_PADDING_DET_PX = 64
SCREEN_MARGIN_PUPIL_PX = 8
SCREEN_CROSSWIND_DIAMS = 8.0
PSF_ENERGY_TOL = 1e-6
STRUCTURE_FUNCTION_REL_TOL = 0.25
KL_MODES = 60
EH_EXPOSURE_TOL = 0.005
EH_PADDING_TOL = 0.005
EH_GRID_TOL = 0.02
STREHL_LOFREQ_TOL = 0.02

OVALS = (
    {
        "name": "oval1",
        "x_px": 45.0,
        "y_px": -12.0,
        "sigma_x_px": 2.0,
        "sigma_y_px": 3.5,
        "angle_deg": 20.0,
        "contrast": -0.16,
    },
    {
        "name": "oval2",
        "x_px": 20.0,
        "y_px": 18.0,
        "sigma_x_px": 2.5,
        "sigma_y_px": 4.0,
        "angle_deg": -15.0,
        "contrast": 0.12,
    },
    {
        "name": "oval3",
        "x_px": -25.0,
        "y_px": -24.0,
        "sigma_x_px": 1.8,
        "sigma_y_px": 3.0,
        "angle_deg": 35.0,
        "contrast": -0.12,
    },
)
