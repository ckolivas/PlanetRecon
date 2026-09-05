"""Locked R9 physical constants and object geometry."""

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
REVISION = "R9"
ROADMAP_REVISION = "R10"
# Simulator convolution/bin/crop contract. Bump only when that operator changes.
SIMULATOR_OPERATOR_VERSION = "1.0"
# Estimator projection/quadratic contract. W01 replaces clip-then-support with
# Dykstra projection onto positivity ∩ spectral support.
ESTIMATOR_OPERATOR_VERSION = "1.2"
DEV_SEEDS = (1001, 1002, 1003)
EVAL_SEEDS = tuple(range(2001, 2013))
EXT_SEEDS = tuple(range(2013, 2025))
MANDATORY_DR0 = (8.0, 4.0)
R0_REF_M = 1.0
DEFAULT_N_DIAM = 64
DEFAULT_PUPIL_PAD_FACTOR = 8.0
DEFAULT_SUBHARMONICS = 4
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
TILT_LOFREQ_TOL = 0.10
TILT_GRID_TOL = 0.05
MEASUREMENT_APERTURE_SIGMA = 2.0
MEASUREMENT_ANNULUS_INNER_SIGMA = 3.0
MEASUREMENT_ANNULUS_OUTER_SIGMA = 5.0

# Prompt 2 — ranking, estimators, gaps (R9)
RANK_LAPLACIAN_KERNEL = ((0.0, 1.0, 0.0), (1.0, -4.0, 1.0), (0.0, 1.0, 0.0))
RANK_BORDER_PX = 2
RANK_P_GRID = (5, 10, 25, 50, 100)
DECISION_P = 10
RH_ILLCONDITIONED = 1e-5
MID_BAND_RHO_MIN = 0.2
MID_BAND_RHO_MAX = 0.5
E1_E2A0_EH_REL_TOL = 1e-4
E1_E2A0_IMAGE_REL_TOL = 1e-6
G3_SUM_TOL = 1e-10
OVAL_CONTRAST_PATHOLOGY = 0.05
G_STRONG_MEDIAN = 0.10
G_NEGATIVE_MEDIAN = 0.05
G_STRONG_POSITIVE_FRAC = 2.0 / 3.0
G_NEGATIVE_GE10_FRAC = 0.25
DEN_FLOOR = 1e-15
# Frozen on development seeds 1001–1003, both mandatory regimes, feature-rich
# crop, E1(S_100). Smallest λ_rel whose median E_H is within 2% of the grid
# minimum (grid minimum was 0.03). E2a uses the same quadratic stabilisation
# plus positivity and |f|<=f_c support.
E1_LAMBDA_REL = 3.0e-2
E2A_LAMBDA_REL = 3.0e-2
E1_LAMBDA_GRAD = 0.0
E2A_FISTA_MAXITER = 300
E2A0_CG_MAXITER = 200
E2A_FISTA_TOL = 1e-6
SUPPORT_RHO_MAX = 1.0
DYKSTRA_MAXITER = 2048
DYKSTRA_WARM_MAXITER = 256
DYKSTRA_TOL = 1e-12
FEASIBLE_POS_TOL = 1e-8
FEASIBLE_SUPPORT_TOL = 1e-10
RANK_LOFREQ_JACCARD_MIN = 0.50
RANK_LOFREQ_SPEARMAN_MIN = 0.85
DEFAULT_CPU_THREADS = 32
MAX_CPU_THREADS = 32
FRAME_SOURCE_SCHEMA = "planetrecon-frame-source"
FRAME_SOURCE_SCHEMA_VERSION = "1.0"
RESULT_SCHEMA = "planetrecon-reconstruction-result"
RESULT_SCHEMA_VERSION = "1.0"
CONFIG_SCHEMA = "planetrecon-reconstruction-config"
CONFIG_SCHEMA_VERSION = "1.0"
JOB_SCHEMA = "planetrecon-job"
JOB_SCHEMA_VERSION = "1.0"
BASELINE_OPERATOR_VERSION = "1.0"
SER_OPERATOR_VERSION = "1.0"

# Prompt Q2 — E2b production prior and blind D / D-tail (R9 §8.5, §18)
# Quadratic stabilisation stays at the Prompt-2 freeze. Charbonnier TV μ was
# scanned on development seeds 1001–1003, both regimes, feature-rich E2b(S_100).
# Smallest μ within 2% of the grid minimum is 0 (the grid min at 0.03 improves
# median E_H by ≪ 2%). E2* remains E2b; the matching G3-like gap is not
# prior-limited.
E2B_LAMBDA_REL = E2A_LAMBDA_REL
E2B_TV = 0.0
E2B_TV_EPS = 1.0
E2_STAR = "E2b"
M_FIT_GRID = (15, 35, 60)
Q2_OUTER_ITERS = (3, 2, 2)
Q2_ALPHA_ITERS = 4
Q2_FRAME_WORKERS = 1
Q2_HOLDOUT_FRAC = 0.10
Q2_CLOSURE_TARGET = 0.40
Q2_PRIOR_LIMITED_RATIO = 0.5
Q2_SCOPE_DR0 = (4.0,)
Q2_INITS = ("zero", "subset")

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
        "x_px": 40.0,
        "y_px": 18.0,
        "sigma_x_px": 2.5,
        "sigma_y_px": 4.0,
        "angle_deg": -15.0,
        "contrast": 0.12,
    },
    {
        "name": "oval3",
        "x_px": -25.0,
        "y_px": -40.0,
        "sigma_x_px": 1.8,
        "sigma_y_px": 3.0,
        "angle_deg": 35.0,
        "contrast": -0.12,
    },
)
