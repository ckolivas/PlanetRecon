# Separating interpolation and peak refinement across noise strength

The previous experiment (`9687110`) reduced noiseless fractional-matching bias
with cubic sampling plus direct NCC refinement, but its noisy controls
regressed. This follow-up tests the two changes separately at fixed noise
amplitudes before drawing conclusions from their combination.

Here "current" means the existing AP matching path inside the experimental
coherent field fitter. These are controlled comparisons to that baseline,
not measurements of a new GUI production stack.

## Fixed design

The four policies are unchanged: existing bilinear/quadratic matching, cubic
sampling alone, direct peak refinement alone, and both. Every policy keeps the
same reference, AP geometry, coherent field fit, field regularization and final
bilinear stacking sampler. No captured-pixel output filter or normalization is
added.

The stationary, long-motion, short-motion and short-plus-offset controls use
16 evenly spaced positions from the preceding 32-pose sequences (indices
0, 2, ..., 30). Added Gaussian noise sigma is 0, 0.25, 0.5, 1 or 2 ADU. All
doses scale the same underlying seed-9033 noise samples, and the generator
advances through skipped positions, preserving correspondence with the
preceding experiment. Sigma is a synthetic intervention, not an estimate of
Saturn sensor noise.

This gives 80 policy/case/dose cells and 1,280 field applications. The baseline
and combined-policy fields at sigma 0 and 2 are reused from hashed artifacts,
256 fields in total. The first reused field in every endpoint cell is also
recomputed and required to match within 1e-10 pixels. All other fields are
computed fresh. Every output stack contains the same 16 poses; no 16-frame
image metric is compared against a previous 32-frame stack.

Noise-free twins are sampled through each estimated field to measure the
geometric component of stack error. The noiseless exact-motion target uses
the same 16 detector-integrated frames and final sampler. All 84 exports
(20 variants plus one target per case) receive the actual PlanetaryTools
Wavelet 27/0/0/0 followed by Adaptive Deconvolution 15.6, Contrast Adaptive.

The analysis reports both conditional effects: cubic minus bilinear at each
peak method, and refined minus quadratic peaks at each sampler. It also
reports their interaction. These are differences in a chosen metric, not
additive fractions of physical noise. A positive error difference means worse.

Sign changes are reported only as brackets between tested doses. No fitted
universal threshold or monotonic response is assumed. This is an exploratory
screen using one fixed scene and one paired noise realization per pose.

## Development results

The combined policy's field-accuracy advantage changes sign between sigma
0.5 and 1 ADU for stationary, long-motion and offset-motion controls, and
between 1 and 2 ADU for short motion. These are case-specific observed brackets,
not a noise threshold that can be transferred to Saturn or an interface rule.

At sigma 2, vector field RMSE in pixels separates the main culprit:

| Case | Current | Cubic only | Refinement only | Both |
|---|---:|---:|---:|---:|
| Stationary | 0.23006 | 0.25382 | 0.23054 | 0.25378 |
| Long motion | 0.30254 | 0.31923 | 0.30098 | 0.31775 |
| Short motion | 0.78164 | 0.78880 | 0.77819 | 0.78581 |
| Short + offset | 0.94766 | 0.94939 | 0.94247 | 0.95054 |

Cubic sampling increases high-noise field error whether the quadratic or
refined peak estimator is used. Refinement alone is close to the current
stationary result and slightly better in these three moving cases. The
interaction matters: the offset case's combined result is worse than either
individual change.

Sharpened-image behavior is not identical to field accuracy. Refinement alone
reduces sharpened error by 9.34% and 10.81% at sigma 0.25 in long and short
motion, respectively; at sigma 0.5 the reductions are 5.45% and 5.63%. At
sigma 2, changes are much smaller: reductions of 0.24%, 0.27%, 0.05% and 0.12%
for the four cases. Other cells regress, so this is not a uniform improvement.

The stationary combined-policy sharpened error at sigma 2 decreases slightly
in this 16-pose subset, whereas the previous full 32-pose experiment increased.
Both results are retained. Different frame subsets and nonlinear sharpening
make a small error change unreliable as a general claim; this prompted a
separate validation rather than a production change.

## Fresh-noise and unused-phase validation

Refinement alone was frozen without retuning, then compared with the current
matcher at sigma 0.25, 0.5, 1 and 2 ADU. Validation uses the complementary odd
positions (1, 3, ..., 31) and a fresh seed, 29033. Reference, scene, PSF and
all other settings stay fixed. The validation therefore changes both noise
realization and phases; their contributions to differences are not separated.

This adds 32 cells, 512 fresh field estimates and 36 identically sharpened
exports. One case worker was terminated before writing any output; only that
empty case was rerun. All four cases then completed and passed artifact audits.

Validation change in sharpened RMSE relative to the current matcher:

| Case | Sigma 0.25 | Sigma 0.5 | Sigma 1 | Sigma 2 |
|---|---:|---:|---:|---:|
| Stationary | −6.50% | −2.60% | −0.50% | +0.23% |
| Long motion | −12.15% | −6.91% | +0.67% | +0.09% |
| Short motion | −8.93% | −6.85% | −5.32% | −0.24% |
| Short + offset | +1.53% | +1.42% | +0.27% | −0.11% |

The low-noise sharpened gains for long and short motion reproduce, but the
low-noise offset improvement does not. Small high-noise effects also change
sign between development and validation. No confidence interval is inferred
from these two small fixed-scene runs.

Crucially, the clean twins do not endorse the strongest sharpened gains.
At sigma 0.25/0.5, long-motion clean-image error **increases 12.1%/18.4%**,
and short-motion clean-image error **increases 22.9%/27.6%**, even while their
sharpened noisy-image errors fall. Thus a lower sharpened RMSE cannot be
described as better motion or better recovered detail. In the high-noise
sigma-2 controls, clean-image error instead improves by 8.2–12.8% across all
four cases, while sharpened changes remain small.

All newly computed development and validation fits retain sufficient APs and
pass their final geometric guard. Endpoint parity errors are below the declared
1e-10-pixel tolerance.

## Decision

Do not promote cubic matching or introduce a noise-dependent automatic switch
from these data. Cubic sampling explains most of the high-noise field regression
in the combined method. Direct refinement with bilinear matching is a separate
candidate with reproducible but conditional effects.

A bounded real-capture pilot of refinement alone is now more informative than
another combined-policy simulation. It must keep frames, reference, weights,
final sampling and exact sharpening fixed, and examine detail/edge behavior
alongside grain and split-stack repeatability. The clean-twin regressions show
why choosing solely by apparent smoothness or sharpened RMSE would be unsafe
scientifically. No Saturn improvement or AutoStakkert equivalence is claimed.

## Reproduction

```sh
PYTHONPATH=. OPENBLAS_NUM_THREADS=4 .venv/bin/python tools/ap_noise_ladder.py \
  --out out/ap-noise-ladder --prepare
# Run one worker per case; they may run concurrently and use four CPU threads each.
PYTHONPATH=. OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python tools/ap_noise_ladder.py --out out/ap-noise-ladder --case stationary
# Repeat for long_motion, short_motion and short_offset.
```

After all four case workers finish, apply the following sharpening command
to each case directory, substituting its name in both paths:

```sh
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
MKL_NUM_THREADS=4 QT_QPA_PLATFORM=offscreen \
  ../PlanetaryTools/planetary-app/.venv/bin/python tools/planetarytools_sharpen_experiment.py \
  --planetary-tools ../PlanetaryTools/planetary-app \
  --out out/ap-noise-ladder/stationary/sharpened out/ap-noise-ladder/stationary/*.png
PYTHONPATH=. OPENBLAS_NUM_THREADS=4 .venv/bin/python tools/analyse_ap_noise_ladder.py
MPLCONFIGDIR=/tmp/pr-mpl /usr/bin/python3 tools/plot_ap_noise_ladder.py
PYTHONPATH=. OPENBLAS_NUM_THREADS=4 .venv/bin/python -m pytest -q tests/test_ap_noise_ladder.py
```

The output directory must be fresh. Four independent case workers use CUDA;
system Matplotlib is already installed. No packages are installed and no
production settings are changed.

The independent validation can be reproduced after the development analysis:

```sh
PYTHONPATH=. OPENBLAS_NUM_THREADS=4 .venv/bin/python tools/validate_ap_noise_ladder.py --prepare
PYTHONPATH=. OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python tools/validate_ap_noise_ladder.py --case stationary
# Repeat the worker for long_motion, short_motion and short_offset.
# Apply the same sharpening command to each case under out/ap-noise-ladder-validation.
PYTHONPATH=. OPENBLAS_NUM_THREADS=4 .venv/bin/python tools/analyse_ap_noise_ladder_validation.py
MPLCONFIGDIR=/tmp/pr-mpl /usr/bin/python3 tools/plot_ap_noise_ladder_validation.py
```

Two focused tests pass, covering conditional factorial effects, interactions,
multiple sign-change brackets and unique cell identifiers. Analyzers recheck
source/input hashes, pose counts, raw-array metrics and all 120 exact-sharpening
exports. Both charts were visually inspected.

Numerical records are `results/registration/ap-noise-ladder.json` and
`results/registration/ap-noise-ladder-validation.json`. Raw arrays and charts
are in the matching directories under `out/`. Production remains unchanged.
