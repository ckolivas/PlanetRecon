# Matching without detector noise and exchanging noise realizations

This continues the known-motion AP experiment in `3077cf4`. It repeats
matching with no added noise, then uses an independent second noise realization
to test whether assigning each image its own estimated field creates a
detectable coupling between alignment and grain.

## Design

The same continuous scene, reference, 32 motion phases, sampling, mask and
four motion cases are reused. Noise A is the preceding sigma-2 ADU additive
Gaussian realization, seed 9033. Noise B uses seed 19033. Both share the same
clean signal at every motion phase. This noise level is an experimental
intervention, not a measurement of Saturn's sensor noise.

The original A-derived fields are reused from hashed arrays. Fields are also
estimated from clean frames and from B. Reconstructing A with the saved fields
reproduces the preceding noisy stack, clean twin and exact-motion target within
1e-9 ADU. Every estimator uses the same known global translation and the same
coherent AP fitter. Matching without added noise retains detector integration,
proxy processing, fractional resampling and multiscale composition.

For each phase, let A and B denote the independently noisy images and WA/WB
the final samplers using their estimated fields. Two paired stacks are formed:

- Same-noise assignment: `(WA(A) + WB(B)) / 2`.
- Cross-noise assignment: `(WA(B) + WB(A)) / 2`.

The source images, motion phases, fields and scalar weights are identical
between these stacks. Only their assignment changes. Both contain 32 phases
times two noise realizations, or 64 contributions. When the common clean
signal is substituted, the paired outputs coincide exactly. Their raw
difference therefore comes entirely from noise being paired with the field
it helped estimate. This is not an independent estimate of total noise.

Single-A outputs are also retained for the original noisy matcher, clean
matcher, independent-B matcher and exact-motion sampler. The paired controls
use the same A+B samples with either clean-derived fields or exact motion;
comparisons do not mix the 32-frame and 64-contribution outputs.

## No-added-noise results

Vector field RMSE over the same object mask, in pixels:

| Case | Clean matcher | Noise-A matcher | Noise-B matcher |
|---|---:|---:|---:|
| Stationary, with camera translation | 0.05652 | 0.24297 | 0.27050 |
| Long motion | 0.21133 | 0.31124 | 0.34643 |
| Short motion | 0.75131 | 0.78828 | 0.81561 |
| Short motion + offset | 0.92773 | 0.95684 | 0.97131 |

Noise substantially affects the stationary and long-motion estimates. Much
of the short-motion error persists without it. This agrees with the preceding
oracle experiment, which found substantial field reconstruction error even
with exact patch-average observations.

AP displacement RMSE against the known patch-average model, measured only on
the common set accepted by all three estimators:

| Case | Clean, px | Noise A, px | Noise B, px |
|---|---:|---:|---:|
| Stationary | 0.04134 | 0.17118 | 0.17750 |
| Long motion | 0.06305 | 0.18175 | 0.18564 |
| Short motion | 0.09382 | 0.19011 | 0.19423 |
| Short + offset | 0.10997 | 0.19883 | 0.20520 |

Thus there is also a systematic disagreement between noiseless measured AP
motion and the fitter's scalar gradient-weighted observation model. It could
include matching, composition and interpolation effects; the experiment does
not identify which stage contributes how much. Even the stationary case has
residual error despite a supplied exact global shift, so deformation alone
cannot explain all of that disagreement.

Clean-stack RMSE in ADU, using the estimated fields on noiseless twins:

| Case | Clean matcher | Noise-A matcher | Noise-B matcher |
|---|---:|---:|---:|
| Stationary | 0.07837 | 0.22488 | 0.27752 |
| Long motion | 0.07446 | 0.23555 | 0.27985 |
| Short motion | 0.06851 | 0.21602 | 0.27000 |
| Short + offset | 0.26508 | 0.32956 | 0.37235 |

All newly computed fits retained enough APs and passed the geometric guard.

## Does matching the grain change the result?

Yes, in this controlled experiment. The raw same-minus-cross image has RMS
0.06764, 0.06733, 0.06749 and 0.06558 ADU respectively across the four cases.
Their clean-signal twins are identical. However, the total raw image RMSE is
almost unchanged, and a nonzero coupling is not itself evidence of worse
overall reconstruction.

All 36 exports were processed by the actual PlanetaryTools implementation:
Wavelet 27/0/0/0, followed by Adaptive Deconvolution 15.6, Contrast Adaptive.
Recipe and implementation hashes match the preceding experiments.

Sharpened image RMSE against the exact-motion noiseless target, linear 0–1:

| Case | Same-noise assignment | Cross-noise assignment | Same relative to cross |
|---|---:|---:|---:|
| Stationary | 0.0071544 | 0.0066623 | +7.39% |
| Long motion | 0.0069715 | 0.0070374 | −0.94% |
| Short motion | 0.0066590 | 0.0067408 | −1.21% |
| Short + offset | 0.0072130 | 0.0071545 | +0.82% |

The stationary result is consistent with matching the frame's own noise
making the sharpened image worse. Moving cases are mixed and much closer.
Nonlinear sharpening means the raw additive noise-only difference cannot be
carried through unchanged to the sharpened outputs. Absolute error also mixes
remaining noise, geometry and interpolation effects; known true motion need
not minimize this particular sharpened metric.

The records retain both error against the noiseless target and error against
an oracle using the same noisy source samples, without choosing one favorable
metric after the fact. These are fixed-scene exploratory controls, not
population estimates or confidence intervals.

## Decision

The algorithm can couple alignment to noise. That mechanism is now directly
demonstrated for the synthetic controls, but the results do not establish it as
the cause of Saturn's grain or explain the much larger AutoStakkert gap.
No sensor-noise level has been inferred from the real capture, and this result
does not justify output filtering or a production change.

The next focused check is the residual motion in the noiseless translated
scene: sweep known fractional translations and inspect the matching stages.
The current exact-global-shift control still invents about 0.057 px RMS of
local motion. Testing the matching sampler and subpixel peak estimation there
can isolate a systematic bias without atmospheric ambiguity, another field
regularization sweep, or a new full Saturn replay.

## Reproduction and verification

```sh
PYTHONPATH=. OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python tools/ap_noise_coupling.py --out out/ap-noise-coupling --device cuda:0
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
MKL_NUM_THREADS=4 QT_QPA_PLATFORM=offscreen \
  ../PlanetaryTools/planetary-app/.venv/bin/python tools/planetarytools_sharpen_experiment.py \
  --planetary-tools ../PlanetaryTools/planetary-app \
  --out out/ap-noise-coupling/sharpened out/ap-noise-coupling/*.png
PYTHONPATH=. OPENBLAS_NUM_THREADS=4 .venv/bin/python tools/analyse_ap_noise_coupling.py
MPLCONFIGDIR=/tmp/pr-mpl /usr/bin/python3 tools/plot_ap_noise_coupling.py
PYTHONPATH=. OPENBLAS_NUM_THREADS=4 .venv/bin/python -m pytest -q tests/test_ap_noise_coupling.py
```

The focused regression passes. It verifies clean-signal cancellation and the
noise-coupling identity with distinct linear samplers, including the zero
coupling case when both fields coincide. The analyzer rechecks source and
artifact hashes, frame counts, prior replay parity, saved-array metrics and
the exact sharpening recipe. The chart was visually reviewed. System
Matplotlib was already installed; no packages were added.

Numerical record: `results/registration/ap-noise-coupling.json`. Raw arrays,
sharpened outputs and the comparison chart are in `out/ap-noise-coupling/`.
Production settings, pixel filtering and brightness normalization are unchanged.
