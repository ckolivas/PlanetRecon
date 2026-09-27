# Fractional sampling and subpixel AP peak estimation

This follows the no-added-noise experiment in `70df556`, which still found
0.0565 pixels RMS of false local motion in translated, undeformed images whose
global shifts were supplied exactly. The experiment changes matching only;
final pixel accumulation remains the original bilinear sampler.

## Fixed factorial screen

The independent continuous Gaussian scene generator produces exact detector
pixel integrals at 25 translations: each axis takes −0.5, −0.25, 0, 0.25 or
0.5 pixels. Three extra integer translations and the zero-shift self-match
serve as controls. The scene has a fixed sigma-1 circular PSF and no added
noise. Every policy shares the reference, AP geometry and coherent field fit.

Four policies are specified before the sweep:

| Policy | Matching interpolation | Subpixel peak estimate |
|---|---|---|
| Baseline | Bilinear | Existing quadratic estimate |
| Cubic | Catmull–Rom cubic | Existing quadratic estimate |
| Refined | Bilinear | Direct fractional NCC refinement |
| Both | Catmull–Rom cubic | Direct fractional NCC refinement |

Interpolation changes only the warped matching proxy and fractional patches
used for forward/reverse matching. Field composition and final stacking remain
bilinear. Catmull–Rom interpolation has negative coefficients; this experiment
does not use it to resample captured pixels into the final stack.

The direct refinement starts from the better-scoring integer or quadratic
proposal. Three finite-difference steps (0.25, 0.125, 0.0625 pixels) estimate
the local NCC gradient and Hessian. Newton proposals are bounded and accepted
only if the directly evaluated fractional NCC improves. The original integer
peak ambiguity, curvature and confidence rules remain, as do forward/reverse
consistency checks. Their resulting AP decisions can still change because the
fractional patches change. The final motion-fit guard is unchanged.

The copied diagnostic trace reproduces the original trace and composed AP
observations in the focused regression. Every baseline translation also checks
its final field against the unchanged estimator within 1e-10 pixels.

## Phase-sweep results

Aggregate vector RMS false motion over the 25 translations, in pixels:

| Policy | Error |
|---|---:|
| Baseline | 0.058467 |
| Cubic | 0.035241 |
| Refined | 0.053055 |
| Both | 0.003839 |

The combined policy reduces the error by 93.43%. Neither individual change
achieves that improvement. All four produce zero motion on the exact
self-match, and none fails the final field guard or lacks sufficient APs.

The error also grows through the existing multiscale stages. Their dense
field RMS values, before the coherent spline fit, are:

| Policy | Coarsest | Second | Third | Finest |
|---|---:|---:|---:|---:|
| Baseline | 0.00443 | 0.01498 | 0.03361 | 0.04300 |
| Both | 0.00186 | 0.00217 | 0.00258 | 0.00292 |

This locates an avoidable part of the noiseless error in the matching and
multiscale refinement path, before the final field fit.

Integer-shift controls are not exact copies of the reference proxy: filtering
the finite translated detector image uses reflected boundaries. Baseline
errors there are 0.0345–0.0362 pixels, versus 0.00062–0.00100 for the combined
policy. That result should not be described as an exact self-match failure.

The lowest-error non-baseline policy without guard/insufficient-AP fallbacks
is selected only for further validation, not deployment.

### Checking the integer-shift boundary effect

A separate diagnostic generates the known continuous scene for 48 pixels
beyond each image edge, computes the same matching proxy there, then crops
to the original detector area. Both reference and moving proxies get this
oracle boundary extension. AP geometry and field-fitting kernels remain fixed;
only matching templates are rebuilt. This uses unavailable exterior scene
information and is not a proposed padding fix for real captures.

With this extension, all tested integer translations recover zero false
motion to better than 1e-12 pixels in both methods. Thus the integer-shift
anomaly is a synthetic finite-boundary effect. Fractional errors remain:

| True shift | Extended-scene baseline, px | Extended-scene combined, px |
|---|---:|---:|
| (0.25, 0.25) | 0.055173 | 0.004578 |
| (0.5, 0.5) | 0.066985 | 0.003656 |

The fractional improvement survives removal of that boundary artifact. These
controls still do not measure boundary effects or noise levels in Saturn.

## Validation design

Three genuine residual translations are supplied with zero global shift to
check that the new method is not merely suppressing every local displacement.
The stationary, long-motion, short-motion and short-plus-offset cases are then
replayed with 32 frames each at noise sigma 0 and 2 ADU. These are the same
independent continuous scenes, phases and noise realization as the preceding
experiment. Baseline fields are loaded from hashed arrays; regenerated noisy
stacks, noiseless twins and zero-noise stacks must match prior results within
1e-9 ADU. The selected policy is fixed throughout validation.

Every image is sampled with the same final bilinear sampler and exported with
the same brightness mapping. Clean twins separate geometric effects from
source noise. The actual PlanetaryTools implementation applies Wavelet
27/0/0/0 followed by Adaptive Deconvolution 15.6, Contrast Adaptive. No output
filtering or brightness normalization is introduced.

## Validation results and decision

The nonzero residual checks recover genuine motion better rather than simply
returning zero residual everywhere. With the supplied global shift fixed at
zero, baseline versus combined field errors are 0.0672/0.0396 px for a true
(0.3, −0.4) px translation, 0.0686/0.0413 px for (−0.45, 0.2), and
0.1203/0.0856 px for (0.7, 0.65). These whole-object metrics include the existing
support taper, which fades local corrections near unsupported regions.

The 32-frame moving controls give the following vector field RMS errors:

| Case | No noise: baseline / combined, px | Noise sigma 2: baseline / combined, px |
|---|---:|---:|
| Stationary | 0.05652 / 0.00412 | 0.24297 / 0.26786 |
| Long motion | 0.21133 / 0.20641 | 0.31124 / 0.32520 |
| Short motion | 0.75131 / 0.74834 | 0.78828 / 0.79246 |
| Short + offset | 0.92773 / 0.92586 | 0.95684 / 0.95759 |

Clean-twin stack RMSE against exact-motion stacking, in ADU:

| Case | Clean-derived field: baseline / combined | Noisy-derived field: baseline / combined |
|---|---:|---:|
| Stationary | 0.07837 / 0.00312 | 0.22488 / 0.27945 |
| Long motion | 0.07446 / 0.01503 | 0.23555 / 0.27893 |
| Short motion | 0.06851 / 0.02168 | 0.21602 / 0.26642 |
| Short + offset | 0.26508 / 0.26347 | 0.32956 / 0.35765 |

The selected method improves every noiseless case but worsens every noisy
case in both field and clean-twin image error. More precise maximization of
NCC does not ensure a more accurate physical motion estimate under noise.
Both interpolation and peak refinement change together in this validation,
so their individual contributions to the noisy regression are not isolated.

All 20 exports received the exact user sharpening recipe. Sharpened RMSE
against the identically processed noiseless exact-motion target, linear 0–1:

| Case | No noise: baseline / combined | Noise sigma 2: baseline / combined |
|---|---:|---:|
| Stationary | 0.000476 / 0.000085 | 0.008825 / 0.008852 |
| Long motion | 0.000456 / 0.000148 | 0.009967 / 0.010015 |
| Short motion | 0.000546 / 0.000359 | 0.010105 / 0.010139 |
| Short + offset | 0.001772 / 0.001664 | 0.011117 / 0.011258 |

Noisy sharpened error increases by 0.30%, 0.48%, 0.33% and 1.27%, respectively.
No selected-policy fit falls back for insufficient APs or fails the final
geometric guard. The noisy differences are small and no population confidence
interval is claimed; they nevertheless provide no evidence for promotion.

**Decision:** retain this as a diagnostic, not a production replacement. A
real numerical fractional-matching bias has been reduced, but the noiseless
gain has not translated into better noisy stacks. No full Saturn replay or
claim about closing its AutoStakkert gap follows from these controls.

Noise sigma 2 ADU is an experimental setting, not an estimate of Saturn noise.
The same scene and fixed circular PSF are used throughout. The next focused
comparison would separate cubic sampling from direct peak refinement on the
noisy controls and vary noise strength, testing where the regression starts
before another expensive real-capture replay. The existing independent-pixel
validation experiments already addressed a different form of overfitting;
this result should not be presented as a reason to repeat those unchanged.

## Reproduction

```sh
PYTHONPATH=. OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python tools/ap_fractional_phase.py --out out/ap-fractional-phase
PYTHONPATH=. OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python tools/validate_fractional_ap.py
PYTHONPATH=. OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python tools/ap_phase_boundaries.py
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
MKL_NUM_THREADS=4 QT_QPA_PLATFORM=offscreen \
  ../PlanetaryTools/planetary-app/.venv/bin/python tools/planetarytools_sharpen_experiment.py \
  --planetary-tools ../PlanetaryTools/planetary-app \
  --out out/ap-fractional-validation/sharpened out/ap-fractional-validation/*.png
PYTHONPATH=. OPENBLAS_NUM_THREADS=4 .venv/bin/python tools/analyse_fractional_ap.py
MPLCONFIGDIR=/tmp/pr-mpl /usr/bin/python3 tools/plot_fractional_ap.py
PYTHONPATH=. OPENBLAS_NUM_THREADS=4 .venv/bin/python -m pytest -q tests/test_fractional_ap_trace.py
```

Use fresh output directories. System Matplotlib is already installed. The
diagnostic is intentionally not optimized: refinement evaluates many
fractional correlation scores per AP. No production speed claim is made.

Two focused tests pass: integer/affine preservation of the cubic sampler and
exact baseline trace/composition parity. The phase sweep additionally checks
baseline parity on CUDA, and the analyzer recomputes validation metrics from
hashed arrays and verifies the original sharpening recipe. The boundary
controls reproduce unextended CPU/GPU phase results within 1e-10 px and recover
extended integer shifts within 1e-12 px. The chart was visually inspected.

The numerical record is `results/registration/ap-fractional-phase.json`, which
includes the screen, boundary controls, residual-translation checks, noisy and
noiseless validation, source hashes and sharpening identities. Raw outputs and
the chart are in `out/ap-fractional-validation/`; phase and boundary records
are in `out/ap-fractional-phase/`. Production remains unchanged.
