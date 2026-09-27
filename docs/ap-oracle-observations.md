# Separating AP measurements from field reconstruction

Commit `29d7c6f` showed that stronger field constraints reduced reference
sensitivity but worsened the full Saturn stack. This follow-up supplies
known-correct patch-average displacements to the unchanged coherent fitter.
It separates the measurements from acceptance/weighting, spline reconstruction
and the support taper. It does not change production behavior.

## Controlled intervention

Four cases contain 32 frames each: stationary, long motion, short motion and
short motion with a retained offset. The moving cases reuse the independent
continuous-scene generator, seed 9033, sigma-2 ADU added noise and input frames
from the preceding validation. Long motion has wavelengths 160/240 px; short
motion has wavelengths 80/120 px. The GPU replay reproduces the preceding CPU
measured-AP stacks and their noiseless twins within 1e-9 ADU.

Each frame is processed six ways:

1. **Measured:** existing measured and composed AP motion, confidence and gates.
2. **Truth, same gates:** exact truth averages, retaining those confidence and
   acceptance values.
3. **Truth, accepted uniform:** the same accepted APs with unit weights.
4. **Truth, all APs:** all configured APs, unit weights.
5. **Truth, untapered:** the preceding fit with the final taper and geometric
   guard bypassed. This is a diagnostic, not a candidate for deployment.
6. **Oracle:** sample the same frames with the exact dense motion field.

Truth observations are averages of the known residual motion under the exact
fixed reference-gradient kernels used to construct the spline design matrix.
A regression verifies this independently by comparing direct patch averages
with the matrix applied to an arbitrary representable spline field. This is
truth for the fitter's observation model; a translation matcher observing a
deforming patch is not guaranteed to return that particular average. The
measured branch includes both matching and multiscale composition errors.

All branches use the same noisy frames, global translations, equal frame
weights, pixel sampler and export mapping. Noiseless twins pass through each
estimated field. The reference is an independent, noiseless scene with the
same fixed circular PSF. There is no output filter or normalization.

## Motion accuracy

Vector RMSE over the fixed object mask, in pixels:

| Case | Measured | Truth, same gates | Truth, accepted uniform | Truth, all APs | Truth, untapered |
|---|---:|---:|---:|---:|---:|
| Stationary | 0.24297 | <1e-12 | <1e-12 | <1e-12 | <1e-12 |
| Long motion | 0.31124 | 0.20989 | 0.20986 | 0.20834 | 0.21059 |
| Short motion | 0.78828 | 0.71976 | 0.71757 | 0.70144 | 0.70238 |
| Short + offset | 0.95684 | 0.89234 | 0.88949 | 0.86899 | 0.87005 |

The stationary case exposes spurious measured motion, which vanishes when
correct observations are substituted. Substantial errors remain for spatially
varying motion even with correct observations. Confidence values, rejected
APs and the final support taper are not the main source of that residual in
these controls. The result also holds inside the untapered portion of the
object: short-motion error with all true APs is 0.662 px there.

There are 107 configured APs and 143 spline coefficients per motion component.
The measured branches accept 94–107 APs per frame. No guarded branch falls back
for inadequate AP count or fails its geometric guard in these controls.

## Stack and sharpening results

Clean-twin stack RMSE versus the noiseless exact-motion stack, in ADU:

| Case | Measured | Truth, same gates | Truth, all APs |
|---|---:|---:|---:|
| Stationary | 0.22488 | <1e-12 | <1e-12 |
| Long motion | 0.23555 | 0.00149 | 0.00102 |
| Short motion | 0.21602 | 0.00915 | 0.00794 |
| Short + offset | 0.32956 | 0.25268 | 0.24536 |

Correct AP observations almost eliminate the long-motion clean-stack error,
even though per-frame field errors remain. Errors can cancel across frames
and interact differently with image gradients; field error and stack error
must be assessed separately. The persistent offset exposes a substantial
remaining stack bias.

All 28 exports were sharpened by the actual PlanetaryTools implementation using
Wavelet 27/0/0/0 followed by Adaptive Deconvolution 15.6, Contrast Adaptive. The
recipe and implementation hashes match the preceding experiments.

The following compares each sharpened noisy stack against the **same noisy
frames aligned with the exact motion**, so the comparison highlights changes
caused by alignment. RMSE is in linear 0–1 image units:

| Case | Measured | Truth, same gates | Truth, all APs |
|---|---:|---:|---:|
| Stationary | 0.001726 | 0 | 0 |
| Long motion | 0.002128 | 0.000752 | 0.000748 |
| Short motion | 0.002973 | 0.002723 | 0.002678 |
| Short + offset | 0.004220 | 0.003681 | 0.003668 |

This is not total image noise. Absolute sharpened error against the noiseless
target does not improve monotonically with more accurate alignment; measured
alignment even has lower absolute error than exact motion in some cases.
Changing a warp changes the sampling and correlation of noise as well as
detail. Both comparisons are retained in the numerical record rather than
selecting only the metric favorable to an intervention.

## Is the spline grid too coarse?

A separate algebraic diagnostic projects the known dense residual fields onto
the spline basis using all detector pixels. This tests representational
capacity without having to infer motion from AP averages:

| Case | Dense projection error, px | Existing fit with all true APs, px |
|---|---:|---:|
| Long motion | 0.000133 | 0.20834 |
| Short motion | 0.003013 | 0.70144 |
| Short + offset | 0.003773 | 0.86899 |

The grid can represent these motions. The difficulty is recovering them from
the patch summaries with the current regularized fit.

To examine that inverse problem without arbitrary coefficient scaling, QR
factorizations make the spline basis orthonormal in detector-pixel L2 before
forming the patch-average operator. Its numerical rank is 107 of 143, leaving
36 unobserved dimensions. Its retained singular values run from 0.1493 down to
6.15e-14, a condition number of about 2.4e12. Thus some nominally observable
patterns are also extremely weakly constrained.

For the spline-projected truth, an exact noiseless row-space reconstruction
has only 0.010–0.023 px of error, but that favorable round trip removes basis
mismatch by construction. Using the actual continuous-field patch averages
in the full, unregularized pseudoinverse instead amplifies tiny mismatch into
errors of order 1e5–1e6 pixels. Those absurd outputs are an instability
diagnostic, not a proposed reconstruction. Simply removing regularization is
not justified. The null component is invisible to these patch summaries,
not necessarily to the original images; a physical prior may constrain it.

## Conclusion and next test

There are two distinct limitations in these synthetic controls: measured and
composed AP displacements contain errors, and the regularized field fit
suppresses genuine motion even with ideal patch-average inputs. Increasing
spline resolution or removing the support taper is not supported by these
results. They also do not support removing regularization outright.

The next bounded test should replay the measured-AP branch without added
detector noise, retaining these same motion cases. This would distinguish
noise-dependent matching errors from the systematic mismatch between a
translation estimate on a deforming patch and the fitter's scalar
gradient-weighted observation model. That distinction should precede another
full Saturn replay or a new field regularizer.

These are synthetic causal interventions, not a measured explanation of the
Saturn noise gap. The fixed smooth shears, one scene and invariant PSF do not
cover atmospheric seeing. No full real-capture replay was needed because
known truth is unavailable there and no deployable improvement was selected.

## Reproduction and artifacts

```sh
PYTHONPATH=. OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python tools/ap_oracle_observations.py --out out/ap-oracle-observations --device cuda:0
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
MKL_NUM_THREADS=4 QT_QPA_PLATFORM=offscreen \
  ../PlanetaryTools/planetary-app/.venv/bin/python tools/planetarytools_sharpen_experiment.py \
  --planetary-tools ../PlanetaryTools/planetary-app \
  --out out/ap-oracle-observations/sharpened out/ap-oracle-observations/*.png
PYTHONPATH=. OPENBLAS_NUM_THREADS=4 .venv/bin/python tools/analyse_ap_oracle_observations.py
PYTHONPATH=. OPENBLAS_NUM_THREADS=4 .venv/bin/python tools/ap_oracle_identifiability.py
MPLCONFIGDIR=/tmp/pr-mpl /usr/bin/python3 tools/plot_ap_oracle_observations.py
PYTHONPATH=. OPENBLAS_NUM_THREADS=4 .venv/bin/python -m pytest -q tests/test_ap_oracle_observations.py
```

The plotting command uses the already-installed system Matplotlib. No packages
were installed. The focused regression passes; it covers patch/design
equivalence, coordinate ordering, translation offsets, zero-residual recovery,
orthogonal-basis energy preservation and null-space invisibility.

Numerical records: `results/registration/ap-oracle-observations.json` and
`results/registration/ap-oracle-identifiability.json`. Raw field arrays, noisy
and noiseless stacks, exact sharpening stages and the comparison chart are in
`out/ap-oracle-observations/`. The analyzer checks hashes, frame counts and
recomputes image and field metrics directly from saved arrays.
