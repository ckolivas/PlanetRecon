# Field constraints under small reference changes

The previous reference-intervention experiment found that a 0.25 ADU RMS
template change could move the coherent field by roughly half a pixel while
changing only a handful of accepted APs. This experiment changes the field fit
after AP measurement, keeping the observations and confidence values identical
between candidates. It does not alter the matcher thresholds, captured pixels,
global translations, scalar frame weights or output sampling.

## Fixed candidates and comparisons

Eight policies are specified before running the screen:

| Policy | Spline spacing | Bending penalty | Coefficient-amplitude penalty |
|---|---:|---:|---:|
| Baseline | 16 | 0.01 | 0.000001 |
| bend_01 | 16 | 0.1 | 0.000001 |
| bend_1 | 16 | 1 | 0.000001 |
| bend_10 | 16 | 10 | 0.000001 |
| anchor_001 | 16 | 0.1 | 0.001 |
| anchor_01 | 16 | 0.1 | 0.01 |
| grid32 | 32 | 0.1 | 0.000001 |
| grid64 | 64 | 0.1 | 0.000001 |

The existing spacing-dependent bending-energy scale is retained. The added
amplitude penalties pull residual motion towards the fixed global solution;
they are numerical regularizers, not calibrated atmospheric priors. Unlike
bending alone, they also penalize the affine component, which can bias genuine
offset motion. All candidates keep the baseline's spatial support taper and
the same robust reweighting and geometric safety limits.

One trace produces the unfaded AP displacements, their composition with
preceding multiscale warps and their confidence values. Every policy fits those
same measurements. Patch-average design matrices change only when spline
spacing changes. This avoids conflating regularization with a different set
of measured displacements or a different support boundary.

Three known-motion controls use 32 frames each, independent continuous-scene
pixel integrals, circular PSF sigma 1 and independent sigma-2 ADU noise. The
first two exactly reproduce the preceding stationary and moving controls.
The third retains a real offset component in the local field by supplying only
the known camera translation as its global shift. A candidate cannot qualify
merely by suppressing all residual motion. Noiseless twins go through the same
estimated fields to isolate geometric error.

Sensitivity is evaluated on the same sixteen predetermined Saturn positions
as the prior decision trace. Each sees the original reference and the fine and
broad probes with both signs at 0.25 ADU RMS. Sensitivity is vector RMS field
change over the same fixed object mask. Pooling both signs gives 32 values per
spatial scale and policy.

## Screening rule

For each of the three known-motion cases, a candidate must have:

- Clean-twin image RMSE no greater than baseline times 1.05 plus 0.002 ADU.
- Vector field RMSE no greater than baseline times 1.05 plus 0.005 pixels.
- No increase in rejected/insufficient-data control fits.

It must also halve median reference sensitivity at both probe scales. Among
qualifying candidates, the one with the lowest worst-scale sensitivity ratio
is selected for follow-up. These are explicit exploratory engineering gates,
not statistical significance thresholds or proof of improved real resolution.
The selection code and policies are hashed with the report.

Baseline field parity is checked against the unchanged coherent estimator on
CPU and CUDA. Every policy is checked for finite fields and preservation of
the supplied AP observations/confidences. The stationary and moving control
stacks and their clean twins must reproduce the previous experiment to
1e-10 ADU. The selection test rejects an inaccurate or fallback-only solution
even if its reference sensitivity is arbitrarily small.

## Reproduction

```sh
PYTHONPATH=. OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python tools/ap_stability_screen.py --out out/ap-stability-screen
PYTHONPATH=. OPENBLAS_NUM_THREADS=4 .venv/bin/python -m pytest -q \
  tests/test_regularized_ap_fit.py
```

Use a fresh output directory. The run uses CUDA and four CPU threads, hashes
the full SER before and after the real-data part, and leaves production
stacking unchanged. Apply the actual PlanetaryTools Wavelet 27/0/0/0 followed
by Adaptive Deconvolution 15.6, Contrast Adaptive recipe to every exported
control before inspecting the final outcomes.

## Screen outcome

Both amplitude-anchored candidates passed the predefined screen. `anchor_01`
had the lowest worst-scale sensitivity ratio and was frozen for follow-up.
Its median response to the fine reference probe fell from 0.551 to 0.083 px
(85% reduction), and its broad-probe response from 0.543 to 0.140 px (74%).
Field RMSE improved in all three screening controls:

| Control | Baseline field RMSE, px | anchor_01, px |
|---|---:|---:|
| Stationary | 0.252 | 0.100 |
| Moving | 0.325 | 0.248 |
| Moving with retained offset | 0.336 | 0.271 |

Stronger bending alone and coarser grids could reduce sensitivity but failed
one or more accuracy gates. The selected candidate's exact sharpened image
error was slightly worse in every screening control. Sharpened error was not
part of the selection rule; this screen establishes improved stability and
particular geometric metrics, not final image superiority.

## Independent noise and shorter motion

The selected policy was tested without retuning on 32 fresh-noise frames per
case (seed 9033 rather than the screen's 9032). Motion wavelengths were
160/240 px in the first case and 80/120 px in the other two. In the offset case,
only camera translation was supplied as the global alignment. A mean component
of the real spatial motion therefore still had to be recovered locally.

The generator evaluates continuous Gaussian scenes with a fixed circular PSF
and integrates over detector pixels. Four- versus eight-node quadrature differed
by at most 2.22e-8 ADU. Tests separately verify exact stationary pixel integrals
and equivalence to the preceding long-motion generator.

| Fresh control | Field RMSE baseline / candidate, px | Clean-twin RMSE baseline / candidate, ADU | Sharpened RMSE baseline / candidate, linear 0–1 |
|---|---:|---:|---:|
| Long motion | 0.311 / 0.249 | 0.236 / 0.101 | 0.009967 / 0.010587 |
| Short motion | 0.788 / 0.878 | 0.216 / 0.082 | 0.010105 / 0.011762 |
| Short motion with offset | 0.957 / 1.111 | 0.330 / 0.448 | 0.011117 / 0.012242 |

The anchor suppresses some genuine shorter-scale motion. Lower clean-twin
image error can coexist with higher field error because the two metrics weight
spatial structure differently. No single one of these metrics is sufficient.
The short-offset case is worse in all three measures. None of these fits
failed the geometric guard or fell back for insufficient AP constraints.

## Full 5,738-frame Saturn result

All 5,738 selected frames were replayed in both variants, with exactly the same
AP observations, scalar weights, reference, global shifts and final bilinear
sampler. The capture was rehashed, each frame was associated with its original
selection position, and all 45 nonoverlapping shards were required. The baseline
reproduced the previous coherent stack within 1.99e-13 ADU; its sharpened PNG
pixels were identical. Both new full stacks and their alternating half-stacks
were processed by the actual PlanetaryTools implementation: Wavelet 27/0/0/0,
then Adaptive Deconvolution 15.6, Contrast Adaptive.

| Output | Fine variation upper / lower, % | Ring transition left / right, px |
|---|---:|---:|
| Coherent baseline | 2.627 / 2.738 | 7.396 / 6.047 |
| anchor_01 | 2.993 / 2.981 | 7.457 / 6.250 |
| Production PR context | 2.967 / 2.975 | 8.142 / 6.364 |
| AutoStakkert manual-64 context | 0.573 / 0.642 | 7.275 / 5.651 |

The candidate increases fine variation by 13.95% in the upper patch and 8.89%
in the lower patch. Ring transitions are slightly wider. Across 75 nearby
paired windows per side, the median widening is 0.204 px left and 0.068 px
right; the candidate is wider in 100% and 89.3% of the windows respectively.
These overlapping windows are a robustness check, not independent trials.
Visual review agrees that the grain remains and the AS gap is not closed.

The candidate eliminates the baseline's 254 geometric-guard fallbacks to
global alignment, but this does not rescue image quality. Raw half-difference/2
variation changes from 0.0748% to 0.0815% in the upper patch and from 0.0810%
to 0.0765% in the lower patch, a mixed result. Shared reference and estimator
biases cancel from this difference, so it is not a measure of all grain.

Fine variation includes real detail and artifacts, and ring widths are not
calibrated resolution. AS uses its own selection and reference and is included
only as context. There is no added pixel filtering or brightness normalization.

**Decision:** do not promote this regularization policy. Reducing sensitivity
to small template changes is useful diagnostically but has not delivered a
better stack. The next useful distinction is between inaccurate AP measurements
and the field fitted to them. A controlled known-motion replay that substitutes
truth-derived patch-average observations for measured AP shifts can isolate
those stages without another arbitrary smoothing sweep. Real-frame temporal
or bidirectional consistency can then be evaluated only after that separation.

Numerical records are `results/registration/ap-stability-screen.json` and
`results/registration/ap-stability-replay.json`; the latter includes the fresh
controls, full replay, input/source hashes and exact sharpening identities.
The visual comparison is `out/saturn-regularized-ap/comparison.png`.

## Follow-up reproduction

```sh
PYTHONPATH=. OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python tools/ap_stability_validation.py --out out/ap-stability-validation
PYTHONPATH=. OPENBLAS_NUM_THREADS=4 .venv/bin/python tools/regularized_ap_replay.py \
  --out out/saturn-regularized-ap --prepare
# Run each worker once, concurrently if desired; every worker uses four CPU threads.
PYTHONPATH=. OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python tools/regularized_ap_replay.py --out out/saturn-regularized-ap --worker 0 --workers 4
# Repeat the preceding command with --worker 1, 2 and 3, keeping --workers 4.
PYTHONPATH=. OPENBLAS_NUM_THREADS=4 .venv/bin/python tools/regularized_ap_replay.py \
  --out out/saturn-regularized-ap --combine
```

Apply the exact sharpening script to all PNGs in each of the screen, validation
and replay output directories, using a fresh `sharpened` subdirectory:

```sh
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
MKL_NUM_THREADS=4 QT_QPA_PLATFORM=offscreen \
  ../PlanetaryTools/planetary-app/.venv/bin/python tools/planetarytools_sharpen_experiment.py \
  --planetary-tools ../PlanetaryTools/planetary-app \
  --out out/saturn-regularized-ap/sharpened out/saturn-regularized-ap/*.png
PYTHONPATH=. OPENBLAS_NUM_THREADS=4 .venv/bin/python tools/analyse_ap_stability_screen.py
PYTHONPATH=. OPENBLAS_NUM_THREADS=4 QT_QPA_PLATFORM=offscreen \
  .venv/bin/python tools/analyse_regularized_ap.py
```

Four focused tests pass, including actual CPU and CUDA baseline parity and the
independent generator checks. Production code and defaults remain unchanged.
