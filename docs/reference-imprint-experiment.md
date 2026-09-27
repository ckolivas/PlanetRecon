# Paired reference-only texture interventions

The preceding experiments demonstrated false motion from PSF mismatch. This
test asks a different question: can a shared reference cause frames with
different seeing/noise to acquire the same artificial structure in the stack?
It measures sensitivity directly, without assuming that the native reference
texture is sensor noise or that it causes the AutoStakkert discrepancy.

## Method

Two seeded random probe patterns are differences of Gaussians, with sigmas
1/3 pixels and 4/12 pixels respectively. They are tapered to the object region,
centred and scaled to unit RMS in a fixed object mask. For each pattern, the
matching reference is changed by plus or minus 0.25 and 1 ADU. The unmodified
reference is a ninth branch. These are artificial interventions, not an
estimate of the camera noise or an assertion that the reference contains that
much grain.

All nine branches use identical observed frames, weights, global translations
and bilinear output sampling. Only the matching reference proxies, centred
patch templates and template strengths change. AP positions, eligibility,
support and, for coherent fitting, its original gradient-response kernels and
spline design stay fixed. Forward/reverse matching and confidence decisions
respond normally to the changed templates. This isolates local matching
sensitivity; it does not test changes to global registration or AP placement.

Both the production circular matcher and the experimental coherent AP
estimator are exercised. Each uses two 32-frame independent continuous-scene
controls, stationary and moving, with a known circular PSF, known fractional
translations and independent noise of sigma 2 ADU. Raw observations are area
integrals rather than samples made by the fitter. Noiseless twins pass through
the same estimated fields to separate geometric error from noise carried by
the selected warp.

The capture replay uses 512 evenly spaced positions from the established
5,738-frame Saturn selection. It saves sums and support for alternating
256-frame halves and diagnostics after 64, 128, 256 and 512 frames. The prefixes
also span progressively more capture time, so this is not a pure sample-count
scaling experiment. Independent halves check repeatability of the response;
they do not establish independent atmospheric or detector noise.

## Measurements

For an injected reference pattern `p` and dose `d`, the paired output response
is `(stack_plus - stack_minus)/(2*d)`. Its units are output ADU per injected
reference ADU. The projected gain onto `p` measures the component shaped like
the injected pattern. Response RMS also includes unrelated-looking geometric
changes. Correlation must therefore accompany any claim of copied texture.

The even component `(stack_plus + stack_minus)/2 - stack_base` measures
nonlinearity. Agreement between doses checks whether a local linear-response
description is adequate. A least-squares projection onto the baseline image's
two spatial gradients describes translation-like sensitivity; this is an
analysis only and never modifies stack pixels. Interior disc-patch high-pass
statistics additionally separate fine-scale response from strong limb/ring
gradients. Neither analysis operation is an added output filter.

Every exported image is sharpened with the actual PlanetaryTools Wavelet
27/0/0/0 and Adaptive Deconvolution 15.6, Contrast Adaptive recipe. The sharpened
paired response is nonlinear and must not be interpreted as a calibrated
transfer function. The 512-frame branches are compared with one another, not
with a full-count AutoStakkert stack.

## Completed results

Both methods completed all nine branches on the same 512 capture frames and
both 32-frame controls. The projected reference-pattern gains on Saturn are:

| Matcher | Fine, dose 0.25 | Fine, dose 1 | Broad, dose 0.25 | Broad, dose 1 |
|---|---:|---:|---:|---:|
| Production | 0.02002 | 0.02247 | 0.04286 | 0.03803 |
| Coherent AP | 0.01824 | 0.01880 | 0.03487 | 0.02752 |

These are output ADU per injected reference ADU in the component shaped like
the probe. They are **not percentages of existing image noise**. Both
interleaved halves retain positive projected gains. For example, production's
fine dose-1 gains are 0.02270 and 0.02224 in the two halves; its broad dose-1
gains are 0.03809 and 0.03797. The production fine gain is 0.02065 at 64 frames
and 0.02247 at 512, while broad gain is 0.03603 and 0.03803. A small response
shaped like the reference probe survives increasing the number of frames.

The entire response is much less stable than that projected component. With
the fine dose-1 probe, full-response RMS is 0.1011 ADU/ADU for production and
0.1708 for coherent AP fitting. Correlations with the probe are only 0.222 and
0.110. For the broad dose-1 probe they are 0.184 and 0.079. Thus most of the
output change should not be described as a copy of the reference texture.

The coherent dose-0.25 fine response has RMS 0.4846 ADU/ADU, much larger than
the 0.1708 value at dose 1. Its two half-response maps correlate at -0.016 for
dose 0.25 and 0.055 for dose 1, despite positive projected gains in both halves.
Its dose-0.25/dose-1 maps correlate only 0.177. The corresponding broad maps
agree better (dose correlation 0.609; half correlation 0.644 at dose 1).
The even response is also substantial: 0.3454 ADU for coherent fine dose 1,
versus 0.0814 for production. Treating these changes as a single linear
texture-copying process would be misleading.

Projection onto a uniform translation accounts for only 8.9% and 6.5% of
fine dose-1 response energy for production and coherent fitting respectively.
For broad dose 1 those fractions are 4.6% and 0.17%. The remaining sensitivity
is not explained by one small shift of the whole image. No image was adjusted
using these descriptive translation fits.

The synthetic controls reinforce the distinction between reference texture
and alignment behaviour. Both signs of added fine texture can *reduce*
clean-twin error, while broad texture substantially increases it. For example,
the stationary coherent baseline has 0.2382 ADU clean-twin RMSE; fine dose-1
plus/minus gives 0.1492/0.1801, whereas broad dose-1 gives 0.6068/0.5587.
This is not evidence that adding noise is useful: it shows that template
perturbations change the matcher decisions and geometric errors, in addition
to any component resembling the injected texture.

## Decision trace on sixteen predetermined frames

`tools/reference_imprint_gates.py` repeats all branches on sixteen positions
spread through the 512-frame selection. The detailed production trace matches
the actual production field to 1e-10 pixels in every branch. Both estimators
start with 396 eligible APs across scales; 45–80 have positive confidence on
these baseline frames (median 71.5).
After also applying the multiscale stage guards, the coherent fitter uses
39–80 AP measurements, median 70.5. Its spacing-16 field has 1,363 spline
coefficients per axis on this detector, so the bending penalty is an essential
constraint. This dimensional comparison motivates a regularization-stability
test; it does not by itself establish the cause of the measured sensitivity.

| Fine probe | Median changed active APs | Production field change, median/max pixels | Coherent field change, median/max pixels |
|---|---:|---:|---:|
| +0.25 ADU RMS | 5 | 0.123 / 0.412 | 0.601 / 1.327 |
| -0.25 ADU RMS | 4 | 0.133 / 0.255 | 0.351 / 0.978 |
| +1 ADU RMS | 17 | 0.311 / 0.491 | 1.130 / 1.405 |
| -1 ADU RMS | 16 | 0.302 / 0.468 | 0.993 / 1.400 |

Neither multiscale-stage guards nor the final coherent spline guard change in
the fine dose-0.25 comparisons. Hence the large coherent-field sensitivity in
those comparisons cannot be attributed to a final guard switching an entire
frame to global alignment. AP acceptance, AP displacement/confidence and their
subsequent field fit change together; this trace does not isolate their
individual causal contributions. At larger/broader perturbations, multiscale
stage guards sometimes change too. Only one of the sixteen broad-minus-1
comparisons switches the final spline guard.

**Decision:** reference texture can induce a small shared component, but these
tests do not identify the native reference grain or explain the full AS gap.
They reveal substantial nonlinear local-alignment sensitivity, especially in
the experimental coherent fit. The next useful target is stability of AP
measurements and their fitted field under small template changes, with
known-motion accuracy retained as a separate requirement. No reference
denoising, production default change or output filtering is justified here.

The complete tracked record is
[`reference-imprint.json`](../results/registration/reference-imprint.json).
`out/reference-imprint-response.png` displays the interventions and paired
response maps with explicit scales; these are differences, not candidate
planet images. All 58 exported control/capture PNGs received the exact user
sharpening recipe, and the response-map gallery was visually inspected.

## Reproduction and validation

```sh
PYTHONPATH=. OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python tools/reference_imprint_probe.py \
  --method production --out out/reference-imprint-control-production
PYTHONPATH=. OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python tools/reference_imprint_probe.py --capture \
  --method production --out out/reference-imprint-saturn-production
```

Repeat with method `coherent` and corresponding fresh directories. CUDA is used
with four CPU threads per process. Sharpen every PNG in the four directories
using `tools/planetarytools_sharpen_experiment.py` in the PlanetaryTools Python
environment, then run `tools/analyse_reference_imprint.py`.

Before analysis, also run the decision trace into a fresh file:

```sh
PYTHONPATH=. OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python tools/reference_imprint_gates.py --out out/reference-imprint-gates.json
```

The tests verify deterministic centred/unit-RMS probes, known linear and even
responses, exact restoration of the original field after changing templates,
fixed geometry and spline design, and detection of a known translation tangent.
The capture is hashed before/after replay and during analysis. The analyzer
verifies sources, complete selections, probe identity, stored accumulators,
snapshot equality, regenerated response statistics and exact sharpening hashes.
All original observations and production settings remain unchanged.
Four focused tests pass, including the independent translation-component check.
