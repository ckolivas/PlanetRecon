# Correcting the experimental joint fitter's forward sampler

The preceding joint-fit experiment invented local motion on noise-free frames
whose only true motion was a known fractional translation. The generator used
prefiltered cubic B-splines; the predictor used PyTorch bicubic interpolation.
They are different continuous reconstructions of the same sampled image.

This experiment corrects that mismatch and tests the correction against an
independent continuous scene. It changes diagnostic tools only. It does not
establish that this was the cause of the production stack's noise difference
from AutoStakkert.

## Implementation

`SplineJointRegistration` inherits the previous joint fitter unchanged except
for its forward reference representation and sampler. Each reference blur-bank
image is converted once to cubic B-spline coefficients with SciPy's reflect
boundary condition. A differentiable tensor implementation evaluates the
sixteen neighbouring coefficients using cubic B-spline weights. Reflecting
coefficient indices implements the same boundary condition as the prefilter.

The coefficient conversion is not output smoothing: evaluation at integer
coordinates reconstructs the reference samples. Both coordinate derivatives
and derivatives through adjacent blur-bank mixtures are tested. The original
`JointRegistration` source is retained so the previous results remain
reproducible.

The 32-pixel field spacing, 100-iteration budget, quarter-pixel training mask,
held-out prediction rule, photometric nuisance parameters and motion guards
are unchanged. The established controls test penalties 0.03 and 0.3; the
independent probes use 0.3. Original observed pixels and the final bilinear
stack sampler are unchanged. There is no per-frame normalization, stack
normalization or added output filter.

## Translation controls

For the two established scenes, three fractional translations and blur sigma
0 or 1, the supplied global translation is exact and true local motion is zero.

- All 12 cases without added noise now recover exactly zero local motion.
- The validation rule rejects local motion in all 12 noisy cases. Before this
  correction it accepted all 12 noise-free cases and 7 of the 12 noisy cases.
- The ungated noisy fit still invents motion, with RMS 0.028–0.141 pixels. The
  zero-motion result for noisy frames is supplied by validation, not by fitting
  alone.

This verifies consistency with the established SciPy generator. Consistency
alone does not establish a physically adequate sampling model.

## Independent detector model

Two independent continuous scenes comprise a broad Gaussian object and 60
Gaussian features with fixed amplitudes and positions. The resolved scene's
feature widths are 1.5–3 pixels; the fine scene's widths are 0.65–1.2 pixels.
These are illustrative scene widths, not a calibration of the capture's PSF
or its focal-ratio/pixel-size multiplier.

The translation probe evaluates exact unit-square detector integrals using
error functions. Continuous Gaussian blur broadens each component while
preserving its integrated flux. The reference has no added noise. Observed
frames are computed directly from the continuous functions, without sampling
the reference array or calling either fitter's interpolator.

For noise-free resolved features, false-motion RMS falls from 0.038–0.057 pixels
with the old predictor to at most 0.000239 pixels with the spline predictor.
The validation rule still accepts all six of these very small residuals: its
numerical tie tolerance is not a physical accuracy threshold or significance
test.

For fine features without additional blur, the corrected ungated fit has
0.006–0.019 pixels RMS of false motion, but validation rejects all three cases.
With additional blur, it accepts all three, with at most 0.000675 pixels RMS.
Both predictors reject motion in all 12 independent translation cases with
two-ADU added noise. One fixed noise realization is shared across poses in this
probe; these are not 12 independent statistical trials.

The result supports the corrected interpolator for these sampled scenes. It
does not make information above the detector's sampling limit recoverable
from a single reference, or prove that interpolation error is absent.

## Established noisy stack controls

The stronger corrected fit reduces sharpened-image error relative to the
previous stronger joint fit in all eight established scene/case combinations
at seed 9017. For the four moving controls:

| Scene and motion | Coherent AP baseline | Previous joint 0.3 | Corrected joint 0.3 |
|---|---:|---:|---:|
| Saturn, symmetric | 0.021594 | 0.021829 | 0.021257 |
| Saturn, offset | 0.022076 | 0.021248 | 0.020749 |
| Texture, symmetric | 0.016426 | 0.017319 | 0.016793 |
| Texture, offset | 0.014125 | 0.014694 | 0.014233 |

These are RMS errors against identically sharpened, known-motion clean stacks,
in linear 0–1 units. Improvement over the previous joint fit is 2.3–3.1% in
these moving cases. The texture cases still regress against the coherent AP
baseline. The Saturn stationary changing-blur case also remains worse than
that baseline: corrected 0.031918 versus 0.028820.

The corrected ungated fit's clean-stack error gets worse in the four stationary
controls. Validation rejects all 192 stationary frames, yielding zero local
motion, and accepts all 192 moving frames. Neither predictor is a uniformly
better estimator across every metric. The unchanged coherent baseline is
reproduced to at most 1.38e-13 ADU, and the clean known-motion target images are
identical to those in the preceding experiment.

All 768 corrected joint fits in these controls reach the 100-iteration limit;
none formally converges, and none triggers the geometric fallback guard.
The two independent moving scenes below have one successful optimizer
termination among 96 fits and no geometric guard failures. These remain
fixed-budget estimators. The previous longer-budget probe was for the bicubic
model, so its convergence conclusions cannot be assumed for this correction.

## Independent moving scenes

Each independent scene also supplies 24 moving frames with composed smooth
shears, changing continuous blur, and independent two-ADU Gaussian noise.
Gaussian convolution precedes the shear; detector integration follows it.
Four-point Gauss-Legendre integration per axis computes each detector pixel.
An eight-point calculation checks every pose: maximum differences are
1.13e-7 ADU for resolved features and 4.05e-5 ADU for fine features.

Motion is scored against the analytic displacement field. Clean copies of
each frame are warped using the motion inferred from noisy frames, allowing
alignment error to be separated from propagated noise. The image target is
the clean stack sampled with the known motion and the same final bilinear
sampler. It is not an assertion of perfect reconstruction of the continuous
scene after detector integration.

| Scene | Clean-stack RMS, old / corrected (ADU) | Field RMS, old / corrected (pixels) | Sharpened RMS, old / corrected |
|---|---:|---:|---:|
| Resolved | 0.016991 / 0.016084 | 0.34342 / 0.34695 | 0.009789 / 0.009644 |
| Fine | 0.024102 / 0.018902 | 0.39134 / 0.40882 | 0.005117 / 0.005089 |

All 48 genuine-motion cases pass validation with both predictors. Clean-stack
error improves by 5.3% and 21.6%, while per-frame field error gets slightly
worse. Errors in weakly textured areas and temporal cancellation affect these
metrics differently. The exact sharpened-image improvement is only 1.5% and
0.6%. One sequence per scene is insufficient to establish that these small
differences generalize to other noise realizations.

Propagated-noise RMS falls from 0.2822 to 0.2778 ADU for resolved features, and
0.2808 to 0.2775 ADU for fine features. The known-motion stacks have 0.2840 and
0.2839 ADU respectively. Lower noise alone does not prove better alignment:
different sampling phases change interpolation attenuation. These synthetic
noise amplitudes are not an estimate of the real sensor noise.

## Decision and remaining limits

The correction removes the specific sampler mismatch and greatly reduces
false motion in the independent resolved-scene translation tests. It is a
sound replacement for the predictor in further joint-fit experiments, with
the measured residuals and validation limitations above. It is not a proven
solution to the original stack-noise problem.

The small sharpened gains do not explain the much larger real-capture gap to
AutoStakkert. No full 5,738-frame capture replay or production change is made
here. A real-capture comparison can now use the corrected predictor, but must
retain the same frame selection, reference, output sampling and sharpening.
The texture regressions, limited blur family, noisy fixed reference in the
Saturn-derived controls, iteration limits and single noise seed remain
important constraints on interpretation. A held-out prediction improvement
alone still does not establish true geometric motion.

## Reproduction and artifacts

```sh
PYTHONPATH=. OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python tools/joint_translation_probe.py \
  --sampler spline --out out/joint-translation-spline-probe.json

PYTHONPATH=. OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python tools/analytic_sampling_probe.py \
  --out out/joint-analytic-sampling-probe.json

PYTHONPATH=. OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python tools/analytic_motion_probe.py \
  --out out/joint-analytic-motion

PYTHONPATH=. OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python tools/joint_controls.py --sampler spline \
  --scenes saturn --out out/joint-spline-saturn-controls
```

Use `--scenes texture` and a fresh output directory for the second established
scene. The established controls use the same four cases, 48 frames and seed
9017 as the previous joint experiment, for 384 frames in total. The independent
moving scenes add 48 frames and the two translation probes add 48.

Apply `tools/planetarytools_sharpen_experiment.py` to the generated PNGs using
the PlanetaryTools environment. The recipe is Wavelet 27/0/0/0 followed by
Adaptive Deconvolution 15.6 with Contrast Adaptive enabled. Run
`tools/analyse_joint_controls.py` on both established-control directories,
then `tools/collect_spline_joint_results.py` to verify source hashes, complete
case coverage, unchanged coherent baselines, common truth images, sharpening
settings and exported image hashes.

`tools/render_spline_comparison.py` makes
`out/joint-spline-comparison.png`. The established scenes use common native
pixel crops; the smaller independent scenes are enlarged 2x with nearest
neighbour display sampling. The target has no added noise; the Saturn-derived
synthetic reference still contains its original structure.

Numerical results are recorded in
[`results/registration/spline-forward-sampling.json`](../results/registration/spline-forward-sampling.json).

The focused suite passes 33 CPU/CUDA tests, including spline/SciPy parity at
fractional coordinates and reflected boundaries, analytic-gradient checks,
zero-motion translation fits, preservation of observations, independent
detector-integral verification and continuous-blur flux conservation.

The subsequent [full 5,738-frame Saturn replay](joint-spline-saturn-replay.md)
is a negative real-data result: the corrected joint estimator increases
sharpened variation and slightly widens ring transitions relative to coherent
AP fitting. It remains experimental despite the sampling correction passing
the controlled-scene checks.
