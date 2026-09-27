# Joint blur and motion: controlled evaluation

The subsequent [forward-sampling correction](spline-forward-sampling.md)
retains this model for comparison and tests a corrected spline predictor
against both these controls and independently integrated continuous scenes.

The preceding sequential blur-matching experiment improved several synthetic
geometry tests but worsened the full Saturn stack. This experiment fits blur
and motion in one raw-pixel forward model. It is diagnostic code; production
stacking, frame brightness and output filtering are unchanged.

## Model and validation

At each sensor coordinate, predict the reference through an inverse motion
field and one additional isotropic blur value. The inverse field is a tensor
product cubic B-spline with 32-pixel spacing. Its two displacement coefficient
arrays and blur variance are optimized together. Positive gain from 0.5 to 2
and an offset are profiled analytically for the fitting score only.

The reference bank has Gaussian sigma 0 through 2 in steps of 0.25; adjacent
templates interpolate linearly in variance. This is a continuous mixture of
two Gaussian images, not an exact Gaussian at every intermediate sigma. No
observed image is blurred or resampled for the objective. The original raw
frame is sampled once when forming each experimental stack.

Only even-row/even-column detector samples train the model. The other three
quarters assess prediction. A fixed reference support mask excludes blank
space and detector borders. The two tested bending-penalty strengths are 0.03
and 0.3, with a small coefficient ridge of 1e-4. Flat image regions provide
little displacement information, so this penalty constrains their motion.
It expresses a prior; it is not a calibrated posterior uncertainty estimate.

Every fit starts from zero residual motion and the best global-pose blur model.
The global alternative has exactly the same continuous blur-mixture and
gain/offset freedom as the moving model. Its mixture optimum is obtained from
the interval endpoints and stationary points of the free-gain and bounded-gain
branches. All parameter choices use training samples only.

The optimizer is bounded L-BFGS-B: coefficient components within six pixels,
blur variance within 0–4, at most 100 iterations, relative function tolerance
1e-9 and gradient tolerance 1e-5. It is generally a fixed-budget fit, not a
converged maximum-likelihood solution. The model uses input ADU units; the
penalty strengths have not been calibrated for other bit depths or noise levels.

For stacking, invert the fitted map with 30 fixed-point iterations. Require
inversion error at most 0.02 pixels, minimum Jacobian 0.25 and displacement
magnitude at most six pixels. A fixed spatial taper suppresses extrapolation
outside reference support. A failed geometric check falls back to global
translation and retains the frame.

Four methods are compared:

- The existing coherent AP estimator, unchanged.
- Joint fit with bending penalty 0.03.
- Joint fit with bending penalty 0.3.
- The 0.3 fit applied only if its prediction on unused samples improves over
  the global alternative; otherwise global translation is used.

The last policy uses an improvement threshold of 1e-9 times the larger of one
and the global mean squared error to exclude numerical ties. This is not a
statistical significance threshold. The validation samples do not refit blur,
gain or offset. The final field is still fitted from only one quarter of the
detector samples; no full-data refit is silently substituted.

## Experimental controls

The same two scenes, four motion/blur cases, 48 frames per case and two
independent noise seeds are used as before: 768 frames. Each method also warps
the clean counterpart of every noisy input, separating geometry error from
the noise propagated through the chosen warp. Global shifts, weights and
synthetic motion are identical to the earlier experiments.

Gaussian noise of two ADU is illustrative, not a measured noise model for the
capture. The fixed Saturn reference, including any residual structure in it,
serves as synthetic truth. The textured scene supplies an independent kind of
image structure. The motion is composed smooth shears with an analytic inverse.

Sharpening uses the actual PlanetaryTools implementation: Wavelet 27/0/0/0,
then Adaptive Deconvolution 15.6 with Contrast Adaptive enabled. Each output is
compared with the same recipe applied to its noiseless known-motion oracle.
Those errors include detail, geometry and noise, and cannot be interpreted as
pure sensor-noise measurements.

## Controlled stack results

The validation policy rejects all 384 stationary noisy control frames and
accepts all 384 moving control frames. No geometric guard fallback occurs.
All 1,536 joint fits (two penalties per frame) reach their 100-iteration budget;
none is reported as formally converged. The coherent baseline reproduces the
earlier stacks to a maximum difference of 1.40e-13 ADU.

Clean-stack RMS error in input ADU, measured by warping the clean counterparts
using motion inferred from noisy frames:

| Saturn case | Seed | Coherent | Joint 0.03 | Joint 0.3 |
|---|---:|---:|---:|---:|
| Symmetric motion and blur | 9017 | 0.241927 | 0.024651 | 0.023373 |
| Symmetric motion and blur | 9018 | 0.253322 | 0.021819 | 0.019539 |
| Offset motion and blur | 9017 | 0.262709 | 0.030362 | 0.047414 |
| Offset motion and blur | 9018 | 0.294940 | 0.028955 | 0.045197 |

The stronger fit reduces symmetric-motion clean-stack error by 90–92%, and
offset-motion error by 82–85%. These are stack-image errors, not equivalent
improvements in each frame's displacement field. For the primary Saturn motion
case, vector displacement RMS falls from 0.46772 to 0.25135 pixels (46%). For
the primary texture case it changes from 0.05399 to 0.05431 pixels, despite a
clean-stack error reduction from 0.095888 to 0.023409 ADU. Temporal cancellation
and the location of errors matter; a flat patch can have a displacement error
without a large image-intensity error.

The stronger fit's exact sharpened Saturn errors in linear 0–1 image units are:

| Case | Seed | Coherent | Joint 0.3 |
|---|---:|---:|---:|
| Symmetric motion and blur | 9017 | 0.021594 | 0.021829 |
| Symmetric motion and blur | 9018 | 0.021685 | 0.021973 |
| Offset motion and blur | 9017 | 0.022076 | 0.021248 |
| Offset motion and blur | 9018 | 0.022237 | 0.021192 |

Thus the large clean-stack improvement does not yield a consistent sharpened
improvement: symmetric-motion error rises roughly 1%, while offset-motion error
falls roughly 4–5%. In the primary moving Saturn case, the propagated noise
component rises from 0.19638 to 0.20124 ADU RMS. It is separated exactly using
the clean/noisy twins, and is not a measurement available on the real capture.

Across all 16 scene/case/seed combinations, sharpened error improves in 7 for
the weaker penalty, 5 for the stronger penalty, and 4 for the validated stronger
fit. The textured scene's moving cases regress after sharpening at both seeds,
despite their lower clean-stack errors. The stationary validated outputs have
zero geometric error in these controls but are not necessarily the least noisy
after sharpening; erroneous interpolation can attenuate noise as well as signal.

## Iteration-budget sensitivity

Six individual frames are fitted with budgets of 100, 300 and 800 iterations,
holding the stronger penalty at 0.3. Clean-image RMS error in ADU is:

| Scene and case | 100 iterations | 800 iterations |
|---|---:|---:|
| Saturn, stationary blur/noise | 0.08927 | 0.09002 |
| Saturn, motion | 0.18051 | 0.18157 |
| Saturn, offset motion | 0.18379 | 0.18151 |
| Texture, stationary blur/noise | 0.09824 | 0.09844 |
| Texture, motion | 0.20035 | 0.20122 |
| Texture, offset motion | 0.22316 | 0.22160 |

The two stationary fits formally converge at 754 and 769 iterations; all four
moving fits still reach the 800-iteration limit. Longer fitting lowers the
training objective but does not consistently improve geometry or unused-pixel
prediction. The 100-iteration budget is retained for the controlled comparison,
with early stopping treated as part of this experimental estimator. Six frames
are insufficient to establish universal iteration-budget stability.

## Fractional translations expose a model error

A separate probe supplies the exact global translation and no true local
motion, for two scenes, three fractional shifts, blur sigma 0 or 1, and either
no added noise or noise sigma 2: 24 cases. The generator uses SciPy's
prefiltered cubic spline; the predictor uses PyTorch bicubic interpolation.

Validation accepts false local motion in all 12 cases without added noise and
7 of 12 noisy cases. In the noiseless cases the fitted false-motion RMS ranges
from 0.023 to 0.055 pixels. On the textured scene, clean-image error increases
to 0.150–0.243 ADU from the coherent baseline's 0.068–0.120 ADU. Although small
in displacement, this is a clear regression where the correct local field is
known to be zero.

To isolate the cause, 12 otherwise equivalent noiseless cases use the fitter's
own renderer as the generator. All produce exactly zero local motion and none
passes the validation gate. This deliberately matched-generator check isolates
the interpolation mismatch; it does not validate the renderer as a model of a
real telescope/camera.

Independent detector noise does not protect validation against a systematic
model error shared by training and unused pixels. A wrong warp can improve
prediction on both sets. Passing that score therefore cannot by itself justify
applying local motion, even though it is effective against the stationary
random-noise examples.

## Decision

Keep the joint estimator as experimental code. Its large clean-stack gains are
useful evidence, but the full set of tests does not establish a reliable
sharpened-image improvement, and the independent-pixel gate accepts known false
motion when the sampling model is wrong. No full captured-data replay or
production integration is performed in this experiment.

The next prerequisite is a forward sampling model that passes the fractional
translation tests. It should also be tested against an independent analytic or
oversampled scene generator, so that matching the simulator's interpolation
does not become the criterion for success. The real AutoStakkert noise gap
remains unresolved by this experiment.

Review images are `out/joint-fit-saturn-controls/saturn_comparison.png` and the
equivalent files in the other three completed control directories. They use
the exact user sharpening recipe and a common crop; the target column means
no *added* noise, since the fixed real reference retains its original structure.

## Reproduction

```sh
PYTHONPATH=. OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python tools/joint_convergence_probe.py \
  --out out/joint-convergence-v2.json

PYTHONPATH=. OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python tools/joint_controls.py \
  --out out/joint-fit-saturn-controls --scenes saturn --iterations 100
```

`tools/joint_translation_probe.py --out out/joint-translation-probe.json`
reproduces the translation stress test. Add `--generator model --noise 0` and
use a different output path for the matched-renderer isolation check.

Use `--scenes texture` for the other scene and `--seed 9018` for independent
noise. Use fresh output directories; earlier pilot and partial runs under
`out/joint-pilot` and `out/joint-*-controls` are not part of the final record.
Apply `tools/planetarytools_sharpen_experiment.py` to each completed directory's
PNGs, then run `tools/analyse_joint_controls.py --out DIRECTORY`.

`tools/collect_joint_results.py` checks the complete set of cases, identical
configuration/model hashes and sharpening artifacts. The analyzers verify the
coherent stacks and noiseless-oracle PNGs against earlier experiments. The
compact numerical record is
[`results/registration/joint-local-registration.json`](../results/registration/joint-local-registration.json).

The focused CPU/CUDA suite passes 24 tests. New checks cover known stationary
blur and integer translations, continuous stationary blur mixtures at both
gain bounds, preservation of observed brightness, blank-reference fallback,
and exact independence of fitted motion from unused detector samples.

## Limits

The synthetic blur partly matches the proposed model by construction; success
does not establish adequacy for anisotropic or spatially varying atmospheric
seeing. The fixed reference itself may be imperfect. The selected detector
subsample is fixed rather than averaged over multiple phases. Smoothness,
parameter bounds and early stopping can all bias the inferred geometry.
Predictive improvement on unused pixels does not prove true motion when the
image-formation model is incomplete. No claimed geometric truth is available
for the real capture.
