# Corrected joint fitting on the full Saturn selection

This experiment replays the existing 5,738-frame selection from
`2024-09-27-1154_3-CK-R-Sat.ser` using the corrected experimental spline
forward model. The objective is to test whether its controlled-scene
improvements carry over to the real capture under the user's exact sharpening.

## Fixed inputs and methods

Four outputs share the same selected frames, quality weights, reference,
global translations and final float64 bilinear sampling:

- Global translations only.
- The previous coherent alignment-point estimator.
- Joint blur/motion fitting with the corrected prefiltered cubic spline
  predictor, 32-pixel field spacing and bending penalty 0.3.
- The same joint field applied only when it improves prediction on held-out
  detector samples; otherwise the frame uses its global translation.

Every frame remains in each stack, including a frame whose local motion is
rejected. Raw signal and coverage are summed with the original scalar quality
weights and divided after accumulation. Photometric gain and offset belong
only to the joint fitting score. They do not modify frame values or weights.
There is no added output filter or brightness normalization.

The joint optimizer fits even-row/even-column pixels and checks the remaining
three quarters. Its budget is 100 iterations per frame. Additional blur lies
between Gaussian sigma 0 and 2, represented by the same adjacent-template
mixtures used in the preceding controls. No full-data refit follows validation.
This remains an experimental optimizer, not a production-speed stacker.

The reference, selection, weights, global shifts and model sources are hashed.
The complete SER file is hashed before and after replay. Its SHA-256 is
`cdf0a1158f6798106daa96f59f13b52228dcf4ecb00719f960406fc2f2089930`.
Each shard records frame positions and associated fit diagnostics. Combination
requires every selected frame exactly once and identical configuration and
hashes across shards. The output set refuses existing export filenames.

A 32-frame pilot verifies the global result against an independent SciPy CPU
calculation to 1.14e-13 ADU. Unit tests additionally cover weighted half-stack
combination and rejection of diagnostics associated with the wrong frames.

## Full replay result

All 5,738 frames were processed exactly once across 45 verified shards. The
replayed global and coherent stacks match their previous float64 outputs to
6.54e-13 and 1.99e-13 ADU respectively. The complete SER hash is unchanged.
All twelve full/half PNGs were sharpened using the actual PlanetaryTools
implementation and the specified recipe, with matching implementation hashes.

| Method | Upper-disc variation (%) | Lower-disc variation (%) | Left ring width (pixels) | Right ring width (pixels) |
|---|---:|---:|---:|---:|
| Global translations | 2.95770 | 2.98324 | 7.48450 | 6.13284 |
| Coherent AP estimator | 2.62672 | 2.73751 | 7.39574 | 6.04747 |
| Corrected joint fit | 3.04394 | 3.06306 | 7.57159 | 6.27128 |
| Joint fit with held-out validation | 3.04394 | 3.06306 | 7.57159 | 6.27128 |
| Existing production output | 2.96676 | 2.97506 | 8.14171 | 6.36435 |
| AS manual-64 reference, context only | 0.57321 | 0.64187 | 7.27540 | 5.65088 |

The corrected joint fit increases sharpened disc variation by 15.9% and 11.9%
relative to coherent AP fitting. Ring transitions are also wider: across 75
overlapping paired windows per side, median increases are 0.186 pixels on the
left and 0.130 on the right. The joint result is wider in 89.3% and 100% of
those windows. These windows are sensitivity checks, not independent samples
for a significance claim. Visual inspection of the comparison agrees with
the absence of a useful improvement.

Every joint field passes the held-out gate, making the two joint outputs
identical. Median held-out prediction error falls by 2.83% relative to the
global model, yet the stack becomes grainier. A better prediction score under
this model therefore does not establish a better alignment for stacking.
It can still reflect model mismatch shared by fitting and validation pixels.
This experiment does not establish that the fitted motion is intrinsic grain,
or identify which missing part of the seeing model is responsible.

All 5,738 optimizations stop at the 100-iteration budget; none reports formal
convergence. No geometric fallback fires. Median residual vector RMS is 1.046
pixels; 921 fits (16.1%) reach the sigma-2 blur ceiling. The separate sensitivity
check below limits, but does not eliminate, these optimization concerns.

**Decision:** retain this as a negative experimental result. Do not promote
the joint estimator, change production defaults, add normalization, or add an
output filter on this evidence. The numerical sampling correction remains
supported by the independent controls, but its synthetic improvements do not
carry over to an improved real Saturn stack in this configuration. The
AutoStakkert gap remains unresolved; its own selection/reference and measured
spatial response prevent attributing that entire gap to motion estimation.

The complete numerical record is
[`joint-spline-saturn.json`](../results/registration/joint-spline-saturn.json).
Review images are `out/saturn-joint-spline-full/comparison.png` and
`out/saturn-native-drizzle/comparison.png`. The focused CPU/CUDA suite passes
41 tests covering the fitting, sampling, shard combination and deposition
controls. The result analyzers additionally verify the completed replay and
exact sharpening artifacts.

## Half-stack interpretation

The two halves alternate selected-frame positions, giving 2,869 frames in
each half for every method. Raw sums and coverage are retained separately.
Their difference divided by two approximates the random component of a full
mean when the two halves have comparable independent noise. Quality weights
and interpolation make this an approximation; the analysis reports the
coverage of both halves.

Both halves share the same reference and fitting model. Shared grain,
geometric bias and real structure can cancel from their difference, so it does
not measure every artifact in the final stack. Separately sharpened halves
are exported for inspection, but their difference is not treated as a linear
estimate of noise in the sharpened full stack.

The halves carry 49.973% and 50.027% of the scalar quality weight, with
effective frame counts 2,847.5 and 2,846.8 (5,694.3 for the full selection).
The measured raw high-pass variation is:

| Method | Full mean upper/lower (%) | Half-difference/2 upper/lower (%) |
|---|---:|---:|
| Global translations | 0.16870 / 0.14086 | 0.08335 / 0.07857 |
| Coherent AP estimator | 0.16025 / 0.13913 | 0.07478 / 0.08101 |
| Corrected joint fit | 0.17312 / 0.15059 | 0.08448 / 0.07741 |

The joint fit increases full-mean variation in both disc patches; the
half-difference result is mixed. This does not support describing all of the
extra final grain as a simple increase in independent sensor noise.

## Blur-range and iteration-budget sensitivity

The initial completed replay shards showed frequent contact with the
two-pixel blur ceiling. A separate diagnostic fits six predetermined positions
spread across the full selection, with maximum physical sigma 2 or 4 and
budgets of 100 or 300 iterations. It uses the frozen fitter by mapping the
wider bank's physical variances into its existing optimization coordinate.
This changes parameter scaling, so equal iteration counts alone do not prove
equal optimization accuracy.

The three capped frames choose physical sigmas about 2.02–2.09 when the range
is extended. At 300 iterations, held-out mean squared error changes between
-0.062% and +0.028% across all six frames. Other frames remain below sigma 2.
No fit in this 24-fit sensitivity probe formally converges. This small test
does not support changing the ceiling during the fixed replay, nor does it
establish that the limited isotropic blur model is adequate for real seeing.

## Drizzle follow-up

During this replay the user suggested native mono drizzle as a possible
explanation for AutoStakkert's noise characteristics. The separate
[native-size deposition experiment](native-mono-drizzle.md) uses the complete
same selection and fixed global shifts. Unit-square deposition reproduces
the existing bilinear stack; point deposition reproduces integer shifts.
A half-pixel footprint increases fine-scale variation after sharpening.
These explicit kernels do not explain the earlier measured broad AS response
on identical-frame controls. A different local recombination kernel remains
an implementation question, not a measured fact about AutoStakkert.

## Reproduction

```sh
PYTHONPATH=. OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python tools/joint_saturn_experiment.py \
  --out out/saturn-joint-spline-full --prepare

PYTHONPATH=. OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python tools/joint_saturn_experiment.py \
  --out out/saturn-joint-spline-full --worker 0 --workers 4
```

Run workers 1, 2 and 3 with the same worker count, then run the command with
`--combine` instead of the worker arguments. Default shard size is 128 frames.
The four workers each use four CPU threads and share CUDA. The source must
remain fixed while the replay runs, and output directories must be fresh.

Use `tools/planetarytools_sharpen_experiment.py` in the PlanetaryTools Python
environment on all generated PNGs. The recipe remains Wavelet 27/0/0/0,
followed by Adaptive Deconvolution amount 15.6, Contrast Adaptive enabled.

`tools/joint_blur_ceiling_probe.py --out out/saturn-joint-blur-ceiling.json`
produces the sensitivity check. Run `tools/analyse_joint_saturn.py` with
`--out out/saturn-joint-spline-full` and
`--record results/registration/joint-spline-saturn.json` after sharpening and
the native-drizzle analysis. It verifies unchanged baseline stacks, complete
frame diagnostics, source/input identities and sharpening hashes, and records
paired ring-width sensitivity over fixed overlapping windows.
