# Local warp centring: controlled validation

This follows the Saturn [local-warp trace](saturn-local-warp-trace.md). That
experiment found that subtracting the average local displacement reduced one
ring-edge distortion. This validation tests whether that correction preserves
known geometry, rather than merely making that particular edge narrower.

**Decision: do not enable automatic mean-warp subtraction.** It improves the
cases centred on the chosen reference, but removes real local deformation when
the capture's average pose differs from that reference. The result persists
with global translation preserved and with a second independent noise seed.

## Centring results

Clean-image RMSE against the noiseless known-motion reconstruction, in ADU,
for seed 9017 (smaller is better):

| Scene / condition | Current local | Subtract mean local warp | Change |
|---|---:|---:|---:|
| Saturn / clean symmetric motion | 0.2457 | 0.2097 | 15% lower |
| Saturn / symmetric motion, blur, noise | 0.2551 | 0.1410 | 45% lower |
| Saturn / true offset, blur, noise | 0.7957 | 1.1500 | 45% higher |
| Saturn / unequal frame weights | 0.3262 | 0.6048 | 85% higher |
| Textured disc / clean symmetric motion | 0.1692 | 0.0231 | 86% lower |
| Textured disc / symmetric motion, blur, noise | 0.2327 | 0.0369 | 84% lower |
| Textured disc / true offset, blur, noise | 0.2695 | 2.7578 | 10.2 times as large |
| Textured disc / unequal frame weights | 0.2471 | 1.5194 | 6.1 times as large |

The decisive control supplies **perfect displacement measurements** and then
subtracts their mean residual. In the offset case, that changes a geometrically
exact reconstruction into one with RMSE 1.1586 ADU on Saturn and 2.7581 ADU on the
textured disc. Thus the failure is inherent to imposing the mean-pose constraint,
not just a weakness of the current correlation matcher. That constraint could
be an explicit reference-coordinate choice, but it is not a general correction
for false motion and does not automatically preserve a midpoint-time anchor.

The second noise seed retains the same pattern: offset-case error rises by
45% on Saturn and by a factor of 10.3 on the textured disc. The unequal-weight
case deliberately uses strong weights, `exp(0.8*sin(phase))`; it is a stress test,
not a claim that the real Saturn capture had similarly unequal scalar weights.

Exact sharpening retains the offset-case regression. Saturn's error in linear
0..1 output units increases from 0.02865 to 0.02996, and the textured disc from
0.01455 to 0.03728. These are errors against the sharpened noiseless oracle, not
estimates of sensor noise or measured astronomical resolution.

## Controlled inputs

`tools/validate_local_warp_centring.py` generates 48-frame captures from two
scenes: the unchanged Saturn reference and a separately generated textured disc.
Horizontal and vertical sinusoidal shears are composed in a specified order,
and frame generation uses their analytic inverse. The maximum shear amplitudes
are 1.5 and 1.2 pixels. Generation uses cubic interpolation; reconstruction uses
the production bilinear sampler, with one final resampling of each raw frame.

The cases are:

* No motion, independent noise only.
* No motion, changing blur and independent noise.
* Symmetric motion, no additional blur or noise.
* Symmetric motion, changing blur and independent noise.
* Motion correlated with blur.
* Motion offset from the reference pose by shear amplitudes +0.8 and -0.5 pixels.
* Symmetric motion with unequal frame weights that favour one side of the motion.

Added Gaussian blur ranges from zero to 1.5 pixels sigma. Independent Gaussian
noise has sigma 2 in camera ADU; this is an illustrative stress condition, not an
estimate of the camera's noise. Seeds 9017 and 9018 provide separate repetitions.
Within a seed, noise arrays are paired across cases. No frames are discarded.

The global translation supplied to the matcher is the spatial mean of the
**known true displacement** over the scene mask, separately for each frame.
Mean subtraction operates only on the remaining local displacement. Thus the
test does not manufacture a failure by accidentally deleting global translation.
It isolates local registration from errors in the global estimator.

The controls are global translation only, current local alignment, local
alignment minus its weighted mean residual, known true motion (the oracle), and
the oracle minus its weighted mean residual. All preserve raw brightness. The
export mapping is fixed at 0..255 ADU for every image; no stack normalization or
output smoothing is applied.

## Error measurements

Displacement error is measured against the known field, in vector RMS pixels.
It is decomposed into mean error and error varying between frames. Subtracting
a fixed field cannot change frame-to-frame displacement differences at a fixed
reference pixel; it can only change their shared reference geometry.

For image error, a noiseless oracle stack uses exactly the same frame generation,
weights and bilinear reconstruction, but receives the known true displacement.
The measured fields are also applied to noiseless twins of the input frames.
This clean-image error separates geometric damage from the reduction of noise
that fractional interpolation can produce. A lower noisy-image error alone is
insufficient evidence of better registration.

The fixed scene mask includes pixels brighter than 8% of the scene maximum,
eroded by three pixels, excluding an eight-pixel detector border. Error is also
reported against the mean blurred latent scene, exposing the interpolation floor.

The same PlanetaryTools recipe is applied to the first repetition and the
matched-blur diagnostic: Wavelet 27/0/0/0, then Adaptive Deconvolution 15.6 with
Contrast Adaptive enabled. Sharpened errors compare each result with the same
recipe applied to the noiseless oracle; these errors include geometry, detail
and noise, and are not pure noise measurements.

## Matched-blur diagnostic

The additional `--match-known-blur` experiment supplies the **known simulated
additional blur** to the reference template for each frame. Raw frame samples
and stacking interpolation are unchanged. This is an oracle diagnostic, not an
implemented automatic blur estimator. Rebuilding the template also rebuilds its
AP texture and acceptance information, so this tests the whole matching path.

Matching the known blur reduces clean-image error in all ten tested noisy/blurred
conditions. For symmetric motion it falls from 0.2551 to 0.1986 ADU on Saturn
(22%) and from 0.2327 to 0.1677 ADU on the textured disc (28%). For offset motion
the changes are 0.7957 to 0.7791 ADU (2%) and 0.2695 to 0.2127 ADU (21%).

However, Saturn's full-mask displacement error hardly changes: 0.6940 to 0.6942
pixels RMS for symmetric motion and 0.8309 to 0.8364 pixels for offset motion.
The textured scene improves from 0.1130 to 0.0899 pixels for symmetric motion.
Image error and uniformly weighted displacement error measure different things;
matching blur helps some image structure without solving the whole motion field.
Knowing the correct blur is therefore a useful diagnostic but not a sufficient
solution or a validated automatic procedure.

The raw improvement does not translate into lower sharpened error on Saturn:
the symmetric-motion error changes from 0.02830 to 0.02856 in linear 0..1 units,
and the offset-motion error from 0.02865 to 0.02893. All five Saturn sharpened
comparisons are slightly worse with the known-blur template. The textured
scene's four moving cases improve modestly after sharpening. The oracle inputs
are byte-identical across the baseline and matched-blur runs, confirming that
the generated frames, noise and reconstruction target were held fixed.

The next algorithmic target is a spatially coherent local motion estimate with
explicit reference geometry, tested against these known displacements. Any blur
handling should remain in the matching model. These results do not justify
filtering stack pixels or applying a blanket mean-warp correction.

## Reproduction

```sh
PYTHONPATH=. .venv/bin/python tools/validate_local_warp_centring.py \
  --out out/local-warp-validation
PYTHONPATH=. .venv/bin/python tools/validate_local_warp_centring.py \
  --out out/local-warp-validation-repeat --seed 9018
PYTHONPATH=. .venv/bin/python tools/validate_local_warp_centring.py \
  --out out/local-warp-blur-template --match-known-blur \
  --cases static_blur_noise motion_blur_noise motion_correlated_blur \
          offset_motion_blur weighted_motion_blur
```

For each of `out/local-warp-validation` and `out/local-warp-blur-template`, pass
its generated PNGs to `tools/planetarytools_sharpen_experiment.py`, using the
PlanetaryTools Python environment and a new `sharpened` subdirectory. Then run:

```sh
QT_QPA_PLATFORM=offscreen PYTHONPATH=. .venv/bin/python \
  tools/analyse_local_warp_validation.py --out out/local-warp-validation
```

Outputs include float stacks, clean twins, true/estimated mean fields, raw and
sharpened PNGs, per-case parameters and measurements, and a comparison montage.
Earlier `*-pilot` and `*-zero-global` directories are exploratory controls that
supplied zero global shift; the reported conclusions use the explicit global
translation control above.

## Limits

These are synthetic shears and Gaussian blur, not a full physical seeing or
rotation model. Saturn's source already contains its reference texture and
noise. Inputs are floating point rather than quantized SER samples. Knowing
the displacement and blur is possible only in this controlled test. Neither a
successful synthetic result nor an edge-width improvement on one real capture
is enough to enable a correction in production.

## Validation and retained evidence

The complete matrix comprises 38 captures of 48 frames, or 1,824 synthetic
frames: 14 primary conditions, 14 repeat conditions, and 10 known-blur-template
conditions. Nine targeted tests pass, including CPU/CUDA trace parity and the
new analytic-inverse, pull-direction, weighted-reference and zero-motion tests.
The long combined runner was terminated before the final repeat condition;
that condition was rerun separately with the same seed and recorded as a
continuation. No experiment worker remains running.

The compact numerical record is
[`results/registration/local-warp-validation.json`](../results/registration/local-warp-validation.json).
It retains every variant's measurements, generation parameters, source-report
hashes and sharpening implementation identity. Full arrays and image products
remain in the `out/` directories above. Production stacking code and defaults
are unchanged.
