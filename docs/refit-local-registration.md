# Separating detector splitting from motion rejection

The previous independent-pixel check removed invented motion in stationary
controls but increased fine-scale variation in the full Saturn stack. It
simultaneously changed the motion estimate and applied a new acceptance test.
This experiment separates those effects and tests a full-data refit after
independent evidence indicates that local motion is present.

## Fixed alternatives

Five alternatives use the same pixels, reference, scalar frame weights and
supplied global translations:

1. **Coherent:** the previous full-frame coherent fit, unchanged.
2. **Half average:** average the two checkerboard-half fits without the extra
   predictive gate; the existing matcher and geometric guards still apply.
3. **Validated:** the previous gated half-field average, unchanged.
4. **Full any:** if either held-out test accepts local motion, use the original
   full-frame coherent fit at full strength; otherwise use global translation.
5. **Full scaled:** use the full-frame fit with residual strength 0, 0.5 or 1,
   according to the number of accepted halves. This preserves the earlier
   method's reduction in strength while changing its displacement estimate.

The last two are diagnostic refits. Independent samples decide whether local
motion is supported, but the final full-data displacement is not itself an
independent prediction. There is no claim that this prevents every form of
noise fitting. No brightness normalization or added output filter is used.

The existing validation and coherent implementations are left unchanged.
`RefitRegistration` captures their intermediate fields and retains an exact
baseline path, tested on both CPU and CUDA. The final outputs use the same
bilinear raw-pixel sampler and the user's exact PlanetaryTools recipe:
Wavelet 27/0/0/0, then Adaptive Deconvolution 15.6 with Contrast Adaptive.

## Known-motion controls

The full-any candidate was tested on 48-frame stationary-noise,
stationary-blur-and-noise, motion-blur-and-noise, and offset-motion cases,
for both Saturn and the textured disc: eight cases and 384 synthetic frames.
The same seeds, motions and supplied global shifts were retained.

All 192 stationary frames remain at their true global displacement. For moving
Saturn with blur/noise, clean-twin RMS error is unchanged from the original
coherent fit at 0.24193 ADU. In the offset case it changes from 0.26271 to
0.26089 ADU, while sharpened error rises from 0.022076 to 0.022329 in linear
0..1 units. This again separates geometric fidelity from noise or appearance
after nonlinear sharpening. The textured-disc moving cases retain the original
coherent result because every frame has evidence for local motion.

Four further 48-frame moving controls test the full-scaled alternative, bringing
the total to 12 cases and 576 frames. Its clean-twin errors for Saturn motion
and offset motion are 0.23662 and 0.25604 ADU, slightly lower than both the
coherent and full-any fits in these controls. Both textured-disc cases reproduce
coherent because both halves pass on every frame. Its stationary zero-motion
behavior follows the same zero gate and is also covered by the focused test.
Sharpened Saturn errors nevertheless rise slightly: 0.021594 to 0.021661 for
motion and 0.022076 to 0.022388 for offset motion. The textured outputs are
unchanged. Lower clean-twin geometric error again does not guarantee a lower
error after sharpening noisy frames.
This is a limited set of scenes and motions, not evidence that reducing
displacement strength is always appropriate.

## Matched subset and why full-count evaluation matters

A fixed, evenly spaced sample of 512 positions within the existing 5,738-frame
selection was replayed under all five alternatives. Every validation decision
matches the previously recorded full replay at those positions: 512 of 512.
This verifies that the new experiment preserves the original gate behavior.

After the exact sharpening recipe:

| Alternative | Upper variation | Lower variation | Fixed left width | Fixed right width |
|---|---:|---:|---:|---:|
| Coherent | 12.0645% | 12.0055% | 8.9423 px | 5.5389 px |
| Half average | 11.8774% | 12.8720% | 8.6411 px | 6.3831 px |
| Validated | 11.2102% | 12.3310% | 8.4054 px | 6.4409 px |
| Full any | 12.7076% | 11.9575% | 10.0422 px | 6.4306 px |
| Full scaled | 12.1218% | 12.1162% | 8.7391 px | 6.3340 px |

This subset does not reproduce the earlier full-count upper-disc ranking:
validated is lower than coherent here, but was higher in the 5,738-frame
comparison. The lower frame count has much stronger grain after this aggressive
nonlinear recipe. It is therefore an ablation/pilot result, not a reliable
substitute for the final-count image or a direct comparison with AutoStakkert.
The wide spread across overlapping ring windows is another reason not to
select a model from one fixed width measurement.

## Scope of rejection

In the full recorded validation, neither half passes on 1,281 frames, accounting
for 21.94% of total scalar frame weight. Only 98 of those frames already fell
back to global alignment under the coherent model's geometric guard. Thus the
new test discards otherwise admissible local motion on 1,183 additional frames.

Median accepted AP counts in the original full fit are 71, 72 and 72 for frames
with zero, one and two accepted halves. Median quality is 6.397, 6.506 and 6.583
in the stored score units. The gate is not simply rejecting frames with no AP
measurements. Passing a geometric guard is not proof that a fit is correct;
these counts identify where the methods diverge, not which one is ground truth.

## Full 5,738-frame result

The full-count replay retains every original frame and its scalar contribution.
The recomputed coherent image matches the prior coherent stack to a maximum
difference of 8.53e-14 ADU; sample support is identical. Pilot decisions also
match exactly, with maximum floating-point scoring difference 2.60e-13. The
comparison therefore isolates the estimator and gating policy.

After identical sharpening, with no added output filtering:

| Alternative | Upper variation | Lower variation | Fixed left width | Fixed right width |
|---|---:|---:|---:|---:|
| Coherent | 2.6267% | 2.7375% | 7.3957 px | 6.0475 px |
| Previous validated average | 3.0381% | 3.0304% | 7.3415 px | 6.1185 px |
| Full any | 2.7789% | 2.9151% | 7.3358 px | 6.0461 px |
| Full scaled | 2.8419% | 2.7692% | 7.3487 px | 6.0332 px |

Full-any lowers upper/lower variation by 8.5% / 3.8% relative to the previous
validated average. It remains 5.8% / 6.5% above coherent. Full-scaled lowers
variation by 6.5% / 8.6% relative to the validated average, but remains 8.2% /
1.2% above coherent. Using the full-frame estimator recovers part of the prior
regression while retaining the gate; rejection itself also has a measurable
effect. These are variation measurements, not an independent noise estimate.

The window-sensitivity check is more informative than the single fixed left
width. Across 75 paired windows, median left width relative to coherent is
+0.0066 pixels for full-any, +0.0725 for full-scaled and +0.0901 for the prior
validated average. Corresponding right medians are +0.0083, +0.0010 and +0.0134.
The full-any differences span zero on both sides, so its ring profiles are
much closer to the original coherent model than those of the previous guard.
The narrower single fixed left width does not establish a general resolution
gain. Visual inspection of `out/saturn-refit-full/comparison.png` is consistent
with broadly similar ring structure and small differences in disc grain.

Full-any is a useful compromise in the tested controls: it removes stationary
false displacement, preserves most of the original coherent ring improvement,
and recovers some of the previous guard's lost sharpening tolerance. It is
still not a universal improvement over the ungated model, and its final refit
can reuse noise that influenced the gate. It remains an experiment rather than
a new production default.

## Reproduction

```sh
PYTHONPATH=. OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python tools/validate_local_warp_centring.py \
  --out out/refit-example --scenes saturn --coherent-spacing 16 \
  --coherent-stiffness .01 --validate-pixels --refit-policy full_any \
  --cases static_noise static_blur_noise motion_blur_noise offset_motion_blur
```

For the five-way pilot use `tools/refit_saturn_experiment.py --out NEW_DIR`,
which defaults to 512 evenly spaced frames and 256-frame shards. Process both
`--start 0` and `--start 256`, then `--combine`.

For the full-count test add `--sample-count 5738 --cached-full-refit`. The cached
path checks every reference, frame-ID, weight, global-shift, model and parameter
identity and rejects missing or duplicate frame decisions. It avoids repeating
the already verified checkerboard analysis and recomputes the full coherent
field once per frame. Disjoint shards accumulate weighted pixel sums and support
for coherent, full-any and full-scaled outputs; combination requires every
selected frame exactly once.

Run the existing PlanetaryTools sharpening harness on the exported PNGs, then
`tools/analyse_refit_saturn.py --out NEW_DIR`. All generated image/float products
remain under `out/saturn-refit-ablation/` and `out/saturn-refit-full/`.

These are diagnostic tools. Production registration and GUI defaults are
unchanged. Fine-scale variation includes real detail and artifacts; the real
capture has no known geometric truth, and edge width is not calibrated angular
resolution. The 75 overlapping edge windows are sensitivity checks, not
independent statistical replicates.

All 41 focused tests pass, including CPU/CUDA baseline agreement and rejection
of cached decisions with missing/duplicated frames, different weights, model
code or fit parameters. The checked-in numerical record is
[`results/registration/refit-local-registration.json`](../results/registration/refit-local-registration.json).
