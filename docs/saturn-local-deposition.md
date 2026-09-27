# Local distortion: backward sampling versus forward deposition

2026-09-27. Follow-up to the user's mono-drizzle hypothesis and the negative
[peak-refinement](saturn-peak-refinement.md) and
[fixed-decision](saturn-ap-decisions.md) controls.

## Scope

The earlier [native-size deposition test](native-mono-drizzle.md) used global
translations. Unit-square deposition and bilinear backward sampling are
identical for that case, including coverage. A spatially varying deformation
does not have that translation equivalence. This experiment changes only the
combination of detector pixels, holding the estimated local field fixed.

The earlier supplied exact-copy AS controls already establish effective spatial
smoothing for that configuration. They do not locate its internal processing
stage or prove the same response for every capture. The remaining development
question is detail retained at comparable fine variation; treating low grain
alone as evidence of better alignment would be misleading. This deposition
experiment does not claim to recreate AutoStakkert internals.

## Method

The same 512 fixed positions from the 5,738-frame selection are used, with the
same 64-frame reference and epoch, scalar quality weights, global translations,
coherent local fields and output mapping as the preceding pilots.

For each frame the local field maps output coordinates `q` to detector
coordinates `p = q + d(q)`. Forward deposition solves this equation for `q` at
every detector centre. A bounded Newton iteration samples the same bilinear
field, using nearest-value exterior extension. A run fails if any coordinate
fails the 1e-7 pixel inverse-residual tolerance; it cannot silently accept a
poor inverse. Nonfinite fields and singular/reversed Newton Jacobians fail.

Every detector pixel deposits a fixed, axis-aligned unit square centred at its
inverse position. Exact overlap with output unit squares produces triangular
weights. This is a specified kernel at native output scale; it is **not** exact
integration of a warped detector polygon, super-resolution, or a proprietary
AS drizzle implementation.

Three stacks use the same raw frames and displacement field:

1. **Original pull:** existing bilinear backward sampling and support handling.
2. **Pooled deposition:** sum quality-weighted deposited signal and deposited
   coverage across frames, then divide once.
3. **Equal-frame deposition:** divide each deposited frame by its local coverage
   first, then average supported output pixels with the unchanged scalar frame
   quality weights. This separates the deposition kernel from the extra local
   weighting caused by variable sampling density.

The per-frame spatial coverage division in the third control is not a fitted
brightness adjustment. No frame intensity statistic is rescaled. Constant
brightness is preserved in both deposition controls. There is no added output
filter, brightness normalization, or change to production stacking.

All three full stacks and their six alternating half-stacks use the actual
PlanetaryTools Wavelet 27/0/0/0 followed by Adaptive Deconvolution 15.6,
Contrast Adaptive. The previous 512-frame baseline is reproduced as a control.

## Results and decision

**Do not promote either deposition control.** Native unit-square forward
placement under these local warps changes the measured grain by less than 1%
and does not improve ring repeatability. This rules out a large benefit from
this particular substitution in this pilot, not all possible drizzle kernels,
output scales, or local recombination methods.

| Method | Sharpened upper-disc variation | Sharpened lower-disc variation | Change from baseline (upper / lower) |
|---|---:|---:|---:|
| Original pull | 12.06451% | 12.00548% | — |
| Pooled deposition | 11.99940% | 12.08806% | -0.540% / +0.688% |
| Equal-frame deposition | 12.00760% | 12.11942% | -0.472% / +0.949% |

Fine variation is the existing robust sigma-2 residual statistic relative to
patch brightness. It includes real detail and artifacts. It is not a direct
measurement of sensor noise. The measurements use unaltered output stacks;
residual extraction is only part of analysis.

Raw half-difference/2 changes were +0.237% / -0.816% for pooled deposition and
+0.123% / -0.881% for equal-frame deposition (upper / lower). Thus the conclusion
is not simply an artifact of the nonlinear sharpening metric. The halves share
a reference, so common biases cancel and this is not a total-noise estimate.

Median deposition-minus-baseline sharpened ring-width changes across the 75 paired
windows per side were -0.0042 / -0.0026 pixels for pooled deposition and
-0.0018 / -0.0029 for equal-frame deposition. Every side's range includes both
positive and negative changes. These are tiny descriptive changes, not a claim
of calibrated resolution improvement.

Raw ring half-stack correlations in the 0.20–0.30 cycles/pixel band were:

| Method | Left ring | Right ring |
|---|---:|---:|
| Original pull | 0.34754 | 0.37314 |
| Pooled deposition | 0.34561 | 0.37016 |
| Equal-frame deposition | 0.34393 | 0.37018 |

The slightly lower correlations have no formal significance assessment. They
provide no evidence of a detail improvement to offset the mixed grain changes.
Visual inspection likewise found the three sharpened full stacks very similar.

## Verification and limits

All 32 shards and 512 frames completed. Twelve focused tests passed. New tests
compare translation deposition to the independent CPU area-overlap operator,
recover a known affine inverse, preserve a constant's brightness, conserve an
interior impulse, and reject nonfinite fields.

- Worst inverse-equation residual: **9.925e-8 px**, below 1e-7 everywhere.
- Inverse iterations: minimum 0, median 5, maximum 8.
- Median across frames of object coverage 5th/50th/95th percentiles:
  **0.97556 / 1.00753 / 1.05179**. Thus density effects were actually exercised.
- Baseline raw stack maximum difference from the previous pilot: **8.527e-14 ADU**.
- Baseline sharpened output: **pixel-identical** to the preceding pilot.
- Original-matcher checks on 32 frames: maximum field difference **4.885e-13 px**.
- All methods share the same 24 rejected local fits and global fallbacks.
- Capture bytes, input identities, complete frame accounting, half accumulators
  and the nine sharpening source/output artifacts were checked.

This is a 512-frame pilot, not a matched-count comparison to the supplied
5,738-frame AS stacks. The alternating halves are disjoint but share a reference
and potentially correlated atmospheric effects. Ring widths and broad-band
correlations are descriptive diagnostics, not resolution or significance tests.

At an integer translation, the tested deposition reduces to exact pixel
placement. It therefore cannot itself generate the broad Gaussian response
measured earlier on AS's static, identical-frame control. The broader origin
of that response remains unidentified, while this experiment closes the
specific local-warp deposition hypothesis tested here.

## Artifacts

- `out/saturn-local-deposition/comparison.png`: identical-scale sharpened comparison.
- `out/saturn-local-deposition/`: full float stacks, raw halves, fit and inverse statistics.
- `out/saturn-local-deposition/sharpened/`: nine exact-recipe outputs and float stages.
- `results/registration/saturn-local-deposition.json`: audited numerical record.

## Reproduction

Use a fresh output directory and run workers 0 through 3:

```sh
PYTHONPATH=. .venv/bin/python tools/saturn_local_deposition.py \
  --out out/saturn-local-deposition --prepare
PYTHONPATH=. OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python tools/saturn_local_deposition.py \
  --out out/saturn-local-deposition --workers 4 --worker 0
PYTHONPATH=. .venv/bin/python tools/saturn_local_deposition.py \
  --out out/saturn-local-deposition --combine
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  MKL_NUM_THREADS=4 QT_QPA_PLATFORM=offscreen \
  ../PlanetaryTools/planetary-app/.venv/bin/python \
  tools/planetarytools_sharpen_experiment.py \
  --planetary-tools ../PlanetaryTools/planetary-app \
  --out out/saturn-local-deposition/sharpened out/saturn-local-deposition/*.png
PYTHONPATH=. OPENBLAS_NUM_THREADS=4 QT_QPA_PLATFORM=offscreen \
  .venv/bin/python tools/analyse_saturn_local_deposition.py \
  --out out/saturn-local-deposition \
  --record results/registration/saturn-local-deposition.json
.venv/bin/python -m pytest -q tests/test_local_deposition.py \
  tests/test_coherent_saturn_experiment.py tests/test_stack_repeatability.py
```

`--resume` retains same-identity completed shards. Combination verifies hashes,
frame association, unique complete coverage and the capture bytes. The analyzer
creates a gallery after sharpening; exclude that gallery from any later
wildcard sharpening command.
