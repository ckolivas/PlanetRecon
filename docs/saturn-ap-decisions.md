# Saturn: separate AP positions from acceptance decisions

2026-09-27. Follow-up to [the peak-refinement pilot](saturn-peak-refinement.md).

## Question and controls

Direct fractional peak refinement did not consistently improve the preceding
512-frame real Saturn pilot. That change can alter both measured positions and
which AP measurements survive the forward/reverse check. It also changes later
coarse-to-fine stages, robust fitting weights and final field rejection.

This experiment uses the same 512 evenly spaced positions from the frozen
5,738-frame selection, the same reference and epoch, global shifts, quality
weights, final raw backprojection and output mapping. Five methods are replayed:

| Method | AP positions | AP acceptance | Parent stages, robust factors and final guard |
|---|---|---|---|
| Baseline | Original | Original | Original |
| Positions only | Refined on baseline stage inputs | Original | Frozen baseline |
| Acceptance only | Original | Refined on baseline stage inputs | Frozen baseline |
| Positions + acceptance | Refined on baseline stage inputs | Refined on baseline stage inputs | Frozen baseline |
| Full refinement | Previous refined algorithm | Previous refined algorithm | Previous refined algorithm |

The three factorial controls never propagate their changed measurements into
later matching stages. Each layer receives the original baseline parent field.
Stage acceptance, spline design, regularization and support are also fixed.
Since the discrete correlation search sees the same input, its peak-strength
confidence remains the same; the changed acceptance comes from the reverse
check using refined forward and reverse positions.

The robust fitter normally solves three times, updating weights after each
solve. The controls freeze the robust factors actually used by the **third**
solve; they do not accidentally use its subsequent, unused weight update.
These factors are evaluated from baseline residuals for every AP, including
those admitted only by the refined acceptance decision. The acceptance-only
control therefore changes measurement membership while holding positions and
robust residual factors fixed. The fitter still recalculates the weighted
linear system for the new membership.

Final accept/fallback decisions are deliberately forced to match baseline in
these three controls. Their natural guard outcomes are recorded separately.
This is a diagnostic intervention, not a proposed production algorithm or
permission to disable geometric safeguards. Nonfinite fields are forbidden.

All five full stacks and ten disjoint half-stacks use the exact PlanetaryTools
recipe: Wavelet 27/0/0/0, then Adaptive Deconvolution 15.6, Contrast Adaptive.
No brightness normalization or additional output filtering is introduced.

## Results and decision

**None of the controls gives a consistent improvement. Keep production unchanged
and stop pursuing this peak-refinement candidate as the noise solution.**
Freezing acceptance does not rescue the changed positions; freezing positions
does not rescue the changed acceptance. This is evidence about the tested
matcher and this capture, not every possible subpixel estimator.

The raw full stacks reproduce the previous baseline and full-refinement outputs
to maximum differences of 1.07e-13 and 2.56e-13 ADU. Both sharpened PNG arrays
are **pixel-identical** to the prior pilot. The baseline reconstructed with
frozen robust factors matches the original field exactly on all 512 frames.
All 32 shards completed, with complete, unique frame accounting.

Changes below are relative to baseline. Positive fine-variation or
half-difference changes indicate more variation. Fine variation includes real
detail as well as artifacts and is not a detector-noise measurement.

| Method | Sharpened upper disc | Sharpened lower disc | Raw upper half-difference/2 | Raw lower half-difference/2 |
|---|---:|---:|---:|---:|
| Positions only | +6.97% | +1.06% | +0.99% | +1.09% |
| Acceptance only | +4.04% | +0.77% | -1.16% | +0.19% |
| Positions + acceptance | +7.61% | +0.85% | +1.91% | -1.20% |
| Full refinement | +10.59% | -3.39% | +1.94% | +2.99% |

The changes are **not additive**: changing measurement membership changes the
weighted solve, and the sharpening and robust variation statistic are nonlinear.
One cannot subtract these percentages to assign a fraction of the full
refinement's result to each cause. Full refinement also changes parent stages,
robust factors and final guards; this experiment holds those fixed together
and does not distinguish their individual effects.

Median sharpened ring-width changes over 75 paired windows per side were:

| Method | Left (px) | Right (px) |
|---|---:|---:|
| Positions only | +0.0070 | +0.0267 |
| Acceptance only | -0.0118 | +0.0340 |
| Positions + acceptance | -0.0184 | +0.0359 |
| Full refinement | -0.0230 | -0.0838 |

Every method's per-side window range includes both positive and negative
changes. These overlapping windows describe sensitivity to measurement location;
they are not independent trials or calibrated resolution estimates.

Raw half-stack correlations in the 0.20–0.30 cycles/pixel ring band were:

| Method | Left ring | Right ring |
|---|---:|---:|
| Baseline | 0.3475 | 0.3731 |
| Positions only | 0.3591 | 0.3618 |
| Acceptance only | 0.3516 | 0.3665 |
| Positions + acceptance | 0.3584 | 0.3516 |
| Full refinement | 0.3371 | 0.3371 |

A slight gain on one ring accompanies a loss on the other in the three
controls. The north-limb 0.10–0.20 correlation was 0.8692 at baseline, 0.8689
for positions, 0.8604 for acceptance, 0.8597 for both, and 0.8406 for full
refinement. These are descriptive comparisons without significance thresholds.
Visual inspection of the five full stacks did not show a clear cleaner result
with preserved ring detail.

Across all frames and stages, 35,306 AP measurements were accepted in both
baseline and the frozen-parent refined check, 1,101 only in baseline and 1,719
only with refinement. The AP intervention therefore exercised real acceptance
changes, rather than being a no-op.

The three controls inherit all 24 baseline final-field fallbacks. Their forced
acceptances include **2 / 1 / 3** frames (positions / acceptance / both) that
would naturally fail the field guard. Natural accept/fallback outcomes disagree
with baseline on 3 / 3 / 4 frames overall. These controls intentionally hold
decisions fixed and must not be adopted as production algorithms. We have not
separately removed those frames to quantify their contribution.

Eleven focused tests passed, including new controls verifying exact frozen-IRLS
reproduction, zero contribution from rejected observations, preservation of
parent fields, first-stage equivalence with direct refinement, and unchanged
confidence for APs accepted by both methods. The capture was hashed before and
after replay; every sharpening source/output and implementation hash was checked.

## Limits and artifacts

The halves each have 256 frames and share a reference. Their scalar weight
fractions are 49.964% and 50.036%. Half-difference/2 cancels shared biases and
does not measure all full-stack noise. Seeing can remain temporally correlated.
No ground-truth Saturn image is available, and these 512-frame outputs are not
a matched-count comparison with the 5,738-frame AutoStakkert stacks.

- `out/saturn-ap-decisions/comparison.png`: five identically sharpened full stacks.
- `out/saturn-ap-decisions/`: floating stacks, 32 shards, raw halves and fit statistics.
- `out/saturn-ap-decisions/sharpened/`: all 15 exact-recipe exports and float stages.
- `results/registration/saturn-ap-decisions.json`: audited numerical record.

## Reproduction

Use a fresh output directory. Run worker 0 through 3 with four CPU threads each:

```sh
PYTHONPATH=. .venv/bin/python tools/saturn_ap_decisions.py \
  --out out/saturn-ap-decisions --prepare
PYTHONPATH=. OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python tools/saturn_ap_decisions.py \
  --out out/saturn-ap-decisions --workers 4 --worker 0
PYTHONPATH=. .venv/bin/python tools/saturn_ap_decisions.py \
  --out out/saturn-ap-decisions --combine
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  MKL_NUM_THREADS=4 QT_QPA_PLATFORM=offscreen \
  ../PlanetaryTools/planetary-app/.venv/bin/python \
  tools/planetarytools_sharpen_experiment.py \
  --planetary-tools ../PlanetaryTools/planetary-app \
  --out out/saturn-ap-decisions/sharpened out/saturn-ap-decisions/*.png
PYTHONPATH=. OPENBLAS_NUM_THREADS=4 QT_QPA_PLATFORM=offscreen \
  .venv/bin/python tools/analyse_saturn_ap_decisions.py \
  --out out/saturn-ap-decisions \
  --record results/registration/saturn-ap-decisions.json
.venv/bin/python -m pytest -q tests/test_fixed_ap_decisions.py \
  tests/test_fractional_ap_trace.py tests/test_coherent_saturn_experiment.py \
  tests/test_stack_repeatability.py
```

The analyzer creates `comparison.png` only after sharpening. Do not include
that gallery in a subsequent wildcard sharpening run. `--resume` reuses
same-identity completed shards, with full validation again on combination.
