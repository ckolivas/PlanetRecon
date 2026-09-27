# Disjoint native reference ensembles on Saturn

2026-09-27. The earlier [reference interventions](reference-imprint-experiment.md)
showed sensitivity to added artificial texture. This experiment instead changes
the actual reference frame content and asks whether a larger reference or an
ensemble of measured alignment fields gives a useful real-capture improvement.

## Controlled design

The replay contains the same 512 positions from the fixed 5,738-frame selection
used by the recent pilots. Those 512 positions are excluded from reference
construction. From the remainder, the best 256 scalar-quality scores are chosen,
then sorted chronologically into 64 adjacent quartets. A seeded random
permutation (`270927`) assigns one member of every quartet to each of four
64-frame reference cohorts. Thus the four cohorts are disjoint, drawn from the
same quality pool, and approximately matched in temporal coverage.

Each reference averages original raw values using the already frozen global
shifts, accumulating signal and coverage before division. It uses equal weights
within the reference. A pooled reference uses all 256 signal/support arrays.
Where a reference has no detector coverage, the old reference fills that border;
all four cohorts have complete 64-frame coverage over the measured object mask
(old reference above 2% of its peak), so this fill does not supply object texture.

These are disjoint reference *inputs*, not fully independent pipelines: they
share the old coordinate system and global shift estimates. Seeing and blur,
planet rotation, template sampling and grain all contribute to their differences.
The selected reference frames tend toward times with high quality. This design
does not claim to establish a new physical midpoint epoch.

Seven methods use exactly the same raw replay frames, scalar quality weights,
reference coordinate system and final bilinear sampler:

- Original historical reference, reproducing the preceding baseline.
- Each of the four native 64-frame references (A, B, C, D).
- Their pooled 256-frame reference.
- The arithmetic mean of the four resulting displacement fields for each frame.

Only matching proxies, patch templates and template strengths change. AP
positions, eligibility, support, gradient-response kernels, spline design and
regularization stay fixed at baseline. Matching confidence and geometric guards
respond normally. Rejected fits fall back to global translation as usual.
The mean field has its own unchanged minimum-Jacobian/maximum-residual guard.

The ensemble averages **fields**, then resamples each original frame once. It
does not average four resampled copies or add four contributions per raw frame.
The original reference differs from the new ones in source membership and
construction; baseline-versus-pooled is therefore not a count-only comparison.
The four cohorts versus their own pooled reference form the matched reference
ensemble comparison.

All seven full stacks and fourteen alternating half-stacks receive the exact
PlanetaryTools recipe: Wavelet 27/0/0/0 followed by Adaptive Deconvolution 15.6,
Contrast Adaptive. No output filtering, brightness normalization or production
change is made.

## Results and decision

**Keep production unchanged.** Neither a larger native reference nor averaging
four measured fields gives a consistent reduction in variation with preserved
detail. The pooled reference has a modest ring-repeatability gain but mixed
grain; it is a tradeoff to investigate, not an established noise solution.
The field ensemble's small drop in sharpened variation is contradicted by
increased raw split differences, especially below the rings.

All 32 shards completed for all seven methods. The original baseline float
stack matches the preceding pilot within 9.95e-14 ADU, and its sharpened PNG
is pixel-identical. All 21 full/half exports used verified identical sharpening
settings and implementation hashes.

Relative changes from the original baseline follow. Positive values mean more
variation, not necessarily more sensor noise. Fine variation includes real
detail and artifacts; half-difference/2 excludes common reference biases.

| Reference/method | Sharpened upper disc | Sharpened lower disc | Raw upper half-difference/2 | Raw lower half-difference/2 |
|---|---:|---:|---:|---:|
| A, 64 frames | -2.01% | -2.78% | -2.31% | +2.43% |
| B, 64 frames | +1.57% | +10.22% | -1.76% | +6.67% |
| C, 64 frames | -4.64% | -1.95% | -0.47% | +7.45% |
| D, 64 frames | -2.55% | -2.88% | +0.11% | +2.92% |
| Pooled, 256 frames | +4.49% | -0.74% | -4.48% | +4.32% |
| Mean of four fields | -1.69% | -3.21% | +1.37% | +7.87% |

Absolute baseline sharpened variation is 12.0645% / 12.0055% of patch median
brightness; the mean-field variant is 11.8607% / 11.6202%. The four 64-frame
references span 11.5049–12.2534% in the upper patch and 11.6593–13.2321% below.
Choosing whichever cohort happens to give the smallest number would be a
post-hoc selection, not evidence for a generally better reference algorithm.

The four reference-derived fields have median RMS spread **0.5470 px** over
the fixed object mask, with range 0–1.2518 px across frames. The intervention
therefore substantially changes alignment despite modest/mixed image metrics.
Final field-guard rejections are 24 for baseline; 18, 20, 18 and 17 for A–D;
17 for pooled; and zero for the mean field. Rejected individual fields use the
original global fallback and remain part of the four-field mean.

Raw half-stack ring correlations in the 0.20–0.30 cycles/pixel band:

| Reference/method | Left ring | Right ring |
|---|---:|---:|
| Original baseline | 0.3475 | 0.3731 |
| A | 0.3721 | 0.3647 |
| B | 0.3276 | 0.3417 |
| C | 0.3843 | 0.3617 |
| D | 0.2976 | 0.3563 |
| Pooled | 0.3743 | 0.3993 |
| Mean field | 0.3497 | 0.3860 |

The pooled-reference correlations improve on both rings in this pilot. These
small-region broad-band measurements have no calibrated significance threshold,
and share each method's reference between halves. They are not proof of extra
astronomical detail. Their improvement must be weighed against the other
metrics, rather than discarding the mixed result.

The 75-window ring-width check gives median candidate-minus-baseline differences
of +0.1104 / +0.0446 px for pooled and +0.1589 / -0.0243 px for the mean field
(left / right). All ranges include positive and negative differences. Fixed
single windows can disagree in sign with these medians. No robust resolution
improvement is established. Visual inspection of the seven-panel gallery
showed similar grain with reference-dependent changes to ring structure.

## Verification, limits and artifacts

Ten focused tests passed, including deterministic disjoint/time-balanced
reference membership, replay exclusion, pooled signal/support arithmetic,
constant field averaging, and exact baseline restoration after changing
matching templates. AP geometry and the spline design are verified unchanged.
The 32 original-matcher checks reached a worst difference of 7.10e-13 px.
Capture content and all artifact/recipe identities were checked.

Reference cohort median raw frame IDs are 22789, 22768, 22767.5 and 22787.5;
all span approximately frames 1637–27681. This confirms balanced cohort timing,
but also shows that the chosen high-quality pool is weighted toward the later
capture. It does not provide a midpoint-centred temporal template. The result
must not be interpreted as testing a different anchoring epoch.

The 512-frame pilot is not a matched-count comparison to the 5,738-frame AS
outputs. The two replay halves are disjoint but share reference information and
can have correlated atmospheric effects. The references differ in seeing,
rotation-dependent morphology, blur and resampling as well as noise. Original
reference construction and the new reference construction also differ, so the
study does not isolate the native reference's noise content or establish that
reference noise causes the AS/PR discrepancy.

- `out/saturn-native-references/comparison.png`: all seven identically sharpened stacks.
- `out/saturn-native-references/references.npz`: actual references, membership and raw signal/support sums.
- `out/saturn-native-references/`: snapshots, halves, shards and per-frame fit statistics.
- `out/saturn-native-references/sharpened/`: 21 exact-recipe outputs and float stages.
- `results/registration/saturn-native-references.json`: audited numerical record.

## Reproduction

Use a fresh output directory. Run workers 0, 1, 2 and 3:

```sh
PYTHONPATH=. .venv/bin/python tools/saturn_native_references.py \
  --out out/saturn-native-references --prepare
PYTHONPATH=. OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python tools/saturn_native_references.py \
  --out out/saturn-native-references --workers 4 --worker 0
PYTHONPATH=. .venv/bin/python tools/saturn_native_references.py \
  --out out/saturn-native-references --combine
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  MKL_NUM_THREADS=4 QT_QPA_PLATFORM=offscreen \
  ../PlanetaryTools/planetary-app/.venv/bin/python \
  tools/planetarytools_sharpen_experiment.py \
  --planetary-tools ../PlanetaryTools/planetary-app \
  --out out/saturn-native-references/sharpened out/saturn-native-references/*.png
PYTHONPATH=. OPENBLAS_NUM_THREADS=4 QT_QPA_PLATFORM=offscreen \
  .venv/bin/python tools/analyse_saturn_native_references.py \
  --out out/saturn-native-references \
  --record results/registration/saturn-native-references.json
.venv/bin/python -m pytest -q tests/test_native_reference_ensemble.py \
  tests/test_coherent_saturn_experiment.py tests/test_stack_repeatability.py
```

`--resume` retains same-identity completed shards. Combination verifies complete
unique frame accounting and rehashes the capture. The analyzer creates the
comparison gallery after sharpening; do not include it in a later wildcard
sharpening invocation.
