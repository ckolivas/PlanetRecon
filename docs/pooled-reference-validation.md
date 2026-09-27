# Frozen pooled-reference validation on unused frames

2026-09-27. Follow-up to [native reference ensembles](saturn-native-references.md).
The pooled 256-frame reference gave modest ring-repeatability gains in the first
512-frame coherent-fitter pilot, with mixed grain. This run tests whether that
tradeoff repeats on unused input samples and with the production local matcher.

## Fixed design

Both references are reused byte-for-byte from the previous experiment; no new
reference or parameter is selected using these results. The candidate remains
the same pooled 256-frame image. The historical baseline remains the original
64-frame reference. Baseline AP placement, eligibility, gradient kernels, spline
constraints, global shifts, scalar quality weights and final raw sampler stay
fixed. As before, the references differ in preparation as well as membership,
so this is not a pure reference-count test.

Exclude the earlier 512 replay positions, all 256 native reference positions,
and every original historical reference frame. Select 1,024 evenly spaced
positions from the remaining fixed 5,738-frame selection. Alternating positions
form two disjoint 512-frame cohorts spanning the capture. Alternating positions
within each cohort produce two 256-frame halves. Neither cohort overlaps either
reference or the first pilot. They share the observing session and may share
correlated seeing; disjoint frames are not independent atmospheric realizations.

Each cohort has four matched stacks:

- Production circular matcher, historical reference.
- Production circular matcher, frozen pooled reference.
- Experimental coherent fitter, historical reference.
- Experimental coherent fitter, frozen pooled reference.

One AP trace supplies both the production field (its last accepted stage) and
the observations for the coherent fit. Per-shard checks compare each reference
and matcher against their original implementations. This tests the production
*local matcher* with frozen experiment inputs, not a fresh GUI run that would
reselect frames and rebuild registration.

The comparison looks for repeatable direction on both rings in both new
cohorts, with grain and raw split differences considered alongside it. No
calibrated significance threshold or success claim follows from a single
favourable region. All eight full stacks and sixteen half stacks receive the
same actual PlanetaryTools Wavelet 27/0/0/0 then Adaptive Deconvolution 15.6,
Contrast Adaptive recipe. No output filter or brightness normalization is added.

## Completed results and decision

**The initial gain does not validate consistently. Do not promote the pooled
reference or change the production default.** The coherent fitter improves both
ring correlations in set 1 but loses both in set 2. The production matcher has
mixed ring results and sharpened variation changes around one percent, with
opposite signs between sets. This does not establish a reliable improvement
for the tested reference; it does not rule out all larger-reference designs.

Ring half-stack correlations in the same 0.20–0.30 cycles/pixel band:

| Sample | Matcher | Left, original → pooled | Right, original → pooled |
|---|---|---:|---:|
| Initial pilot | Coherent | 0.3475 → 0.3743 | 0.3731 → 0.3993 |
| New set 1 | Production | 0.3170 → 0.3294 | 0.2390 → 0.2158 |
| New set 1 | Coherent | 0.4138 → 0.4444 | 0.3621 → 0.3878 |
| New set 2 | Production | 0.3202 → 0.3283 | 0.1852 → 0.2464 |
| New set 2 | Coherent | 0.4683 → 0.4429 | 0.3269 → 0.3195 |

Relative changes in the pooled-reference stack versus its matching original
reference stack are below. Positive values mean more variation. The sharpened
statistic includes real detail and artifacts, and is not a pure noise estimate.

| Sample | Matcher | Sharpened upper / lower disc | Raw half-difference/2, upper / lower |
|---|---|---:|---:|
| New set 1 | Production | -0.97% / -0.75% | -0.71% / +0.52% |
| New set 1 | Coherent | -1.47% / -4.71% | -2.72% / -3.98% |
| New set 2 | Production | +1.16% / +0.50% | +1.84% / -0.22% |
| New set 2 | Coherent | -6.38% / +8.81% | +2.30% / +2.64% |

The coherent result in set 1 is encouraging in isolation, but set 2 does not
repeat it. Averaging the favourable and unfavourable regions into a single
number would hide that lack of consistency. Likewise, shared reference biases
can make a feature repeat in both halves; correlation alone does not establish
that the feature is astronomical truth.

Across 75 paired sharpened ring-edge windows per side, median pooled-minus-
original widths (left / right) were:

- Set 1, production: -0.0100 / +0.0037 px.
- Set 1, coherent: +0.1232 / +0.0743 px.
- Set 2, production: -0.0527 / +0.0496 px.
- Set 2, coherent: -0.0036 / -0.0223 px.

Every per-side range includes both signs. The windows overlap and are not
independent trials; widths are not calibrated resolution measurements. Visual
review of the eight-stack montage also showed no consistent improvement across
both samples and matchers.

## Verification and limits

- All 1,024 unique new frame IDs were replayed, with two complete 512-frame
  cohorts and correct disjoint 256-frame halves in every method.
- Eleven focused tests passed, including an explicit check of exclusions,
  cohort membership and the four accumulator slots.
- Each of the four matcher/reference combinations has 32 comparisons against
  its original implementation. Worst production difference: 8.13e-14 px;
  worst coherent difference: 7.52e-13 px.
- Coherent final-field rejections were 55/1,024 for the original reference and
  40/1,024 for pooled. Rejected local fits use the normal global fallback.
  A lower rejection count is not itself proof of better registration.
- Effective scalar-weight frame counts were 509.45 and 509.43. Half weights
  were 50.030%/49.970% for set 1 and 49.900%/50.100% for set 2.
- Capture bytes, reference arrays, frozen input identities, snapshot equality,
  complete accounting, and all 24 sharpening artifacts were verified.

Half-difference/2 is a near-equal-half diagnostic and excludes common biases.
Neither the samples nor the reference populations are independent observing
sessions. All frequency/edge findings are descriptive and have no formal
resolution or significance threshold. This validation does not identify the
cause of AutoStakkert's measured smoothing response or attribute the variation
to sensor noise. Production code and defaults remain unchanged.

## Artifacts

- `out/saturn-pooled-validation/comparison.png`: eight matched full-stack images.
- `out/saturn-pooled-validation/`: manifests, full snapshots, raw half sums and fit diagnostics.
- `out/saturn-pooled-validation/sharpened/`: 24 exact-recipe exports and float stages.
- `results/registration/pooled-reference-validation.json`: audited numerical record.

## Reproduction

Use a fresh output directory and run workers 0 through 3:

```sh
PYTHONPATH=. .venv/bin/python tools/validate_pooled_reference.py \
  --out out/saturn-pooled-validation --prepare
PYTHONPATH=. OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python tools/validate_pooled_reference.py \
  --out out/saturn-pooled-validation --workers 4 --worker 0
PYTHONPATH=. .venv/bin/python tools/validate_pooled_reference.py \
  --out out/saturn-pooled-validation --combine
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  MKL_NUM_THREADS=4 QT_QPA_PLATFORM=offscreen \
  ../PlanetaryTools/planetary-app/.venv/bin/python \
  tools/planetarytools_sharpen_experiment.py \
  --planetary-tools ../PlanetaryTools/planetary-app \
  --out out/saturn-pooled-validation/sharpened out/saturn-pooled-validation/*.png
PYTHONPATH=. OPENBLAS_NUM_THREADS=4 QT_QPA_PLATFORM=offscreen \
  .venv/bin/python tools/analyse_pooled_reference_validation.py \
  --out out/saturn-pooled-validation \
  --record results/registration/pooled-reference-validation.json
.venv/bin/python -m pytest -q tests/test_pooled_reference_validation.py \
  tests/test_native_reference_ensemble.py tests/test_coherent_saturn_experiment.py \
  tests/test_stack_repeatability.py
```

`--resume` retains same-identity completed shards. The combined half slot is
`position % 4`: cohort 0 uses slots 0/2; cohort 1 uses 1/3. Tests independently
verify that each exported cohort and half contains the correct unique IDs.
The gallery is created after sharpening; exclude it from later wildcard runs.
