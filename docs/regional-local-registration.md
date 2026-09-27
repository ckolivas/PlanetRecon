# Regional evidence for local motion

The previous full-frame refit recovered part of the independent-pixel guard's
regression, but its single accept/reject decision still discarded local motion
over the entire image. This experiment tests whether the evidence differs
between the disc and rings, and whether spatial decisions improve reconstruction.
It leaves production registration and GUI defaults unchanged.

## Fixed model

The existing full-data coherent fit and checkerboard-half fits are unchanged.
Each half's candidate motion is evaluated against the other half's detector
samples using the existing forward-rendered reference model. Both hypotheses
retain the same blur/contrast nuisance freedom. Raw observations are not
resampled to calculate the validation score.

The reference defines six fixed sectors: upper/lower left, centre and right.
The horizontal centre is the reference intensity centroid; column boundaries
are at that centre plus/minus 0.22 times the reference support width. The row
boundary is the vertical centroid. On this Saturn capture these roughly
separate the disc and two ring sides. They are a heuristic spatial partition,
not a general planet or ring segmentation algorithm.

Each sector uses its own blur and contrast fit and needs at least 64 supported
pixels. Two policies are evaluated without changing parameters between runs:

- **Regional any:** accept a region if either half's held-out score improves.
- **Regional both:** require improvement in both half predictions.

The regional decisions multiply the residual displacement of the full-data
coherent fit. Sector boundaries blend over a fixed Gaussian sigma of 16 pixels,
with nonnegative weights summing to one. This smooths the motion-confidence
map only; it does not filter the raw frames or stack pixels. The resulting
field must still have minimum Jacobian 0.25 and maximum residual six pixels.
A rejected field falls back to global translation and the frame still
contributes at its original brightness and scalar weight.

As in the previous refit, the final full-data field is not itself an independent
prediction. Regional scores also have fewer samples and more opportunities for
false acceptance. Requiring both halves to agree is a tested policy, not a
formal statistical guarantee. Neither frame normalization nor output smoothing
is introduced.

## Real-frame probe

The probe evaluates 96 evenly spaced positions from the fixed 5,738-frame
selection without making a small stack. All whole-image decisions reproduce
the prior experiment. Of the 20 frames rejected by the whole-image gate,
19 have some regional support from either half and 18 have a region supported
by both halves. Seventy-two frames mix accepted and rejected sectors under
the any rule; 90 do so under the both rule.

| Sector | Either half accepts | Both halves accept | Both accept while whole image rejects |
|---|---:|---:|---:|
| Upper left ring | 82 | 49 | 5 |
| Upper disc | 64 | 26 | 1 |
| Upper right ring | 88 | 71 | 12 |
| Lower left ring | 85 | 66 | 11 |
| Lower disc | 55 | 18 | 1 |
| Lower right ring | 89 | 71 | 9 |

This supports the hypothesis that one image-wide score conceals regional
disagreement. It does not establish that the accepted shifts are geometrically
correct. In particular, regional nuisance fits have more flexibility than
the uniform whole-image blur/contrast model.

## Known-motion controls

Two scenes, four motion/blur cases and two independent noise seeds give 768
frames. Each method also warps the corresponding clean frames, so displacement
errors can be measured without confusing them with residual random noise.
The earlier coherent stacks reproduce to floating-point rounding, and the
noiseless-oracle exports match the earlier runs.

Requiring both halves to approve a region produces zero false displacement
in all 384 stationary frames. Accepting either half produces some false
displacement in 12 stationary Saturn frames. Although small, this is evidence
against using the less restrictive policy as a general replacement.

Clean-image RMSE against the known-motion oracle, in input ADU:

| Saturn case | Seed | Coherent | Whole-image any | Regional any | Regional both |
|---|---:|---:|---:|---:|---:|
| Motion and blur | 9017 | 0.241927 | 0.241927 | 0.238855 | 0.228522 |
| Offset motion and blur | 9017 | 0.262709 | 0.260894 | 0.260476 | 0.253423 |
| Motion and blur | 9018 | 0.253322 | 0.253322 | 0.250375 | 0.247067 |
| Offset motion and blur | 9018 | 0.294940 | 0.286295 | 0.289730 | 0.304221 |

The stricter regional gate helps three of these four cases relative to the
coherent fit, but regresses the offset-motion repeat. Both regional policies
preserve the coherent result on both moving texture cases at both seeds.
These controls support continued investigation, not a universal improvement.

Exact sharpening does not reverse that qualification: regional-both increases
noisy sharpened-image RMSE in all four moving Saturn cases, despite the clean
geometry improvements in three. For example, the first motion case changes
from 0.021594 to 0.021752 in linear image units, while the offset-motion repeat
changes from 0.022237 to 0.023000. Geometry and the sharpened noisy image must
be judged separately.

## Full 5,738-frame replay

All four policies use the same selected frames, scalar weights, global shifts,
reference and export mapping. The coherent and whole-image-any baseline stacks
reproduce the previous full replay to 1.43e-13 ADU; coverage agrees within
1e-10. No regional field triggered the final geometric fallback.

Of 1,281 frames rejected by the whole-image gate, 1,008 have at least one
region supported by both halves. There are only 55 frames with no regional-any
approval, and 516 with no regional-both approval. Both halves accept the upper
and lower disc on 1,752 and 1,040 frames respectively; the four ring sectors
receive 2,844, 3,910, 3,680 and 3,880 approvals. Regional disagreement persists
at full count, but admitting that evidence does not give an unambiguous benefit.

After the exact user sharpening recipe, with no added output filter:

| Method | Upper-disc variation % | Lower-disc variation % | Left ring width px | Right ring width px |
|---|---:|---:|---:|---:|
| Coherent | 2.62672 | 2.73750 | 7.39574 | 6.04747 |
| Whole-image any, full-data refit | 2.77885 | 2.91506 | 7.33578 | 6.04609 |
| Regional any | 2.71101 | 2.85055 | 7.46354 | 6.13231 |
| Regional both | 2.87417 | 2.83939 | 7.44048 | 6.14235 |
| Earlier gated half-field average | 3.03808 | 3.03037 | 7.34150 | 6.11851 |

Variation is robust high-pass MAD divided by the local median, not an isolated
sensor-noise measurement. Widths are fixed-window 10–90% transitions, not
calibrated angular resolution.

Relative to whole-image-any, regional-any lowers variation by about 2.4% and
2.2%, while broadening both fixed ring profiles. Regional-both raises upper
variation by 3.4%, lowers lower variation by 2.6%, and also broadens both fixed
profiles. Against the ungated coherent fit, regional-both has 9.4% and 3.7%
more variation.

The 75 overlapping common-centroid windows per ring side qualify the fixed
profile result. Relative to coherent, regional-any has median width changes
of +0.060 px left and +0.002 px right; regional-both has +0.119 px left and
-0.002 px right. The left side is wider in 95% and 96% of windows respectively.
The right-side change depends strongly on window placement. These are
sensitivity checks, not independent samples supporting a significance claim.

**Decision:** do not promote either regional gate into production. The stricter
rule protects all stationary controls, but neither rule consistently improves
known-motion reconstruction and the full real stack. Spatial gating alone does
not resolve the alignment/variation trade-off. A useful next experiment would
change the displacement estimator itself, measuring how blur and weak local
structure bias its shifts, rather than adding further accept/reject rules.

Local review artifacts are `out/saturn-regional-full/comparison.png`, the four
raw PNG/NPZ stacks, their `sharpened/` outputs and `analysis.json`.

## Reproduction

```sh
PYTHONPATH=. OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python tools/regional_controls.py \
  --out out/regional-saturn-controls --scenes saturn
```

Use `--scenes texture` for the second reference and `--seed 9018` for the
independent-noise repeat. Each case defaults to 48 frames. All methods share
the same generated frames, known motion, supplied global translations and
scalar weights, and are also applied to clean twins of the noisy inputs.
`tools/analyse_regional_controls.py` checks coherent baseline parity and
noiseless-oracle PNG hashes against the earlier runs, then scores the exact
PlanetaryTools sharpening recipe: Wavelet 27/0/0/0 followed by Adaptive
Deconvolution 15.6, Contrast Adaptive.

The real probe is `tools/regional_saturn_probe.py --out NEW_JSON`. The full
replay uses:

```sh
PYTHONPATH=. OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python tools/refit_saturn_experiment.py \
  --out out/saturn-regional-full --sample-count 5738 --regional \
  --start 0 --count 256
```

Process disjoint ranges through position 5,737, then add `--combine` with the
same model/sample arguments. Four stacks are produced: coherent, full-any,
regional-any and regional-both. The combiner checks input/model identity and
requires each frame exactly once. Pass the PNGs through the existing exact
sharpening harness and run `tools/analyse_refit_saturn.py` on the output root.

The real image has no known geometric truth. Fine-scale variation contains
detail and artifacts as well as noise; ring-width measurements are not
calibrated resolution. Overlapping edge windows are sensitivity checks, not
independent statistical replicates. This remains one capture and a limited
synthetic motion/blur family.

## Verification

The focused CPU/CUDA suite passes 45 tests across regional, refit, validated
and coherent registration, shard combination and the existing warp controls.
The regional tests check complete confidence coverage, blank-reference fallback
and preservation of the whole-image baseline on CPU and CUDA. Cross-instance
field comparisons allow 1e-12 pixels because independent CUDA reductions can
differ by a few float64 ULPs; accept/reject decisions remain exact checks.

`tools/collect_regional_results.py` assembles the compact numerical record at
[`results/registration/regional-local-registration.json`](../results/registration/regional-local-registration.json).
It verifies model and sharpening identities, complete control counts, the full
frame count and reproduction of both earlier real baseline stacks. Large
per-frame traces and images remain in the referenced local `out/` directories.
