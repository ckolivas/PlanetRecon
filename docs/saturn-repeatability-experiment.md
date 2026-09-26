# Saturn: where does the fine structure repeat?

2026-09-26. Follow-up to the alignment controls and the user's observation that
moving the reference anchor toward the midpoint made no noticeable difference.

**Subsequent decision:** the user does not want filtering introduced to explain
away the AS/PR discrepancy. The frequency controls below remain diagnostic only.
Their results do not establish that AS filters its stacks, or identify where PR's
non-repeatable variation arises. The earlier recommendation to pursue an output
filter was premature; investigation should continue with unfiltered stacking.

## Question and method

AP size, local alignment and reference-anchor changes did not explain the grain.
The next diagnostic asks which spatial frequencies repeat between disjoint
stacks, and whether the answer differs between the disc and sharp boundaries.

The exact 5,738 selected frame IDs from the original controlled experiment were
divided into consecutive pairs. A seeded random choice placed one member of
each pair into each half. Both halves contain 2,869 distinct frames spanning the
capture. Each run built its own reference from its best 64 frames. The reference
candidate sets are also disjoint. All assertions passed, and both runs used CUDA.
The two production stacking runs, including input verification and exports,
took 194.4 seconds on the RTX 5070.

Separate reference anchors put the results at slightly different locations. A
single relative translation was fitted using Gaussian-sigma-3 proxies and
applied to the second image in Fourier space. The measured images were not
Gaussian-smoothed. The fitted correction was (-0.6457, -0.8044) pixels in (y,x);
proxy residual RMS fell from 0.00759 to 0.000537 of peak brightness.

Five centroid-relative regions were measured: two disc patches, left and right
rings, and the north limb. Each patch has its sigma-2 smooth component removed,
a four-pixel border discarded, and a Hann window applied. The real Fourier
cross-power is summed in broad radial frequency bands and divided by the
geometric mean of the two powers. The same analysis was applied to the earlier
shared-reference halves for comparison.

This uses the two-image correlation principle described by
[Koho et al. (2019)](https://www.nature.com/articles/s41467-019-11024-z).
Here it is a broad-band repeatability diagnostic: no calibrated resolution
threshold or confidence interval is inferred from these small, windowed patches.

## Results with separate references

Correlation near one indicates strong agreement; near zero indicates little
agreement. Negative small values can occur from finite-sample fluctuations.

| Region | 0.05–0.10 cycles/pixel | 0.10–0.20 | 0.20–0.30 | 0.30–0.40 | 0.40–0.50 |
|---|---:|---:|---:|---:|---:|
| Upper disc | 0.993 | 0.509 | 0.075 | 0.099 | -0.016 |
| Lower disc | 0.989 | 0.538 | -0.061 | 0.048 | 0.035 |
| Left ring | 1.000 | 0.982 | 0.813 | -0.043 | 0.068 |
| Right ring | 1.000 | 0.983 | 0.759 | -0.046 | -0.059 |
| North limb | 1.000 | 0.989 | 0.399 | -0.052 | 0.013 |

The distinction survives using different reference data and a different random
split. In the 0.20–0.30 band (spatial periods about 3.3–5 pixels), the ring regions
have substantial repeatable structure while the disc patches have little.
Above 0.30 cycles/pixel all five measured regions have weak agreement. The
earlier shared-reference halves show the same overall pattern.

This explains why suppressing high frequencies indiscriminately is a poor
tradeoff: the sigma-one Gaussian control attenuates frequencies around
0.20–0.30 strongly, including the repeatable ring structure. These measurements
support testing a more selective response before changing the stacker.

They do not identify the origin of the non-repeatable variation as sensor noise,
prove that AS uses a particular filter, or establish that every correlated
feature is astrophysical truth. Seeing and independent reference deformations
can lower agreement; fixed detector or processing patterns can raise it.
Disjoint frames also do not guarantee independence of temporally correlated
atmospheric effects. The frequency bands describe sinusoidal spatial periods,
not a claimed minimum resolvable feature width.

## New diagnostic outputs

Two cosine frequency-taper controls were made from the original full PR stack:

- `local_taper_25_35_control.png`: preserves frequencies through 0.25
  cycles/pixel, smoothly tapering to zero at 0.35.
- `local_taper_20_35_control.png`: preserves frequencies through 0.20,
  then tapers to zero at 0.35; a stronger suppression comparison.

These use reflection padding and the original PR black/white mapping. They are
explicitly experimental filters, not an inferred AS kernel, a validated
restoration, or a new production default. They need evaluation for ring/limb
definition and ringing under sharpening. Floating `.npy` counterparts preserve
the complete values, and PNG quantization was checked against those arrays.
No claim is made that their post-sharpening performance has yet been verified.

The useful development direction is to estimate reliable structure by location
and scale and evaluate noise suppression against that structure. An improvement
must lower non-repeatable variation while retaining supported ring/limb detail;
making the disc look smooth is insufficient. The two filters are simple
controls for that tradeoff, not the complete adaptive method.

## Reproduction

```sh
PYTHONPATH=. .venv/bin/python tools/independent_stack_halves.py \
  --capture 2024-09-27-1154_3-CK-R-Sat.ser \
  --experiment out/saturn-controlled-alignment \
  --out out/saturn-independent-halves
PYTHONPATH=. .venv/bin/python tools/stack_repeatability.py \
  --independent out/saturn-independent-halves \
  --shared out/saturn-controlled-alignment
PYTHONPATH=. .venv/bin/python tools/frequency_response_control.py \
  out/saturn-controlled-alignment
.venv/bin/python -m pytest -q tests/test_stack_repeatability.py
```

The independent output directory must be new. The full halves, frame IDs and
provenance live there; the two taper controls live in the original experiment
directory. The tracked machine-readable record is
`results/real-data/saturn-repeatability.json`.

Validation: all selected IDs accounted for once; disjoint 64-frame references;
two complete 2,869-frame CUDA stacks; three passing scientific controls
(identical images, independent random images, known subpixel translation).
Production stacking and GUI behavior are unchanged.
