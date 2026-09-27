# Saturn pilot: direct fractional peak refinement

2026-09-27. Follow-up to [the noise ladder](ap-noise-ladder.md).

**Decision:** do not promote peak refinement or launch a full-count replay on
this evidence. Its mixed synthetic improvements did not produce a consistent
real-capture improvement. This is a negative 512-frame pilot, not proof that
fractional refinement can never help another dataset or registration model.

## Controlled comparison

Selected 512 evenly spaced positions (`np.linspace(0, 5737, 512, dtype=int)`)
from the frozen 5,738-frame Saturn selection. Both methods used exactly the
same raw frames, 64-frame reference, reference epoch, global shifts, scalar
quality weights, coherent field fitter, spatial support and raw backprojection.
The full capture SHA256 was checked before and after replay:
`cdf0a1158f6798106daa96f59f13b52228dcf4ecb00719f960406fc2f2089930`.

The baseline retains the original quadratic peak estimate. The candidate
refines fractional normalized correlation directly using the previously frozen
steps 0.25, 0.125 and 0.0625 pixels. Matching interpolation remains bilinear;
the cubic matching candidate is excluded. Forward/reverse consistency and
field guards retain their original rules. Their *outcomes* can change when the
estimated peaks change; this is an end-to-end refinement comparison, not an
ablation with all rejection decisions held constant.

Both 512-frame outputs and their alternating 256-frame halves received the
actual PlanetaryTools recipe: Wavelet **27/0/0/0**, then Adaptive Deconvolution
**15.6**, Contrast Adaptive. Recipe, implementation and source/output hashes
match the preceding experiments. No added output filtering, per-frame
brightness adjustment or output normalization was used. The shared PNG mapping
is unchanged. Production code and defaults are unchanged.

## Results

Fine variation is the existing robust sigma-2 residual statistic, expressed as
a percentage of each disc patch's median. It includes real detail and artifacts;
it does not identify a detector-noise source.

| Measurement | Original peaks | Refined peaks | Relative change |
|---|---:|---:|---:|
| Sharpened upper-disc fine variation | 12.0645% | 13.3426% | +10.59% |
| Sharpened lower-disc fine variation | 12.0055% | 11.5989% | -3.39% |
| Raw upper-disc half-difference/2 | 0.27336% | 0.27865% | +1.94% |
| Raw lower-disc half-difference/2 | 0.25665% | 0.26433% | +2.99% |
| Raw upper-disc full-stack variation | 0.33286% | 0.34306% | +3.06% |
| Raw lower-disc full-stack variation | 0.29384% | 0.27242% | -7.29% |

The halves carry 49.964% and 50.036% of the scalar quality weight. The effective
full frame count from those weights is 507.49. The difference divided by two
is therefore a near-equal-half diagnostic, not an exact estimate of every
component of full-stack noise. Shared reference/estimator biases cancel, and
interleaving does not make atmospheric effects independent.

The initial fixed-window sharpened ring widths were 8.9423 → 8.7375 pixels on
the left and 5.5389 → 6.0163 on the right. Wider window sensitivity gives a
mixed picture: across 75 paired windows per side, refined-minus-baseline
width differences (minimum / median / maximum) were:

- Left: **-0.3057 / -0.0230 / +0.2278 px**; refined was wider in 34.7%.
- Right: **-0.5207 / -0.0838 / +0.6397 px**; refined was wider in 33.3%.

These overlapping windows are a descriptive robustness check, not 75
independent observations or calibrated resolution estimates. The default
right window even changes the sign of the median result. They do not support
a strong sharpening claim.

Raw split-stack Fourier correlations also fail to show a consistent detail
improvement:

| Region and frequency band (cycles/pixel) | Original peaks | Refined peaks |
|---|---:|---:|
| Left ring, 0.10–0.20 | 0.9063 | 0.9075 |
| Right ring, 0.10–0.20 | 0.9071 | 0.9113 |
| Left ring, 0.20–0.30 | 0.3475 | 0.3371 |
| Right ring, 0.20–0.30 | 0.3731 | 0.3371 |
| North limb, 0.10–0.20 | 0.8692 | 0.8406 |

The full regional/frequency table is retained in the numerical record. Gaussian
residual extraction and Hann windows belong to the *measurement*, not the
exported stacks. No significance threshold is inferred from these small regions.
Visual inspection of the six-panel comparison likewise did not reveal a clear
reduction in grain with preserved ring detail.

## Registration checks and interpretation

All 32 shards completed, with every pilot frame accounted for once per method.
One frame from each shard was checked against the original coherent matcher:
maximum field difference **3.95e-13 px**. Nine existing focused tests passed
(matcher equivalence, shard validation and Fourier repeatability controls).

The original/refined field guards rejected 24/27 frames, respectively; these
frames still contribute through global alignment. Both methods accepted 481
frames and rejected 20, with 7 accepted only by baseline and 4 only by refined.
Median accepted AP counts were 72 and 73. The median paired field RMS difference
was 0.3852 px over supported pixels, so the candidate was actually exercised.
There were no insufficient-point fallbacks.

Recorded matching time summed to 33.9 seconds for baseline and 141.2 seconds
for refinement, about 4.2 times as much. These are per-frame timings under four
concurrent workers, not an isolated throughput benchmark. All four workers
used four CPU threads and the same CUDA device.

This pilot rejects a practical gain for the tested candidate. It does not
explain the AutoStakkert discrepancy, and a 512-frame stack must not be judged
as a matched-noise comparison to its 5,738-frame outputs. A remaining diagnostic
question is whether moving the estimated peaks or changing their acceptance
and fallback decisions drives the observed differences. Separating those would
require an explicit fixed-decision control; the present outputs cannot answer
that causal question.

## Artifacts and reproduction

- `out/saturn-peak-refinement/comparison.png`: identical-scale sharpened full and half stacks.
- `out/saturn-peak-refinement/{baseline,refined}.npz`: floating stacks and provenance.
- `out/saturn-peak-refinement/halves.npz`: raw weighted sums and support.
- `out/saturn-peak-refinement/sharpened/`: six exact-recipe exports and float stages.
- `results/registration/saturn-peak-refinement.json`: audited numerical record.

```sh
PYTHONPATH=. .venv/bin/python tools/saturn_peak_refinement.py \
  --out out/saturn-peak-refinement --prepare
# Run worker 0, 1, 2 and 3, each with these thread limits:
PYTHONPATH=. OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python tools/saturn_peak_refinement.py \
  --out out/saturn-peak-refinement --workers 4 --worker 0
PYTHONPATH=. .venv/bin/python tools/saturn_peak_refinement.py \
  --out out/saturn-peak-refinement --combine
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  MKL_NUM_THREADS=4 QT_QPA_PLATFORM=offscreen \
  ../PlanetaryTools/planetary-app/.venv/bin/python \
  tools/planetarytools_sharpen_experiment.py \
  --planetary-tools ../PlanetaryTools/planetary-app \
  --out out/saturn-peak-refinement/sharpened out/saturn-peak-refinement/*.png
PYTHONPATH=. OPENBLAS_NUM_THREADS=4 QT_QPA_PLATFORM=offscreen \
  .venv/bin/python tools/analyse_saturn_peak_refinement.py \
  --out out/saturn-peak-refinement \
  --record results/registration/saturn-peak-refinement.json
.venv/bin/python -m pytest -q tests/test_fractional_ap_trace.py \
  tests/test_coherent_saturn_experiment.py tests/test_stack_repeatability.py
```

Use a fresh output directory. `--resume` reuses existing same-identity shards;
combining revalidates their shapes, finite values, hashes, frame associations and
complete coverage. Rendering the analyzer's gallery is separate from sharpening;
do not feed that gallery back into the sharpening wildcard on subsequent runs.
