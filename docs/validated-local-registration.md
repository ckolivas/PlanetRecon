# Independent-pixel validation of local motion

The [coherent-field experiment](coherent-local-registration.md) improved the
real Saturn replay but introduced false motion in stationary synthetic frames.
This experiment tests whether a proposed correction predicts detector samples
that were not used to estimate it. It is an experimental registration safeguard;
no production alignment or GUI default is changed.

## Method

Each monochrome frame is divided into two interleaved checkerboard sets of
detector pixels. A normalized Gaussian interpolation forms a matching proxy
from each set separately. The existing area-aware coherent fit estimates a
displacement field from each proxy, with the previous parameters held fixed:
16-pixel spline spacing, bending strength 0.01, and the same AP measurements,
support taper and geometric guards.

Each proposed field is then tested against the other set's proxy. The comparison
is between the supplied global translation and the proposed local field. To
avoid rewarding a correction merely because interpolation smooths the observed
noise, the observations stay fixed: the reference is rendered into detector
coordinates by inverting the candidate pull field. Inversion uses 20 fixed-point
iterations, and a candidate fails if the maximum coordinate residual in the
score mask exceeds 0.02 pixels.

Both hypotheses have the same nuisance model for blur and contrast. The model
bank contains reference proxies with additional Gaussian blur sigma 0 through
2 pixels in 0.25-pixel increments, including positive combinations of adjacent
templates. A positive gain between 0.5 and 2 and a fitted constant offset are
used only for this comparison. This prevents a change in focus or brightness
from automatically being counted as evidence for displacement. It is an
approximate image model, not an estimate of the optical point-spread function.

A local field is accepted only when its held-out squared residual is strictly
lower than the global model's. Otherwise that half uses the global translation.
The two resulting fields are averaged and checked for a minimum Jacobian of
0.25. Original raw pixels are then sampled once with this final field. Every
selected frame contributes with its existing scalar weight, without changing
its brightness. Neither output smoothing nor stack normalization is added.

The checkerboard independence concerns detector samples used to form the
proxies. It does not imply independence for spatially correlated detector noise,
and the fixed real reference can contain frames also present in the stack.
The blur bank is finite and spatially uniform. These are explicit limits of
this prototype.

## Known-motion controls

The unchanged synthetic generator provides exact invertible motion and clean
twins of noisy frames. Comparisons use the same reference, supplied global
translations, weights, blur and random samples as the prior experiment. The
26 cases comprise all 14 primary cases, four shorter-wavelength motion cases,
and eight cases with an independent noise realization: 1,248 frames in total.
Parameters were not retuned to these results.

All eight stationary cases, totaling 384 frames, reject both proposed local
fields on every frame. Clean-twin error against known-motion reconstruction
is exactly zero. This removes the prior false displacement for both Saturn
and the textured disc, including changing blur and both noise realizations.

An additional detector-phase check supplies three known fractional global
translations to each scene, with and without extra blur and with independent
noise. All 12 frames retain exactly their supplied global displacement and
introduce zero local residual. This exercises a case absent from the original
stationary controls, where the global shift was zero. Reproduce it with
`tools/validate_detector_phase.py --out out/validated-detector-phase-replay.json`
using the repository Python environment and `PYTHONPATH=.`.

Selected clean-image RMS errors against the known-motion reconstruction, ADU:

| Scene / condition | Previous coherent fit | Validated fit |
|---|---:|---:|
| Saturn / stationary, noise | 0.31419 | 0 |
| Saturn / stationary, blur and noise | 0.24222 | 0 |
| Saturn / motion, blur and noise | 0.24193 | 0.22524 |
| Saturn / true offset, blur and noise | 0.26271 | 0.27332 |
| Saturn / shorter-scale motion, blur and noise | 0.25253 | 0.25913 |
| Textured disc / motion, blur and noise | 0.09589 | 0.09486 |
| Textured disc / true offset, blur and noise | 0.10903 | 0.10804 |
| Saturn / second noise seed, motion and blur | 0.25332 | 0.26051 |

Clean-image error improves in 19 of 26 cases; some moving Saturn cases regress
slightly. This fixes the stationary failure without establishing superiority
for every moving scene.

All outputs receive the exact PlanetaryTools recipe: Wavelet 27/0/0/0 followed
by Adaptive Deconvolution 15.6 with Contrast Adaptive. Sharpened error against
the identically processed noiseless oracle improves in only 7 of 26 cases.
For stationary Saturn with blur, it rises from 0.02882 to 0.03610 in linear
0..1 units, despite eliminating geometric error. For stationary Saturn without
extra blur, it falls from 0.02673 to 0.02065. The previous false warps can
soften noise along with structure, and nonlinear sharpening responds to both.
Thus the method is a motion-validation safeguard, not a denoiser or a uniform
improvement in sharpened error.

## Full Saturn replay

All 5,738 fixed frames were included exactly once, with the same 64-frame
reference, scalar weights and global shifts as the earlier comparisons.
Both independent estimates passed on 2,641 frames; only one passed on 1,816;
neither passed on 1,281 (22.3%), which used global translation. No final averaged
field failed its geometric guard. These counts describe validation decisions,
not discarded frames.

After the user's exact sharpening recipe, without added output filtering:

| Output | Upper disc variation | Lower disc variation | Left ring width | Right ring width |
|---|---:|---:|---:|---:|
| Production PR | 2.9668% | 2.9751% | 8.1417 px | 6.3644 px |
| Previous coherent fit | 2.6267% | 2.7375% | 7.3957 px | 6.0475 px |
| Independent-pixel validated fit | 3.0381% | 3.0304% | 7.3415 px | 6.1185 px |

The validated result increases upper/lower fine-scale variation by 15.7% /
10.7% relative to the previous coherent fit, and by 2.4% / 1.9% relative to
production PR. The fixed left transition is slightly narrower than with the
previous coherent fit, while the right is slightly wider. Visual inspection
agrees that the candidate retains noticeable disc grain. It does not solve
the real-capture grain difference.

Applying the previously measured AS response **only to comparison copies**
gives:

| Output | Upper variation | Lower variation | Left width | Right width |
|---|---:|---:|---:|---:|
| Previous coherent fit | 0.4966% | 0.4932% | 7.1286 px | 5.5950 px |
| Validated fit | 0.5356% | 0.5309% | 7.2489 px | 5.5959 px |
| Global-only PR | 0.5692% | 0.5449% | 7.2842 px | 5.6721 px |
| Supplied AS manual-64 | 0.5732% | 0.6419% | 7.2754 px | 5.6509 px |

Across 75 overlapping left-ring windows, the validated result is narrower
than global alignment in 65 windows, with median difference -0.0932 pixels
and range -0.3066 to +0.0278. The previous coherent fit was narrower in all
75, with median -0.2275. Common-centroid measurements give the same values.
Thus the safeguard retains some of the earlier ring improvement but weakens
its consistency in this comparison. These are sensitivity checks, not
independent replicates or calibrated angular-resolution measurements.

The experiment separates two effects: avoiding false geometric motion can
improve accuracy on known stationary scenes without reducing sharpened grain.
It therefore does not justify promoting this particular guard to production,
nor interpreting the smoother previous result as automatically more accurate.
The practical remaining question is how to validate smaller local corrections
without losing the precision of the full-frame measurements; this test alone
does not settle that question. The full raw and sharpened products and
`comparison.png` remain in `out/saturn-validated-field/`.

## Reproduction and evidence

```sh
PYTHONPATH=. OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python tools/validate_local_warp_centring.py \
  --out out/validated-example --scenes saturn \
  --coherent-spacing 16 --coherent-stiffness .01 --patch-average --validate-pixels
```

Use `--scenes texture`, `--motion-wavelengths 64 96`, or `--seed 9018` for
the other controls. `--validate-pixels` always enables the AP-area model.
Pass generated PNGs through `tools/planetarytools_sharpen_experiment.py` using
the PlanetaryTools environment, then run `tools/analyse_local_warp_validation.py`.

The real replay uses the same fixed 5,738 frames as earlier comparisons:

```sh
PYTHONPATH=. OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python tools/coherent_saturn_experiment.py \
  --out out/saturn-validated-field --start 0 --count 128 \
  --spacing 16 --stiffness .01 --patch-average --validate-pixels
```

Continue with disjoint ranges through position 5,737 and combine using
`--combine --validate-pixels` with the same model arguments. The combiner checks
input and model hashes and requires every selected frame exactly once. It
accumulates weighted sums before dividing by support. A separately named
AS-response comparison copy is generated using the previously measured
sigma-1, radius-3 response; the raw exported candidate remains unfiltered.

`tools/analyse_validated_registration.py` collects the numerical record and
checks that control conditions, noiseless-oracle PNG hashes and the exact
sharpening implementation agree. Results are recorded in
[`results/registration/validated-local-registration.json`](../results/registration/validated-local-registration.json).

All 29 focused tests pass with CUDA enabled. They cover disjoint detector
samples, unchanged scoring observations, blur-only rejection, inverse-coordinate
sign, moving CPU/CUDA agreement, coherent-field geometry, synthetic generation,
the traced matcher, and missing/duplicated/incompatible replay shards. This
prototype evaluates each frame separately and has not been optimized for
production stacking throughput.
