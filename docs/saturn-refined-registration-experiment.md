# Saturn: global peak accuracy, interpolation and local ring softening

Following the measured AS smoothing comparison, this experiment tests whether
global shift fitting or bilinear interpolation explains the remaining detail
difference. It also retains the prior production-local result as an external
control. No production algorithm or setting was changed.

## Controlled four-way experiment

All four cells use the same 5,738 raw frames, scalar quality weights, fixed
64-frame reference image and export mapping. All use global translation only:

1. Existing independent-axis three-point peak fit, bilinear interpolation.
2. Continuous two-dimensional Fourier correlation maximum, bilinear interpolation.
3. Existing peak fit, cubic spline interpolation.
4. Continuous correlation maximum, cubic spline interpolation.

The continuous fit maximizes the same Gaussian-weighted amplitude correlation
as production, using a joint Hessian including mixed-axis curvature. It changes
only how the subpixel peak is located. Bounded Newton iterations fall back to
the original shift if curvature or convergence checks fail.

CUDA computes batched FFTs and peak refinements; 16 CPU workers interpolate
bounded batches. SER reads and accumulation retain capture order. Cubic output
uses conservative interior support; neither method rescales frame brightness.
Raw and alternating-half stacks are retained alongside diagnostics.

Every displayed PR variant receives the measured AS Gaussian response (sigma 1,
radius 3) before the exact PlanetaryTools recipe: wavelet 27/0/0/0 followed by
contrast-adaptive deconvolution 15.6. The supplied manual-64 AS output and the
prior production-local comparison remain unchanged.

## Validation

Eight focused tests pass, including known Fourier translations on both CPU
and CUDA, flat-frame fallback, preservation of constant brightness, and a
known translated fine pattern. The continuous fit recovers the noiseless test
translations to within 1e-8 pixels. Cubic substantially reduces interpolation
error on the synthetic fine pattern; this does not imply an equivalent gain
with noisy, seeing-distorted observations.

The complete capture run reproduces the previous global shifts exactly and the
previous global floating stack to within 1.28e-13 ADU. All 5,738 refinements
converge. Changes to the original shifts have magnitudes:

| Percentile | Change (pixels) |
| --- | ---: |
| Median | 0.00318 |
| 90th | 0.00881 |
| 99th | 0.01742 |
| Maximum | 0.02730 |

The existing global shift estimator is therefore already close to the more
precise maximum of this objective. This does not prove that global translation
models the atmosphere correctly.

## Real-image results

Disc variation is the same robust high-pass statistic used previously; it
includes detail and artifacts. Ring width is a descriptive 10–90% transition
across the upper edge of each ansa, averaging a fixed 30-column window. It
includes physical ring geometry, seeing and processing; it is not a calibrated
resolution measurement. Smaller widths describe a steeper transition.

| Result after sharpening | Upper variation | Lower variation | Left ring width | Right ring width |
| --- | ---: | ---: | ---: | ---: |
| Production local + AS smoothing | 0.5644% | 0.5401% | 7.819 px | 5.815 px |
| AS manual-64 | 0.5732% | 0.6419% | 7.275 px | 5.651 px |
| Existing shifts / bilinear | 0.5692% | 0.5449% | 7.284 px | 5.672 px |
| Refined shifts / bilinear | 0.5679% | 0.5423% | 7.281 px | 5.681 px |
| Existing shifts / cubic | 0.6177% | 0.6041% | 7.266 px | 5.633 px |
| Refined shifts / cubic | 0.6165% | 0.5996% | 7.267 px | 5.632 px |

The six-image comparison was visually inspected. Refined versus existing
global shifts have little visible effect. Cubic modestly increases both fine
contrast and grain; it makes little difference to these edge widths. Neither
experiment demonstrates a substantial practical improvement over the existing
global stack on this capture.

The production-local result is visibly different around the left rings. The
global-only result is closer to the AS edge width at similar disc variation.
This narrows the remaining investigation toward local warp behaviour instead
of the precision of the global peak estimator.

## Robustness of the local/global finding

The edge measurement was repeated over 75 paired windows on each side, varying
the crop centre by ±2 pixels in each axis and the ansa distance across
165/175/185 pixels. The peak search was also extended five pixels beyond the
ring centre to avoid clipping the selected maximum. These overlapping windows
are a sensitivity check, not independent statistical replicates.

* Left: local alignment gives a broader edge in all 75 windows. Local minus
  global width ranges from 0.142 to 1.034 pixels, median **0.529 pixels**.
* Right: local minus global ranges from −0.182 to +0.345 pixels, median −0.028;
  only one third of windows are broader. There is no consistent degradation
  on this side.

Consequently the supported finding is **local alignment softens the tested
left ring edge in this capture**. It is not a general claim that local alignment
is harmful, nor proof of an AP confidence-gate bug. Alignment accuracy,
reference distortion and field blending remain possible causes. The prior
local/global grain test did not detect this because grain and edge shape are
different measurements.

The next focused code investigation is the local corrections and their blending
around the left ring. Retain the existing global algorithm as the control;
these results do not justify replacing it or adopting cubic solely for lower
synthetic interpolation error.

## Artifacts and reproduction

* `tools/refined_registration_experiment.py`: synthesis tests aside, complete
  four-way real-data experiment; optional `--limit 128 --out NEW_DIRECTORY`
  makes a pilot.
* `tests/test_refined_registration_experiment.py`: CPU/CUDA known-input checks.
* `tools/analyse_refined_registration.py`: common sharpened-image comparison
  and edge-window sensitivity analysis.
* `out/saturn-refined-registration/`: floating outputs, half stacks, shifts,
  AS-transfer PNGs, sharpened PNGs, reports and `comparison.png`.
* `results/real-data/saturn-refined-registration.json`: tracked measurements,
  input hashes and sharpening provenance.

Run the experiment with `PYTHONPATH=.` in the project environment and CUDA
access. Process its four `*_as_transfer.png` outputs using
`tools/planetarytools_sharpen_experiment.py` in the PlanetaryTools environment,
writing to the experiment's `sharpened/` directory. Run the analyser with
`QT_QPA_PLATFORM=offscreen PYTHONPATH=.`. Output preparation requires a new
directory; original captures and supplied AS images are untouched.
