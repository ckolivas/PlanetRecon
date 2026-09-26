# Saturn local-warp diagnosis

The preceding global/local comparison found consistent broadening of the left
ring edge with production local alignment. This experiment traces the individual
AP measurements and the composed displacement field to separate the cause.
It preserves the original 5,738-frame selection, scalar quality weights and
64-frame reference. Detector brightness is never normalized.

## Method

`tools/local_warp_trace.py` mirrors the resident CUDA matcher using the same
sampling, NCC, peak-gating and correlation implementations. Tests compare every
stage with the production matcher on CPU and CUDA. The real run also compares
sampled final fields and checks that its final stack reproduces the previously
saved production-local stack.

The first pass saves the stack after each scale: 67, 47, 33 and 23 pixels.
For each AP it records the forward residual, accepted status, correlation peak,
confidence and reverse distance. It also saves the weighted mean and standard
deviation of the final displacement field and traces seven fixed points across
the rings and globe. Per-AP residuals use the preceding warped coordinate
system; they are not directly interchangeable with final sensor displacements.

The second pass uses the unchanged production matcher and compares:

* Original local displacement.
* Only its weighted mean residual added to each frame's global translation.
* Its frame-dependent residual with the weighted mean removed.
* Spatially smoothed local displacement, Gaussian sigma 11.5 pixels: half the
  smallest AP diameter. This changes coordinates only, not recorded pixels.

The same frames and weights determine the mean and the ablations. Each raw
frame is resampled once for each variant. Mean removal is a diagnostic change
of the average warp, not a claim to recover the true astronomical geometry.
Spatial regularization can suppress both unwanted and genuine local motion.

For comparison, all PR outputs receive the previously measured AS image
smoothing (Gaussian sigma 1, radius 3), then the actual PlanetaryTools recipe:
wavelet 27/0/0/0 and contrast-adaptive deconvolution 15.6. This image smoothing
is distinct from the displacement-map regularization. Both are confined to
diagnostic outputs; production defaults remain unchanged.

The ring edge metric and its 75-window sensitivity check are reused from the
preceding experiment. Width includes physical geometry and all processing;
it is not a calibrated resolution measurement. Fine-scale disc variation
includes detail and artifacts and should not be interpreted as pure noise.

## Reproduction

Use a new output directory for each preparation run. GPU access is required.

```sh
PYTHONPATH=. .venv/bin/python tools/local_warp_trace.py
PYTHONPATH=. .venv/bin/python tools/local_warp_trace.py \
  --centre out/saturn-local-warp-trace --smooth-field 11.5 \
  --out out/saturn-local-warp-components
```

Process each run's `*_as_transfer.png` files with
`tools/planetarytools_sharpen_experiment.py` using the PlanetaryTools environment,
writing into that run's `sharpened/` directory. Then run:

```sh
QT_QPA_PLATFORM=offscreen PYTHONPATH=. .venv/bin/python tools/analyse_local_warp.py
QT_QPA_PLATFORM=offscreen PYTHONPATH=. .venv/bin/python tools/analyse_local_warp.py \
  --out out/saturn-local-warp-components
PYTHONPATH=. .venv/bin/python -m pytest -q tests/test_local_warp_trace.py
```

Small independent pilots use `--limit 64` and new output directories. A
second-pass pilot must use the corresponding first-pass pilot's mean field;
the script verifies that the frame indices and weights match.

## Scale tracing results

The complete first pass reproduces the prior production-local floating stack
within 1.77e-12 ADU. Its sharpened final-stage PNG is byte-identical to the
previous production-local comparison. Every stage passes the field guard for
the 67/47-pixel scales, 99.98% for 33 pixels and 96.31% for 23 pixels.

| Accumulated stages | Left edge width | Right edge width | Upper disc variation | Lower disc variation |
| --- | ---: | ---: | ---: | ---: |
| Global only | 7.284 px | 5.672 px | 0.5692% | 0.5449% |
| Through 67 px | 7.472 px | 5.663 px | 0.5687% | 0.5457% |
| Through 47 px | 7.472 px | 5.663 px | 0.5634% | 0.5417% |
| Through 33 px | 7.467 px | 5.607 px | 0.5641% | 0.5403% |
| Through 23 px (production final) | 7.819 px | 5.815 px | 0.5644% | 0.5401% |
| Supplied AS manual-64 | 7.275 px | 5.651 px | 0.5732% | 0.6419% |

The final 23-pixel stage adds most of the measured left-edge broadening. Across
the 75-window sensitivity check, the median excess width over global alignment
is 0.109/0.093/0.095/0.529 pixels at the four successive stages. The first three
stages are not broader in every window; the final stage is broader in all 75.

At the fixed left-ring point `(x=165, y=194)`, the final stage changes horizontal
displacement standard deviation from 0.034 to 0.499 pixels and vertical standard
deviation from 0.024 to 0.331 pixels. Its mean vertical displacement is −0.147
pixels. At the nearby point `(175,193)`, the final displacement remains near
zero. This illustrates how local corrections vary across the ring; it does not
by itself establish whether each individual correction is erroneous.

After both AP and whole-field gates, the accepted fractions across eligible
APs are 28.17%, 21.74%, 17.36% and 14.14% for the successive scales. The small
scale therefore contributes sparse corrections, with parent/global motion
retained elsewhere. Low acceptance alone is not a defect.

## Zero-motion, changing-blur control

`tools/zero_motion_blur_probe.py` adds spatially uniform Gaussian blur to the
fixed reference, without moving it or adding independent noise. The reference
retains its existing texture/noise. The known geometric displacement is zero.
The local matcher is supplied that exact zero global displacement.

| Added blur sigma | Maximum inferred local motion in left-ring region |
| --- | ---: |
| 0 px | 0 px |
| 0.5 px | 0.376 px |
| 1.0 px | 1.051 px |
| 1.5 px | 2.824 px |
| 2.0 px | 0.883 px |

The global estimator stays within 5e-8 pixels of zero. Local motion decreases
again at sigma 2 because AP acceptance changes; it is not a monotonic measure
of blur. At `(165,194)`, sigma 0.5 alone induces a −0.140-pixel vertical
correction, close to the real run's −0.147-pixel mean at that point.

This establishes that appearance changes caused by blur can trigger local
geometry corrections in the current matcher. It does not require newly added
random noise. It provides a plausible contributor to the real-data bias, not
proof that all real local motion is caused by blur or that a particular AP is
always unreliable. Uniform Gaussian blur is a controlled probe, not a complete
model of atmospheric seeing.

The reproducible control is recorded in
`out/saturn-local-warp-trace/zero_motion_blur_verified.json`. Five focused tests
validate every traced stage against production on CPU and CUDA, along with
the displacement-only smoother's preservation of global translation and axis
independence.

## Average distortion versus varying motion

The complete second pass also reproduces the original local floating stack
within 1.77e-12 ADU and its sharpened control byte-for-byte. Each alternative
still uses all 5,738 frames and identical weights and sharpening.

| Displacement used | Fixed left width | Fixed right width | Median excess left width over global, 75 windows |
| --- | ---: | ---: | ---: |
| Original local | 7.819 px | 5.815 px | +0.529 px |
| Mean residual only | 7.595 px | 5.618 px | +0.351 px |
| Original minus mean residual | 7.549 px | 5.917 px | +0.023 px |
| Spatially smoothed residual | 7.520 px | 5.731 px | +0.096 px |
| Global only | 7.284 px | 5.672 px | 0 |

The fixed-window and sensitivity values answer slightly different questions:
the latter varies the averaging window and extends the peak search past the
ring centre. An edge transition measured to a maximum clipped at the centre
can react to a shape change differently from one measured to the complete
nearby maximum. Both are retained, rather than selecting the most favorable
statistic. Neither is a calibrated resolution measurement.

The average residual alone retains a persistent left-edge change: all 75
windows remain broader than global. Removing that average instead reduces the
median excess from 0.529 to 0.023 pixels (range −0.153 to +0.263). The variation
between frames still changes individual profiles, but the strongest persistent
left-edge bias is substantially reversed by removing the mean warp. The right
edge has no consistent degradation over the sensitivity windows, even though
the fixed right window becomes broader under mean removal.

Spatial displacement smoothing reduces the fixed-window excess by about 56%
and the median sensitivity-window excess by about 82%. This confirms that the
spatial structure of the field matters, but it does not identify a universally
correct smoothing scale or prove that genuine small-scale motion survives.

All alternatives have similar sharpened disc variation (upper 0.561–0.569%,
lower 0.538–0.543%). Their edge differences are not accompanied by a large
change in this grain statistic. Both comparison montages were inspected; the
edge changes are visible but small, and do not establish equal overall image
quality to AS.

## Conclusion and implementation boundary

There is direct evidence of two relevant behaviours:

1. The local matcher produces geometry corrections for a blur-only appearance
   change with known zero motion.
2. The average local field introduces a systematic shape bias around the left
   ring; subtracting it substantially reverses the median measured broadening
   on this capture. Sparse small-patch refinement adds most of that broadening.

These observations make blur-sensitive matching and the choice of average
reference geometry concrete development targets. They do not establish that
all local shifts are spurious, that atmospheric blur is globally Gaussian, or
that zero mean is the correct coordinate constraint for every capture. A fixed
reference and a mean-frame coordinate system need not describe the same pose.

The zero-mean output is a useful candidate for further tests with known smooth
local deformations, changing blur and independent noise, followed by another
real capture. A production change should retain real motion while reducing
blur-induced motion; simply adding output blur would not address this defect.
No production settings or algorithms are changed by these experiments.

Tracked reports are `results/real-data/saturn-local-warp-trace.json` and
`results/real-data/saturn-local-warp-components.json`. The full images and arrays
are in the corresponding `out/` directories. The mean-removed result is
`out/saturn-local-warp-components/sharpened/zero_mean_local_as_transfer_sharpened.png`.
