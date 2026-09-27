# Experimental coherent local registration

The previous [mean-warp experiment](local-warp-validation.md) changed genuine
reference geometry. This experiment instead fits a shared spatial displacement
field to the existing multiscale alignment-point measurements, independently
for each frame. No temporal mean is removed and no raw pixel filter is added.

## Model

The production circular matcher supplies its usual coarse-to-fine measurements,
including its texture, correlation, reverse-consistency and whole-stage gates.
The experimental model composes each accepted AP correction with the preceding
warp before fitting its displacement relative to the fixed global translation.
It uses the unfaded AP measurements, avoiding the assumption that a missing
measurement is evidence of zero displacement.

A cubic spline grid shares information between nearby APs. The fit uses a
second-derivative bending penalty, three robust reweighting solves with a
0.35-pixel vector Huber threshold, and a small 1e-6 coefficient ridge for
otherwise unconstrained regions. The selected experimental grid spacing is
16 pixels and bending strength is 0.01, with the discrete bending penalty scaled
by `(32 / spacing)^2`. These are experimental engineering parameters.

The initial fit treated AP measurements as points at their centres. That
oversmoothed genuine motion with wavelengths 64 and 96 pixels. The revised fit
models each AP as an average over its footprint, weighted by matching-proxy
gradient magnitude squared and the existing circular window. The preceding
warp is averaged over that same footprint when composing the measurement.
This is an approximation to the correlation measurement, not an exact physical
model of atmospheric distortion or a full treatment of directional uncertainty.

Spline extrapolation is tapered to zero outside the reference AP coverage:
full support within one grid spacing of an eligible AP centre, cosine taper
to zero by three spacings. The support is fixed from the reference, not chosen
from each noisy frame. This prevents an unconstrained affine tail in blank
detector space from invalidating an otherwise useful fit.

The existing geometric bounds remain: minimum Jacobian 0.25 and maximum local
displacement six pixels. A rejected fit or insufficient data falls back to
global alignment; the frame still contributes to the stack. Raw pixels are
sampled once with the resulting field. Brightness and frame weights are retained.

## Synthetic results

The controlled scene generation, noiseless-twin scoring and sharpening recipe
are the same as the preceding validation. The selected model was evaluated on
all 14 original scene/condition combinations, four shorter-wavelength cases,
and eight cases with independent noise seed 9018. Model development and these
evaluations together processed 2,688 synthetic frames. The short-wavelength
cases informed model selection; they are not independent validation data.

Clean-image RMS error against the reconstruction using known true motion,
in camera ADU, for the primary seed:

| Scene / condition | Current matcher | Coherent model |
|---|---:|---:|
| Saturn / stationary, noise | 0.2764 | 0.3142 |
| Saturn / stationary, blur and noise | 0.2201 | 0.2422 |
| Saturn / clean motion | 0.2457 | 0.1604 |
| Saturn / motion, blur and noise | 0.2551 | 0.2419 |
| Saturn / true offset, blur and noise | 0.7957 | 0.2627 |
| Textured disc / clean motion | 0.1692 | 0.0666 |
| Textured disc / motion, blur and noise | 0.2327 | 0.0959 |
| Textured disc / true offset, blur and noise | 0.2695 | 0.1090 |
| Saturn / shorter-scale clean motion | 0.3324 | 0.2621 |
| Textured disc / shorter-scale clean motion | 0.1751 | 0.0762 |

The selected model improves clean-image error in 12 of the 14 original cases
and all four short-wavelength cases. The two stationary Saturn cases regress,
and the independent noise repeat retains that trade-off. A lower error on
moving scenes therefore does not establish a universally better alignment.

After the exact user sharpening recipe (Wavelet 27/0/0/0, then Adaptive
Deconvolution 15.6 with Contrast Adaptive), all seven primary Saturn cases
improve against their identically sharpened noiseless oracle. For example,
clean-motion error falls from 0.02324 to 0.00614 in linear 0..1 units, and the
offset-motion error from 0.02865 to 0.02208. However, both stationary textured-disc
cases have slightly worse sharpened error, as does its short-wavelength noisy
case. Raw geometric error and nonlinear sharpened error are different metrics;
neither is a pure sensor-noise measurement.

## Real Saturn replay

The replay uses exactly the existing 5,738 frame IDs, scalar quality weights,
global translations and 64-frame reference. Bounded shards store weighted pixel
sums and sample support. The combiner verifies input/model hashes and requires
every selected frame exactly once before dividing the accumulated sums by
support. It never averages separately normalized shard images.

The unfiltered output is `out/saturn-coherent-field/coherent.png` and its floating
snapshot is `coherent.npz`. A separately labelled `coherent_as_transfer.png`
receives only the previously measured AS sigma-1, radius-3 response, for the
same comparison used in earlier experiments. Both receive the user's exact
sharpening recipe. No normalization is introduced.

The replay retained all 5,738 frames. Of these, 254 (4.43%) had a fitted field
outside the geometric bounds and used global translation instead. There were
no insufficient-measurement fallbacks; the median fit used 72 accepted AP
measurements. These are overlapping measurements, not independent observations
or extra contributions of frame pixels.

After sharpening, the **unfiltered** coherent stack reduces upper/lower disc
fine-scale variation from 2.9668% / 2.9751% to 2.6267% / 2.7375%, improvements
of about 11.5% / 8.0%. Fixed left/right ring transition widths change from
8.1417 / 6.3644 pixels to 7.3957 / 6.0475 pixels. This does not close the full
unfiltered PR-versus-AS grain difference; the earlier measured AS transfer
response remains relevant.

With the measured AS response applied to PR comparison copies before the same
sharpening:

| Output | Upper variation | Lower variation | Left edge width | Right edge width |
|---|---:|---:|---:|---:|
| Current local PR | 0.5644% | 0.5401% | 7.8187 px | 5.8148 px |
| Coherent field | 0.4966% | 0.4932% | 7.1286 px | 5.5950 px |
| Global-only PR | 0.5692% | 0.5449% | 7.2842 px | 5.6721 px |
| Supplied AS manual-64 | 0.5732% | 0.6419% | 7.2754 px | 5.6509 px |

Across the 75 paired left-ring windows, the current local model is broader
than global alignment by a median 0.5295 pixels; the coherent model is narrower
by a median 0.2275 pixels. All 75 windows show the latter direction, with the
difference ranging from -0.5030 to -0.0375 pixels. Using a common PR centroid
gives exactly the same result, so this improvement is not a rounded-centroid
window shift. The right-ring median difference relative to global is -0.1069
pixels, with a range crossing zero. The windows overlap and are not independent
statistical replicates.

Visual inspection of the comparison supports cleaner ring transitions and
less disc grain, but widths include physical geometry and processing. A
narrower transition than AS does not prove higher astronomical resolution.
The stationary synthetic regressions and lack of a second real capture mean
this remains an experimental candidate, not a replacement production default.

## Reproduction

```sh
PYTHONPATH=. OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python tools/validate_local_warp_centring.py \
  --out out/coherent-validation --scenes saturn \
  --coherent-spacing 16 --coherent-stiffness .01 --patch-average
```

Use `--scenes texture` for the second scene, `--motion-wavelengths 64 96` for
shorter-scale motion, and `--seed 9018` for the noise repeat. The checked-in
numerical record identifies all run directories and parameter choices.

```sh
PYTHONPATH=. OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python tools/coherent_saturn_experiment.py \
  --out out/saturn-coherent-field --start 0 --count 512 \
  --spacing 16 --stiffness .01 --patch-average
```

Repeat nonoverlapping ranges through index 5,737, then use the same command
with `--combine`. Pass the two exported PNGs to the existing
`planetarytools_sharpen_experiment.py` harness using the PlanetaryTools Python
environment and a new `sharpened` subdirectory. Analyse with:

```sh
QT_QPA_PLATFORM=offscreen PYTHONPATH=. .venv/bin/python \
  tools/analyse_coherent_saturn.py --out out/saturn-coherent-field
```

These are diagnostic tools; production alignment and GUI defaults are unchanged.

## Verification and evidence

All 23 focused tests pass, including moving and static CPU/CUDA checks, affine
geometry preservation, outlier resistance, AP-area integration, missing-data
fallback, and protection against missing, duplicate or incompatible replay
shards. The generated oracle PNGs match their baseline counterparts byte for
byte, and the sharpening implementation and parameters match the previously
verified PlanetaryTools recipe.

The numerical record is
[`results/registration/coherent-local-registration.json`](../results/registration/coherent-local-registration.json).
The real comparison is `out/saturn-coherent-field/comparison.png`; full float
stacks, unfiltered and comparison PNGs, per-frame fit statistics and source
reports remain in `out/`. Runtime logs are from overlapping diagnostic jobs,
not a controlled performance benchmark.
