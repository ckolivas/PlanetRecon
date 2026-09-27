# Elongated seeing blur mistaken for motion

The corrected joint fitter improved held-out detector prediction while making
the complete Saturn stack worse. This experiment tests one specific model
limitation: its additional blur is circular, although a change in image shape
could instead be caused by an elongated point-spread function (PSF).

## Independent known-truth controls

Two continuous scenes contain broad Gaussian structure and either resolved or
fine features. The observations are area integrals of those continuous
features, after convolution with a normalized Gaussian PSF and optional known
invertible shears. No spline, FFT reference blur or fitter renderer generates
the observations. Four-point and eight-point quadrature are compared at every
pose. Circular special cases are checked against independent exact erf-based
pixel integrals; rotated PSFs are checked for flux and second moments.

Each scene has twelve frames in four conditions: circular blur with no motion
or noise; elongated blur without motion or noise; elongated blur with noise but
no local motion; and elongated blur with local motion and noise. All have known
fractional global translations. The elongated PSF has principal-axis sigmas
1.8 and 0.5 pixels, with its orientation changing across frames. Added noise
is independent Gaussian sigma 2 ADU, a simulation parameter rather than a
measurement of the Saturn camera noise.

The estimators are:

1. The existing corrected joint circular-blur/motion fit.
2. An elliptical Gaussian fitted at the global pose using only the same
   even-row/even-column training pixels, then held fixed during motion fitting.
3. A diagnostic given the true PSF-matched reference from the continuous scene.
4. A separate ablation fitting a circular Gaussian at the global pose and
   holding it fixed too. This separates changing the PSF shape from changing
   the order in which blur and motion are fitted.

The estimated blur modifies only the reference templates. The field geometry,
support, 100-iteration motion budget, stiffness 0.3, raw frame values and final
bilinear sampling remain fixed. Photometric gain and offset affect fitting
scores only. Original and noiseless twin frames are accumulated with identical
estimated fields to distinguish geometric error from added-noise effects.
Known motion supplies the output target; no blur is removed from that target.

## Mechanism confirmed

On the resolved stationary scene, a circular-blur model invents approximately
0.075 pixels RMS of local motion when the only real change is elongated blur.
An estimated elliptical PSF reduces this to approximately 0.00008 pixels.
Holding a fitted circular PSF fixed leaves the 0.075-pixel error essentially
unchanged. Thus this example is specifically about PSF shape, rather than
merely freezing a nuisance parameter before fitting motion.

The held-out gate accepts the wrong circular-model motion on all twelve
noiseless elongated-blur frames. The very small residual interpolation errors
of the correctly matched models can also pass the gate on noiseless data; an
acceptance alone does not establish physically meaningful motion. With added
noise, the resolved stationary control admits three false circular-model
fields and no elliptical-model fields, while both admit all genuinely moving
frames. Complete results and the finer-feature scene are recorded below.

This demonstrates a failure mechanism even without independent sensor noise.
It does not prove that elongated Gaussian blur is the dominant cause of the
real Saturn discrepancy, or that all apparently granular structure is noise.

All 96 control frames completed. Field errors below are vector RMS pixels,
averaged over frames and a fixed object mask. They describe the fitted fields
before optionally rejecting them with the held-out gate.

| Scene and condition | Joint circular | Fixed circular | Estimated ellipse | Known-PSF diagnostic |
|---|---:|---:|---:|---:|
| Resolved, circular, stationary, noiseless | 0.000075 | 0.000075 | 0.000075 | 0.000079 |
| Resolved, elongated, stationary, noiseless | 0.074871 | 0.074902 | 0.000079 | 0.000085 |
| Fine, elongated, stationary, noiseless | 0.057924 | 0.057953 | 0.001526 | 0.001917 |
| Resolved, elongated, stationary, noisy | 0.135374 | 0.135339 | 0.111799 | 0.112244 |
| Fine, elongated, stationary, noisy | 0.149829 | 0.149421 | 0.139288 | 0.140636 |
| Resolved, elongated, moving, noisy | 0.350223 | 0.351139 | 0.328858 | 0.339459 |
| Fine, elongated, moving, noisy | 0.432495 | 0.431835 | 0.409774 | 0.416841 |

In the fine stationary noisy scene, the circular model accepts one false field
out of twelve and the elliptical model accepts none. Both accept every moving
frame in that scene too. The maximum four-point/eight-point integration
difference is 2.50e-6 ADU. All 96 elliptical PSF profiles report convergence.
Their covariance error grows when real local motion is present, as expected
for a PSF estimated under global alignment only.

Better geometry does not automatically improve the noisy sharpened result.
For the two moving scenes, elliptical matching reduces clean-twin image RMSE
from 0.02974 to 0.02532 ADU and from 0.05075 to 0.02726 ADU. However, sharpened
RMSE against the noiseless known-motion output changes from 0.02049 to 0.02062
and from 0.01797 to 0.01807 in linear 0–1 units: slight increases. These
twelve-frame tests are not evidence of improved final stack quality.

## Real-capture pilot and limits

A fixed, evenly spaced subset of 128 positions from the established 5,738-frame
selection uses the same reference, quality weights, global translations and
export mapping for every method. All variants use the actual PlanetaryTools
Wavelet 27/0/0/0 then Adaptive Deconvolution 15.6, Contrast Adaptive recipe.
The raw SER is hashed before and after the main pilot. Analysis independently
reconstructs the global baseline using SciPy's bilinear sampler and verifies
input, source, frame association and sharpening identities.

The ellipse is fitted at the global pose and held fixed; it is not a joint
spatially varying PSF fit. Local motion can therefore bias that PSF estimate.
Its Cholesky bounds also allow broader blur than the original sigma-2 circular
bank. The fixed circular ablation allows sigma up to 3. These bounds must be
kept in mind when interpreting the real-capture scores; the independent
elongated controls have major sigma 1.8, inside every relevant range.

The 128-frame pilot is substantially noisier than the complete stack. Its
ring-width and fine-variation statistics are descriptive and cannot replace a
full-count comparison. No AutoStakkert full stack is used as its comparator.
No production changes, output filtering or brightness normalization are made.

All 128 pilot frames and the fixed-circular comparison completed. The global
stack matches independent SciPy accumulation to 5.69e-14 ADU. Every export
received the exact sharpening recipe, and the gallery was visually inspected.

| Method | Upper-disc variation (%) | Lower-disc variation (%) | Left ring width (pixels) | Right ring width (pixels) |
|---|---:|---:|---:|---:|
| Global translations | 30.006 | 30.351 | 8.793 | 4.625 |
| Coherent AP | 28.506 | 31.685 | 8.576 | 6.254 |
| Joint circular blur | 30.962 | 31.525 | 9.154 | 5.793 |
| Fixed circular blur | 30.764 | 31.901 | 9.177 | 7.562 |
| Estimated elliptical blur | 32.539 | 30.211 | 9.089 | 7.547 |

The changes are mixed, without a convincing overall improvement. All circular
and elliptical joint fields pass the held-out gate, despite the poor sharpened
appearance. Their median held-out MSE changes only from 66.274 to 66.230 ADU².
Median fitted residual motion changes from 1.036 to 1.015 pixels. Every joint
fit again exhausts its 100-iteration budget; 127 of 128 global elliptical
profiles report convergence. Estimated principal-axis sigmas have medians
0.990 and 2.375 pixels, so the larger permitted blur range is active here.

**Decision:** the independent controls confirm that PSF mismatch can produce
apparently validated false motion without sensor noise. Elliptical template
matching addresses that particular failure, but this sequential global-PSF
method has not demonstrated a better sharpened Saturn result. Keep it
experimental. A spatially uniform Gaussian PSF, even elliptical, remains a
limited seeing model; these results do not establish which additional model
would improve the real capture.

The complete numerical/provenance record is
[`elliptical-blur-probe.json`](../results/registration/elliptical-blur-probe.json).
The same-count pilot gallery is `out/saturn-elliptical-pilot/comparison.png`.

## Reproduction

```sh
PYTHONPATH=. OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python tools/elliptical_blur_probe.py --out out/elliptical-blur-controls
PYTHONPATH=. OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python tools/elliptical_saturn_probe.py --out out/saturn-elliptical-pilot
PYTHONPATH=. OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python tools/fixed_circular_blur_probe.py --out out/fixed-circular-blur-controls
```

Use fresh output directories. The fixed-circular run requires both earlier
reports to be complete. Sharpen every exported PNG in all three directories
using `tools/planetarytools_sharpen_experiment.py` in the PlanetaryTools Python
environment. Then run `tools/analyse_elliptical_blur.py`. Exact recipe and
implementation hashes are checked against the previous experiment.

`tests/test_elliptical_blur_probe.py` checks independent integral equivalence,
flux/covariance, recovery of known blur and exclusion of held-out pixels from
both global PSF estimators. Six focused tests pass.
