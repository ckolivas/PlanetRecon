# Saturn: independent AP selection and patch recombination

2026-09-27. Controlled experiments prompted by AutoStakkert's Local (AP)
quality setting and separate MAP analysis/recombination stages. No brightness
normalization or output denoising is added. These are tests of specific mechanisms,
not reproductions of AutoStakkert's private implementation.

## Results

**Local selection gives a partial improvement, but none of the fourteen variants
reproduces AutoStakkert's low grain.** The strongest result across the two measured
regions is `patch_local2_registered`: 32.5% and 36.5% less fine-scale variation
than the production stack. It remains 3.49 and 2.94 times the AS measurement.

| Variant | Upper disc variation | Lower disc variation |
|---|---:|---:|
| AS_manual64 | 0.5732% | 0.6419% |
| PR_production | 2.9668% | 2.9751% |
| patch_global | 2.9042% | 2.9574% |
| patch_local2 | 2.4048% | 2.2747% |
| patch_local4 | 2.5710% | 2.5722% |
| patch_global4 | 2.8319% | 2.9802% |
| patch_global_sequential | 2.8852% | 2.9250% |
| patch_local4_sequential | 2.5284% | 2.5588% |
| dense_global | 2.9187% | 2.9567% |
| global_control | 2.9577% | 2.9832% |
| patch_global_registered | 2.5545% | 2.6200% |
| patch_local2_registered | 2.0020% | 1.8879% |
| patch_local4_registered | 2.1508% | 2.2307% |
| patch_global4_registered | 2.4172% | 2.6610% |
| patch_global_sequential_registered | 2.5610% | 2.6038% |
| patch_local4_sequential_registered | 2.1240% | 2.2399% |

Using the same frames at every AP, stacking patches instead of a dense warped
image makes very little difference. Independent local selection is more useful:
a typical AP retains only 50.0% (sigma two) or 57.0% (sigma four) of the original
global frame set. Changing only the global quality estimator produces little
benefit. Separately resampling global and local alignment also produces little
benefit beyond the corresponding single-resampling patch stack.

For local sigma two, overlapping patches produce median frame-weight effective
counts of approximately 7,353 and 8,474 in the two disc regions, compared with
5,694 for the original weighted global selection. This is one concrete mechanism
by which 5,738 frames per AP can yield less grain than one globally chosen set of
5,738 frames. It does not account for the complete AS difference. These counts
include the global fallback and ignore resampling-induced covariance.

Completed-patch registration reduces the measured variation further. Its median
shift is only 0.059 pixels and maximum shift 0.405 pixels for local sigma two;
it also introduces another bilinear interpolation. Its lower grain therefore
must not be interpreted as proof of greater recovered detail or a better
physical reference. No calibrated resolution improvement is claimed.

Visual review caught conspicuous front-ring seams in the initial implementation:
normalizing by a vanishing AP footprint produced a hard jump into the global
fallback. The experiment now uses a continuous fallback contribution as AP
coverage falls below one, and the dense-field comparison has the corresponding
boundary treatment. Tests verify this transition and brightness preservation.
The final local-two, registered local-two and registered sequential local-four
images were inspected; the conspicuous polygonal seams are removed. Earlier
sharpened diagnostics are retained separately under `sharpened-before-edge-fix`.

The floating global control differs from the earlier stack by at most
1.734e-12 ADU. Its sharpened output exactly reproduces every pixel in the user's
`stacking/global_subpixels.png`. All six scientific control tests pass.

No production stacking default has been changed on the basis of this probe.
No normalization or denoising stage has been introduced. The origin of the
remaining AS/PR difference is unresolved, and the experiment does not identify
sensor noise as its source.

The complete machine-readable record is
`results/real-data/saturn-patch-stacking.json`. Full snapshots, PNGs, frame masks,
per-AP weights and translations are in `out/saturn-patch-stacking/`, with final
sharpened images in its `sharpened/` subdirectory.

## Controls

The original R Saturn capture contains 27,689 frames. The experiment considers
all 25,476 frames accepted by the existing preprocessing cache and uses exactly
5,738 observations at each AP. The reference is the unchanged production
64-frame array from `out/saturn-controlled-alignment/reference.npy`.

Forty-eight overlapping circular APs have diameter 65 pixels and grid spacing
32 pixels. This is a single-scale experiment. Template matching uses the existing
production circular matching code, including forward/reverse consistency gates;
a failed local match falls back to the global translation. These unblended AP
vectors are saved so selection and combination variants use the same motions.

The original scalar global quality weights are retained. All image accumulation
uses the original raw samples, with bilinear interpolation and observed-support
accounting. Gaussian smoothing is used only in alignment and quality measurement.
There is no per-frame brightness rescaling, output normalization, or new output
filter. All PNGs use the same previously established display/export mapping.

## Variants

- **Global control:** original global translations and original selected frames.
  The float image must match the earlier global-only result within 1e-8 ADU;
  its sharpened pixels must exactly reproduce the user's global-only output.
- **Dense global:** blend the saved AP vectors spatially, then warp each original
  frame once and average. This isolates dense warping versus combining patch
  images. Its single-scale geometric blend is an experimental control, not an
  exact replay of the production multiscale/confidence blend.
- **Patch global:** same original frame set at every AP; stack each patch with
  its own translation, then combine completed patches through circular footprints.
- **Patch local 2:** each AP independently selects its best 5,738 frames using
  squared Laplacian energy after a Gaussian quality-proxy sigma of two pixels.
- **Patch local 4:** same, using a four-pixel quality proxy. These numeric sigmas
  are experimental pixel scales, not equivalents of AS's Noise Robust numbers.
- **Patch global 4:** select a single global set by averaging the four-pixel
  quality scores of the globally aligned APs. This separates changing the
  quality estimator from choosing different frames at different APs.
- **Sequential versions of patch global and patch local 4:** first interpolate
  each raw frame into the globally aligned coordinates, then interpolate its APs
  using the residual local shifts. These use the same total displacements and
  selections as their single-resampling counterparts. Signal and support follow
  both operations, preserving absolute brightness at partially observed edges.
- **Registered versions of all six patch variants:** align each completed
  patch mean to its corresponding fixed reference patch before combination.
  Shifts above three pixels are rejected. This adds one bilinear resampling of
  the completed patches; any improvement must be considered alongside the
  interpolation's effect on image detail.

Circular footprints have a flat inner region and tapered outer region.
Combination divides by the total contribution weight, so it preserves a
constant absolute brightness. Where AP footprint weights sum to less than one,
the remaining weight comes from the global fallback, providing a continuous
transition into areas without AP support. This mathematical averaging is distinct from
brightness normalization. Outside AP coverage the original global stack is used.
The two grain-measurement regions have 99.3% and 100% AP coverage; coverage of
reference pixels above 15% of peak is 98.08%.

With independent local selection, overlapping APs can contribute different
frame IDs at the same output pixel. Consequently, 5,738 frames per AP can imply
more than 5,738 distinct contributing frames at a blended pixel. The analysis
reports effective counts from the combined frame weights in the two disc regions;
these calculations omit covariance changes from the different resamplings and
are not a full photon-noise model. Snapshot `n_used` records the union of actual
contributing frames, while experiment provenance records the per-AP count.

Each image snapshot labels its coverage as binary observed support, rather than
pretending it is the original stack's exposure map. Per-AP accumulated weights,
selected frame masks, measured shifts, quality scores, and completed patch means
are saved separately for inspection.

## Validation

The focused tests cover independent deterministic selection, missing quality
measurements, preservation of constant absolute brightness through unequal
exposures and overlap, agreement between patch and whole-image resampling for
identical translations, and correction of a known completed-patch displacement.
A known-truth scene with complementary blurred regions verifies that local frame
selection can recover both sharp regions where global selection must compromise.

The exact PlanetaryTools replay calls the actual loader, filters and exporter:
wavelet 27/0/0/0 followed by adaptive deconvolution amount 15.6, contrast adaptive.
Previous controls established pixel-exact reproduction of the user's PR and AS
sharpening. The global control is checked again in this experiment.

Fine-scale variation is measured with the same robust sigma-two high-pass
statistic in the same upper/lower disc regions as the preceding experiments.
It includes real detail and artifacts; it is not a direct measurement of sensor
noise. No new AutoStakkert Global-mode result is generated here.

## Reproduction

```sh
PYTHONPATH=. .venv/bin/python tools/patch_stacking_experiment.py \
  --capture 2024-09-27-1154_3-CK-R-Sat.ser \
  --experiment out/saturn-controlled-alignment \
  --out out/saturn-patch-stacking

PYTHONDONTWRITEBYTECODE=1 \
  ../PlanetaryTools/planetary-app/.venv/bin/python \
  tools/planetarytools_sharpen_experiment.py \
  --planetary-tools ../PlanetaryTools/planetary-app \
  --out out/saturn-patch-stacking/sharpened \
  --verify out/saturn-patch-stacking/global_control.png stacking/global_subpixels.png \
  out/saturn-patch-stacking/*.png

PYTHONPATH=. .venv/bin/python tools/analyse_patch_stacking.py \
  --experiment out/saturn-patch-stacking \
  --as-sharpened AS_F5738/2024-09-27-1154_3-CK-R-Sat_r64_lapl4_ap54s.png \
  --pr-sharpened stacking/locals.png

PYTHONPATH=. .venv/bin/python -m pytest -q tests/test_patch_stacking_experiment.py
```

CUDA is required for this standalone full-capture probe. The first command can
be split using `--stage measure` and `--stage stack`; saved measurements can be
reused for stack-stage investigations. Use a new directory for the sharpening
replay. Large arrays and full images remain in the ignored `out/` directory.
