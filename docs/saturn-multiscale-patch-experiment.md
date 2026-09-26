# Saturn: independent multiscale patch stacking

2026-09-27. This tests the specific mechanism suggested by the second
AutoStakkert screenshot: independently select and stack frames at several AP
sizes, then combine the resulting patch images. It differs from PR's production
multiscale registration, which constructs one displacement field before sampling
each original frame once.

## Results

**Multiscale combination gives only a marginal additional reduction beyond the
previous best single-scale patch experiment. It does not explain the AS/PR gap
in this test.**

The best combined-scale result is `multiscale_local2_registered`, with disc
variation of **1.9606% / 1.8697%**, versus **2.0020% / 1.8879%** for the previous
65-pixel result. That is only **2.07% / 0.96%** less variation. It remains
**3.42 / 2.91 times** the supplied AS result of **0.5732% / 0.6419%**.

Without the extra completed-patch registration, independent multiscale selection
gives 2.1217% / 2.1180%, compared with 2.4048% / 2.2747% at 65 pixels alone.
Thus combining scales does have an effect, but it largely overlaps with benefits
already obtained by the preceding experiment rather than providing the missing
large reduction.

The 55-AP control gives 2.0947% / 2.2701% for the corresponding registered
local-two variant. Matching the displayed AP count does not reproduce AS's low
grain. This is a count control, not an exact copy of AS's placement algorithm.

Median effective counts from frame weights increase from approximately
7,353 / 8,474 in the preceding local-two single-scale layout to **8,925 / 9,890**
with all four scales. The sparse 55-AP variant gives approximately **8,593 / 6,922**.
Placement and overlap therefore affect frame diversity differently in the two
regions. These counts omit covariance changes due to subpixel interpolation.

Larger APs pass the existing local matching gates more often: **27.71%, 31.40%,
and 34.83%** at 93, 131 and 185 pixels, respectively, versus 20.84% at 65 pixels.
These fractions demonstrate changed matching behavior; they are not independent
proof that every accepted displacement is physically correct.

The full layout, sparse layout and 185-pixel local-two registered outputs were
visually inspected. The conspicuous support-boundary seams from the earlier
prototype are absent, but substantial grain remains. No calibrated improvement
in resolution is claimed. The remaining AS/PR difference remains unresolved.

All six 65-pixel floating controls reproduce the preceding experiment to within
**2.843e-14 ADU**, and all six sharpened controls reproduce it pixel-for-pixel.
The PR production comparison uses the root `2024-09-27-1154_3-CK-R-SatPs.png`,
verified byte-identical to the previously used `stacking/locals.png` (that folder
was no longer present during this run). The 15 focused controls pass, including
all nine new CUDA/geometry/edge tests. No production defaults were changed.

### Complete sharpened measurements

| Variant | Upper disc variation | Lower disc variation |
|---|---:|---:|
| PR_production | 2.9668% | 2.9751% |
| AS_manual64 | 0.5732% | 0.6419% |
| scale65_global | 2.9042% | 2.9574% |
| scale65_global_registered | 2.5545% | 2.6200% |
| scale65_local2 | 2.4048% | 2.2747% |
| scale65_local2_registered | 2.0020% | 1.8879% |
| scale65_local4 | 2.5710% | 2.5722% |
| scale65_local4_registered | 2.1508% | 2.2307% |
| scale93_global | 2.9294% | 2.8950% |
| scale93_global_registered | 2.8214% | 2.7758% |
| scale93_local2 | 2.6771% | 2.5227% |
| scale93_local2_registered | 2.5686% | 2.2875% |
| scale93_local4 | 2.6531% | 2.5713% |
| scale93_local4_registered | 2.4841% | 2.4922% |
| scale131_global | 2.9024% | 2.8968% |
| scale131_global_registered | 2.6797% | 2.7198% |
| scale131_local2 | 2.5585% | 2.6291% |
| scale131_local2_registered | 2.3048% | 2.5063% |
| scale131_local4 | 2.6019% | 2.6369% |
| scale131_local4_registered | 2.4027% | 2.4762% |
| scale185_global | 2.6296% | 2.5764% |
| scale185_global_registered | 2.5938% | 2.4985% |
| scale185_local2 | 2.3381% | 2.2559% |
| scale185_local2_registered | 2.3039% | 2.1873% |
| scale185_local4 | 2.4652% | 2.3429% |
| scale185_local4_registered | 2.4536% | 2.2636% |
| multiscale_global | 2.7322% | 2.7498% |
| multiscale_global_registered | 2.5621% | 2.5426% |
| multiscale_local2 | 2.1217% | 2.1180% |
| multiscale_local2_registered | 1.9606% | 1.8697% |
| multiscale_local4 | 2.3199% | 2.1951% |
| multiscale_local4_registered | 2.1238% | 2.0768% |
| multiscale55_global | 2.9073% | 2.9742% |
| multiscale55_global_registered | 2.7227% | 2.7253% |
| multiscale55_local2 | 2.2997% | 2.5790% |
| multiscale55_local2_registered | 2.0947% | 2.2701% |
| multiscale55_local4 | 2.4552% | 2.6820% |
| multiscale55_local4_registered | 2.2013% | 2.4450% |

Full images, floating arrays, quality measurements and selected-frame masks are
in `out/saturn-multiscale-patches/`; final sharpened images are in `sharpened/`.
The tracked machine-readable report is
`results/real-data/saturn-multiscale-patches.json`.

## Controls and scope

- Same R Saturn capture, fixed production 64-frame reference, and original
  global translations as the preceding experiment.
- Same 25,476 screened frames and exactly 5,738 selections at every AP.
- Same scalar global quality weights and absolute frame brightness.
- No brightness normalization or output denoising.
- Circular APs with diameters 65, 93, 131 and 185 pixels, following PR's
  four-scale progression, with spacing equal to half the diameter rounded down.
- Independent matching against the fixed reference at every scale. These
  patch translations are not merged into one dense displacement field.
- Same local matching confidence/reverse-consistency gates. An unreliable local
  match falls back to the saved global translation. A fully observed patch
  remains eligible for quality selection even when its wider search margin
  crosses the detector boundary.
- All accumulation uses original detector samples, with one bilinear pull per
  contributing patch/frame, followed by a weighted average. Proxy smoothing is
  used only to measure quality and alignment.

The scale grids contain 48, 36, 22 and 14 eligible APs, respectively: **120 total**.
A separate **55-AP control** uses spatially spread subsets of 22, 15, 10 and 8 APs.
It is derived from the same saved patch stacks and therefore changes only which
patches contribute to the final image. It does not claim to reproduce the exact
positions, shapes or size distribution in the user's AS screenshot.

The screenshot shows a base AP size of 64, Multi-Scale and Close to Edge enabled,
and 55 APs. The supplied AS comparison file is labelled `ap54`; the screenshot
is evidence for the multiscale setting, not an exact reconstruction of that
file's layout. No new AutoStakkert run is performed.

## Experimental combinations

Every individual scale and both multiscale layouts are evaluated with:

1. The original global selection shared by all APs.
2. Independent selection at each AP using a two-pixel quality-proxy sigma.
3. Independent selection at each AP using a four-pixel quality-proxy sigma.

Each is combined both directly and after aligning completed patch means to their
corresponding reference patches. The latter adds one bilinear interpolation.
These experimental sigmas are not claimed to match AS Noise Robust settings.
This produces 36 outputs: 24 individual-scale controls and 12 multiscale outputs.

Combination sums each completed patch mean times its circular footprint and
observed support, across all contributing scales, then divides by the total
contribution weight. Where the footprint sum falls below one, the remainder is
supplied by the original global stack. This preserves brightness and avoids the
hard support-boundary seams found in the earlier prototype. No spatial smoothing
is applied after the combination.

Overlapping APs can choose different frame sets. Effective frame counts are
therefore also calculated from the final frame weights at each disc-region pixel.
These counts include the global fallback, but omit changes in noise covariance
caused by the different subpixel resamplings. They are not a full photon-noise
model.

## Implementation validation

The 65-pixel measurement and patch arrays are reused unchanged. Recombination
must reproduce all six previous 65-pixel floating images within 1e-12 ADU. Their
sharpened versions are compared pixel-for-pixel against the preceding experiment.

Larger patches use a tiled float64 CUDA correlation calculation to avoid
materializing all shifted candidate images. Its complete correlation scores
match the existing weighted NCC implementation within 2e-12 at every tested
size, and the peak offsets and acceptance decisions agree. The one-kernel
bilinear sampler is checked against the existing sampler with float32/float64
coordinates, multiple image planes, broadcasting, and detector-edge samples.
A 128-frame pilot produced identical shifts and finite quality values with both
sampling implementations.

Synthetic tests verify that mixed patch sizes reproduce a shared detailed scene
without introducing smoothing, preserve constant absolute brightness through
partial support, and distinguish patch support from motion-search support.

Preparation runs on eight CPU workers. SER reads remain serialized because the
source owns a seek/read cursor; only independent arrays enter the worker queue.
The queue is bounded. The experiment does not change production CUDA kernels.

## Sharpening and measurements

Every PNG is passed through the actual PlanetaryTools loader, wavelet sharpening
27/0/0/0, and contrast-adaptive deconvolution amount 15.6, then saved using the
actual 16-bit exporter. No normalization step is added.

The upper/lower disc statistics are the same robust sigma-two high-pass
variation measures as before. They include real detail and artifacts and must
not be described as pure sensor noise or a calibrated resolution measurement.

## Reproduction

```sh
PYTHONPATH=. .venv/bin/python tools/multiscale_patch_experiment.py \
  --capture 2024-09-27-1154_3-CK-R-Sat.ser \
  --out out/saturn-multiscale-patches

PYTHONDONTWRITEBYTECODE=1 \
  ../PlanetaryTools/planetary-app/.venv/bin/python \
  tools/planetarytools_sharpen_experiment.py \
  --planetary-tools ../PlanetaryTools/planetary-app \
  --out out/saturn-multiscale-patches/sharpened \
  --verify out/saturn-multiscale-patches/scale65_local2_registered.png \
    out/saturn-patch-stacking/sharpened/patch_local2_registered_sharpened.png \
  out/saturn-multiscale-patches/*.png

PYTHONPATH=. .venv/bin/python tools/analyse_multiscale_patches.py \
  --experiment out/saturn-multiscale-patches

PYTHONPATH=. .venv/bin/python -m pytest -q tests/test_multiscale_patch_experiment.py
```

CUDA is required for measurement and accumulation. `--stage measure`,
`--stage stack`, and `--stage combine` allow reuse of the saved arrays. An optional
`--limit` is restricted to measurement-only pilots; partial arrays are rejected
by the full stack stage. The sharpening destination must be a new directory.
