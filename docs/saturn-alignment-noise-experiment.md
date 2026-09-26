# Saturn: controlled alignment and frequency-response experiments

Run date: 2026-09-26. Capture: `2024-09-27-1154_3-CK-R-Sat.ser`.

**Removing local alignment does not explain the excess fine-scale variation in
this capture.** Translation-only stacking is almost identical by the measured
disc-patch statistics. The much stronger high-frequency roll-off in the supplied
AutoStakkert exports remains unexplained by this experiment.

## Controls and validation

Three full stacks use the same 5,738 selected frames, in the same order, with the
same quality weights and the exact same production 64-frame reference. They
differ only in the displacement supplied to the production raw-image
backprojector:

1. Production circular multiscale local displacement field.
2. Global subpixel translation, with no local correction.
3. Rounded global translation, as an integer-sampling diagnostic.

The reference is built once by the actual stacking pipeline. Instrumentation
checks every processed frame against the raw capture at its expected index.
The production local result and the separately accumulated result agree within
floating-point summation precision. Its exported PNG has **exactly identical
pixel values to the user's original SatP.png**, including all 281,184 pixels.
This also validates using CUDA for this reproduction of the original CPU stack.

Each variant additionally produces two stacks from alternating selected frames
in capture order. Their weight fractions are 0.499728 and 0.500272. Half their
difference estimates the non-common component at the full-stack averaging level.
They share a reference: reference-correlated errors and other common structure
can cancel, so these are not completely independent reconstructions.

CUDA run: RTX 5070, eight CPU threads, 188.5 seconds for the instrumented stacking
pass. This is not a normal-stacker performance benchmark: it computes three
outputs and performs extra frame comparisons and diagnostics.

## Measurements

Two patches on Saturn's disc avoid the ring crossing and are located relative to
each image's intensity centroid. Each statistic below is 1.4826 times the median
absolute deviation of the residual after subtracting a Gaussian with sigma two
pixels, expressed as a percentage of the patch's median brightness. Border
pixels are discarded. These are **fine-scale variation measurements, not pure
sensor-noise measurements**; actual detail and processing artifacts can contribute.

| Stack | Upper patch | Lower patch |
|---|---:|---:|
| PR local alignment | 0.1699% | 0.1416% |
| PR global subpixel only | 0.1687% | 0.1409% |
| PR global integer only | 0.2096% | 0.1865% |
| AS 5,738 frames, automatic reference | 0.0941% | 0.0897% |
| AS 5,738 frames, manual 64-frame reference | 0.0951% | 0.0883% |
| PR local, deliberately Gaussian-blurred sigma 1 pixel | 0.0971% | 0.0848% |

Removing local corrections changes the first statistic by -0.70% and -0.52%
relative to the local result. Integer sampling increases it by 23.4% and 31.8%.
The local-minus-global field's component RMS over a central disc region has
median 0.0955 pixels and 90th percentile 0.1576 pixels across frames.

| Half-stack difference statistic | Upper patch | Lower patch |
|---|---:|---:|
| PR local alignment | 0.0810% | 0.0784% |
| PR global subpixel only | 0.0834% | 0.0786% |
| PR global integer only | 0.1297% | 0.1216% |

The local-versus-global difference also remains small in the non-common
component. Quality weights give an effective count of 5,694.3 rather than 5,738:
under the equal-variance independent-frame model this is only a 0.38% standard
deviation penalty. That model is not a claim about the origin of the variation.

## Frequency-response control

Hann-windowed spectra of two larger, brightness-normalized disc patches show
that PR has about 508 and 493 times the AS manual-reference power in the
0.4–0.5 cycles/pixel band. Those are power ratios, not noise-amplitude ratios,
and the absolute power in this band is small. PR's local and translation-only
spectra are almost identical there. Both supplied AS reference settings produce
similar strong roll-off toward the pixel-scale limit.

A Gaussian sigma sweep of 0.5–1.2 pixels on the existing PR stack shows that
roughly one pixel of smoothing brings the fine-scale statistics close to AS.
This establishes that a difference in spatial frequency response can account
for much of the observed sharpening sensitivity. It does **not** establish that
AS applies a Gaussian, locate the source of the roll-off, or demonstrate that
blurred PR matches AS detail. The smoothed control deliberately sacrifices
detail; it is not a proposed production fix.

## What this resolves and what remains

- Local AP corrections are not the main source of the measured excess for this
  capture. The controlled test avoids the UI toggle changing the reference.
- Removing bilinear subpixel interpolation does not fix it; integer sampling
  increases the finest variation in this test.
- Reference frame count alone and global quality weighting do not account for
  the gap in these comparisons.
- The results do not identify the excess as sensor noise. They do not prove that
  every extra high-frequency component is either useful detail or an artifact.
- AS and PR have matched counts, but AS's exact frame identities and local AP
  frame selections are unavailable. This is a controlled comparison between PR
  variants, with external AS results as benchmarks, not a common-input replay
  of both applications.
- AS drizzle, resizing, and export settings need confirmation before attributing
  its roll-off to an internal interpolation or stacking algorithm. A comparison
  at matched spatial response, with independent detail validation, is the next
  useful step; identical sharpening settings alone do not provide that control.

## Reproduction and artifacts

Run from the repository root with its existing CUDA-enabled environment:

```sh
PYTHONPATH=. .venv/bin/python tools/alignment_noise_experiment.py \
  --capture 2024-09-27-1154_3-CK-R-Sat.ser \
  --export 2024-09-27-1154_3-CK-R-SatP.png \
  --out out/saturn-controlled-alignment

PYTHONPATH=. .venv/bin/python tools/analyse_alignment_noise.py \
  out/saturn-controlled-alignment \
  --comparison AS_F5738/2024-09-27-1154_3-CK-R-Sat_lapl4_ap54.png \
  --comparison AS_F5738/2024-09-27-1154_3-CK-R-Sat_r64_lapl4_ap54.png \
  --comparison 2024-09-27-1154_3-CK-R-SatP.png
```

The first command requires a new output directory. The analysis command can be
rerun. All three unfiltered PNGs use the original PR export's black/white mapping.
The analysis script also generates the smoothing control with the same mapping;
its filename explicitly identifies it as a control. It contains diagnostic text
rather than reconstruction provenance.

Full local artifacts in `out/saturn-controlled-alignment/`:

- `local.png`, `global_subpixel.png`, `global_integer.png`: unfiltered stacks.
- `local_sigma1_control.png`: deliberately blurred comparison for sharpening.
- Per-variant `.npz` files and `_halves.npz`: floating-point arrays and coverage.
- `reference.npy`, `diagnostics.npz`: exact reference, selected frame IDs,
  translations, local-field statistics and quality weights.
- `report.json`, `analysis.json`: machine-readable results.

The smaller tracked record is
`results/real-data/saturn-controlled-alignment.json`. Production stacking and GUI
behavior have not been changed by these diagnostic tools.

## User-sharpened controls supplied in `stacking/`

The three unsharpened inputs in `stacking/` have exactly identical decoded pixels
to the corresponding experiment exports. The supplied `locals.png` is also
byte-identical to the earlier `SatPs.png`. The sharpened images were visually
inspected and measured with the same centroid-relative patches and native
16-bit decoding used above. The sharpening settings themselves are not encoded
in these files.

| Sharpened result | Upper fine-scale variation | Lower fine-scale variation |
|---|---:|---:|
| PR local (`locals.png`) | 2.9668% | 2.9751% |
| PR global only (`global_subpixels.png`) | 2.9577% | 2.9832% |
| PR Gaussian control (`local_sigma1_controls.png`) | 0.5662% | 0.5410% |
| AS 5,738 frames, automatic reference | 0.5627% | 0.6574% |
| AS 5,738 frames, manual 64-frame reference | 0.5732% | 0.6419% |

The local/global differences are only -0.31% and +0.27% of the local variation.
Thus the same conclusion survives the user's actual sharpening workflow:
removing local alignment does not remove the visible grain.

The Gaussian control reduces variation by 80.9% and 81.8%. It approaches AS's
upper-patch value and is below AS in the lower patch. Visually it also softens
the limb and ring structure, and looks softer than the AS manual-reference
result in those features. This is a qualitative inspection, not a calibrated
resolution measurement. Matching the fine-scale variation statistic therefore
does not establish matched detail or equal stacking performance.

These results strengthen the frequency-response explanation for the sharpening
sensitivity, but do not establish the mechanism producing AS's response. They
do not justify adding a one-pixel Gaussian blur as a stacking fix. The remaining
comparison needs both noise/fine-scale variation and retained feature sharpness;
the local AP toggle is not a promising solution for this particular discrepancy.

Measurements, source checks and input hashes are recorded in
`results/real-data/saturn-sharpened-controls.json`, using `measurements()` from
`tools/alignment_noise_experiment.py` and `bands()` from
`tools/analyse_alignment_noise.py`. No source images were modified.

## Larger minimum AP size

The user reports that AS used a minimum AP size of 64 pixels and that increasing
PR's minimum from 23 to 65 pixels produced an indistinguishable result.
The supplied `2024-09-27-1154_3-CK-R-Satpx65.png` confirms that this setting took
effect: sampling multiplier 14, actual circular diameters 65, 93, 131 and 185,
5,738 frames, quality weighting enabled. The original PR stack used diameters
23, 33, 47 and 67. Thus this is a change to all four scales, not only the
smallest scale.

The larger-AP raw export measures 0.1732% and 0.1480% fine-scale variation in the
two patches, compared with 0.1703% and 0.1419% in the original raw PNG. Brightness
normalization accounts for the different export mapping. The saved reference
anchor is 13240 rather than 22877, although the 64 reference candidate IDs are
unchanged; this user comparison is therefore not a strictly AP-only experiment.
Nevertheless the grain remains at a similar level and is consistent with the
controlled local/global experiment. Too-small APs are unlikely to be the main
explanation for this discrepancy. No production behavior was changed.
