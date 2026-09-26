# Exact-copy stack transfer probe

The preceding multiscale experiment improved the best sharpened disc-variation
measure by only about 1–2%, leaving roughly a threefold gap to the supplied
AutoStakkert output. This does not identify an AS filter or prove that the
remaining variation is sensor noise. The next experiment measures what happens
to known, unchanging texture through the stacker.

## Prepared inputs

`out/saturn-transfer-probe/` contains two 256-frame, 696×404, mono 8-bit SERs:

* `static.ser`: 256 exact copies of source frame 13240 (zero-based), the
  previously tried near-midpoint Saturn anchor.
* `integer_motion.ser`: the same pixels moved by known integer offsets within
  ±3 pixels. Offsets are balanced around zero and the first frame is unshifted.
  Newly exposed borders are zero-filled; there is no wraparound or interpolation.

There is no added noise, changing seeing, brightness adjustment, or rotation.
The raw frame's grain is part of the fixed ground truth. Its blur is also fixed;
this is a transfer test, not a test of recovered astronomical resolution.

The directory also contains the exact truth in ADU (`truth_adu.npy`), a 16-bit
TIFF with fixed mapping `code = ADU × 257`, the offsets, PR global stacks, and
`report.json`. The TIFF mapping is code expansion, not brightness normalization.

## Required AutoStakkert run

For each SER, use the normal Planet/COG and Dynamic Background settings, AP size
64, Multi-Scale and Close to Edge enabled, and the usual AP placement. Use the
same reference mode for both. Automatic is suitable: all frames contain the
same object and texture. Record any changed settings or warnings.

1. Analyse and place APs.
2. Stack **256 frames (100%)**.
3. Disable **Normalize Stack**, Sharpened, Drizzle, and Resample.
4. Save the unsharpened **16-bit PNG or TIFF** under the probe directory,
   retaining distinct static and integer-motion names.

No PlanetaryTools sharpening is needed at this stage. A known-input test can
measure fine-scale attenuation directly before a nonlinear sharpening process
amplifies it. The established wavelet 27/0/0/0 and contrast-adaptive deconvolution
15.6 recipe remains available for a subsequent visual comparison.

## Validation already completed

Every serialized frame was read back and checked against its expected pixels.
Undoing the known integer offsets and averaging gives exactly the original
frame on common support for both captures.

The production PR CPU global stack used all 256 frames, equal weights, no
screening, no local alignment, and no brightness normalization. Interior errors
against ground truth, excluding eight border pixels on each side:

| Capture | RMS error (ADU) | Maximum error (ADU) |
| --- | ---: | ---: |
| Static | 0 | 0 |
| Integer motion | 0.00000597 | 0.00008003 |

These are global controls, not a claim that the local pipeline was exercised.
Five generator tests passed. AS outputs were pending at this preparation stage;
the supplied results and their analysis are recorded below.

## Interpretation once AS outputs exist

Compare against the original pixels on common support, allowing for AS cropping,
translation, and output code scale. Explicitly distinguish any global fractional
translation and its interpolation from additional spatial attenuation. Do not
silently reinterpolate the comparison image and attribute that effect to AS.

* If static texture is attenuated, independent-frame averaging cannot explain
  it: all frames are identical. Recentring/resampling or another spatial
  operation becomes the next measurable suspect. This alone does not prove
  deliberate denoising.
* If only the moving input is attenuated, concentrate on motion estimates,
  subpixel placement, and resampling/recombination.
* If both retain texture, the real-data difference depends on varying frames,
  seeing, noise, or selection. The control does not exclude processing that
  activates only when frames differ; independent-noise controls would follow.

This probes the stacker's response to fixed texture. It does not recreate
AutoStakkert's private implementation or resolve the real-capture discrepancy
before its outputs are measured.

## Reproduction

```sh
PYTHONPATH=. .venv/bin/python tools/stack_transfer_probe.py --run-pr
PYTHONPATH=. .venv/bin/python -m pytest -q tests/test_stack_transfer_probe.py
```

The generator requires a new output directory; use `--out` for another run.
No production settings or stacking algorithms were changed.

## Supplied AutoStakkert results (2026-09-27)

Inputs in `stack/` have exactly the same SHA-256 hashes as the generated SERs.
The supplied results are `stack/AS_P100/static_lapl4_ap55.png` and
`stack/AS_P100/integer_motion_lapl4_ap55.png`, both 696×400 pixels. Their coarse
registration samples the truth at `(output_y + 3, output_x)`. A code expansion
of 256 output counts per input ADU accounts for their brightness; the fitted
kernel sum independently confirms unity gain to about 0.002%.

**The supplied AS outputs contain an effective Gaussian blur with sigma 1 pixel
and radius 3 (7×7 support), even when every input frame is identical.**

The measurement does not identify the internal stage or setting responsible,
or establish that every AS configuration behaves this way. The saved local
LastSession file predates these outputs and was not used as evidence of their
active settings.

The comparison region is rows 60–339 and columns 100–599, encompassing the
planet and rings. Models were fitted on columns 100–349 and checked on the
separate columns 350–599. AS output pixels were never interpolated.

| Prediction from original pixels | Held-out RMS difference (input ADU) |
| --- | ---: |
| Integer translation only | 4.043409 |
| Fitted fractional translation, cubic interpolation, gain and offset | 3.059323 |
| Fixed Gaussian sigma 1, radius 3; no fitted gain or offset | 0.002222 |
| Freely fitted 7×7 kernel and offset | 0.001982 |

The fixed Gaussian's maximum error in the entire comparison region is less
than two 16-bit output code values. It explains 99.99997% of the squared
discrepancy from the original pixels on held-out data. This is a measure of
prediction accuracy, not a percentage of sensor noise removed or recovered
astronomical detail. The free kernel agrees with the normalized Gaussian
coefficients within 0.00000791.

The static and moving AS results are **pixel-for-pixel identical throughout
that comparison region**. Across the full images 97.065% of pixels are identical;
differences outside the region reach 0.684 input ADU. Thus integer movement
does not account for the central smoothing in this pair.

The Gaussian effect cannot arise from averaging different independent noise
realizations here: every input carries the same texture. A fractional shift
with the tested cubic interpolation also fails to account for the effect.
This establishes the effective spatial smoothing, not whether AS implements
it as a named Gaussian filter or another mathematically equivalent operation.

This provides the direct evidence missing from the earlier speculative blur
control. That control already brought PR's sharpened fine-scale variation close
to AS's ([earlier measurements](saturn-alignment-noise-experiment.md)). However,
its ring and limb detail looked softer. Matching this transfer function is now
a justified comparison control; it still does not establish equal recovered
resolution or justify silently changing production stacking.

The reproducible analysis is `tools/analyse_stack_transfer.py`; results are
saved in `out/saturn-transfer-analysis/report.json` and tracked in
`results/real-data/saturn-stack-transfer-analysis.json`. Eight focused generator
and analysis tests pass, including recovery of a known Gaussian and independent
subpixel-translation/brightness-scale controls.

```sh
OPENBLAS_NUM_THREADS=8 PYTHONPATH=. .venv/bin/python tools/analyse_stack_transfer.py
PYTHONPATH=. .venv/bin/python -m pytest -q tests/test_analyse_stack_transfer.py tests/test_stack_transfer_probe.py
```
