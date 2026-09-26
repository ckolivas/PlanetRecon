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
Five focused tests pass. AS has **not** yet processed these inputs.

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
