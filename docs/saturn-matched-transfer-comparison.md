# Saturn comparison with measured AS smoothing

The exact-copy controls showed that the supplied AS results have an effective
7×7 Gaussian spatial response with sigma 1 pixel. This experiment applies that
measured kernel to copies of existing PR floating stacks before export and the
user's sharpening recipe. It does not change production stacking.

## Controls

The original PR stack contains 5,738 frames. The AS comparison is the supplied
5,738-frame, manual-64-reference stack. A second PR input is the best preceding
multiscale patch experiment (`multiscale_local2_registered`); it selects 5,738
frames independently per AP, so its contributing frame union differs.

Both PR floating snapshots were checked to reproduce their original exported
PNG pixels exactly. The Gaussian is applied in linear ADU with sigma 1 and
radius 3. Each copy retains the original black/white mapping (0.0873/214.25828)
and gamma 1. No image-dependent brightness normalization was added.

All five inputs were processed through the actual PlanetaryTools filters:
wavelet 27/0/0/0 followed by adaptive deconvolution amount 15.6, contrast adaptive.
The three unchanged inputs reproduced their previous sharpened outputs
pixel-for-pixel and byte-for-byte, including the user's original PR and AS files.

## Results

The metric is the same robust fine-scale variation in two disc patches used
throughout the preceding experiments, expressed as a percentage of local
brightness. It includes real detail and artifacts; it is not a pure sensor-noise
measurement or a calibrated resolution measurement.

| Sharpened image | Upper disc | Lower disc |
| --- | ---: | ---: |
| Original PR | 2.9668% | 2.9751% |
| Original PR + measured AS smoothing | 0.5644% | 0.5401% |
| AS manual-64 reference | 0.5732% | 0.6419% |
| Experimental multiscale PR | 1.9606% | 1.8697% |
| Experimental multiscale PR + measured AS smoothing | 0.4566% | 0.4195% |

Applying the measured smoothing removes the large excess in this fine-scale
variation statistic. This supports a substantial role for differing spatial
response in the original sharpening comparison. It does not prove equal frame
selection, alignment accuracy, or statistical averaging efficiency.

The comparison montage was inspected. The smoothed PR variants have markedly
less visible grain; ring and limb features still differ. The experimental
multiscale result is visually softer in the ring detail at this smoothing, so
its lower variation alone does not justify promoting its more complex stacking
method. The original PR and AS are closer after smoothing, but not demonstrably
identical in retained detail.

The montage uses identical 500×230 crops around rounded centroids and 2× nearest
neighbor enlargement. There is no fractional resampling or per-panel brightness
adjustment. Its location is `out/saturn-matched-transfer/comparison.png`; the
individual raw and sharpened comparisons are in the same directory.

## What this changes

The immediate grain discrepancy has a measured explanation that the earlier
experiments lacked. Further stacking development should assess retained detail
at comparable grain, rather than treating AS's smoother output as evidence of
better averaging by itself. Keep the unfiltered PR stack available; the measured
smoothing is currently a diagnostic postprocessing operation, not an implicit
stacking change. There is no evidence here to promote the experimental multiscale
implementation into production.

The effective kernel was measured on AS's exact-copy controls. Applying it to
these real-data stacks assumes a similar spatial response; the AS code and the
processing stage responsible remain unknown.

## Reproduction and artifacts

`tools/matched_transfer_comparison.py` prepares the inputs and analyses outputs.
It requires a new output directory for preparation. Run the sharpening harness
on the five prepared PNGs using the PlanetaryTools environment, then invoke
the comparison tool with `--analyse` and `QT_QPA_PLATFORM=offscreen`.

Detailed input hashes, mapping, sharpening implementation hashes, unchanged
control verifications and measurements are in
`results/real-data/saturn-matched-transfer.json`. Local arrays and images are in
`out/saturn-matched-transfer/`. Production algorithms and defaults are unchanged.
