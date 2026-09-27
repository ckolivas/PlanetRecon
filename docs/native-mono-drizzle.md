# Native-size mono deposition control

The user suggested that AutoStakkert's Bayer drizzle machinery might also
explain its mono stacking behaviour. This experiment tests explicit native-size
deposition kernels while keeping the established 5,738 frames, quality weights
and global translations fixed. It does not reconstruct AutoStakkert's code.

## What is documented

In [post 39 on 18 January 2024](https://www.cloudynights.com/forums/topic/907378-jupiter-at-20-degrees-11624-not-what-you-think-dont-miss-panel-26/page/2/),
Emil Kraaikamp (MvZ) distinguishes integer placement in regular stacking from
subpixel placement in enlarged drizzle. He describes Bayer drizzle as
reconstructing separate colour channels across successive frames. This is
historical developer evidence, not verification of the user's exact build or
its complete mono recombination path.

## Direct test

Each detector pixel deposits into the output at its position minus the known
global displacement. Three kernels are tested at output scale one:

- A square footprint one input pixel wide, using exact area overlaps.
- A square footprint half an input pixel wide.
- A point placed at the nearest output pixel.

Signal and coverage are accumulated with the original scalar quality weights;
division occurs only after stacking. This is ordinary weighted averaging,
without per-frame brightness fitting, output normalization or an added filter.

For a constant translation and unit square footprint, overlap along either
axis is exactly the triangular weight used by bilinear interpolation. The
forward-deposition and backward-sampling formulations therefore yield the same
signal and coverage, including partially covered detector borders. This
equivalence is specific to the stated geometry and kernel, not arbitrary
local warps, distortion corrections or drizzle implementations.

The implementation is checked independently against SciPy's order-one
grid-constant sampler and on isolated impulses. On the complete capture, the
unit-footprint stack reproduces the earlier bilinear global stack within
1e-10 ADU, and point deposition reproduces the earlier integer-shift stack to
the same tolerance. The actual differences are retained in the numerical
record.

## Exact sharpening comparison

All three raw stacks use the established export mapping and PlanetaryTools
Wavelet 27/0/0/0 followed by Adaptive Deconvolution 15.6, Contrast Adaptive.

| Footprint | Upper-disc variation (%) | Lower-disc variation (%) | Left ring width (pixels) | Right ring width (pixels) |
|---|---:|---:|---:|---:|
| Unit square | 2.95770 | 2.98324 | 7.48450 | 6.13284 |
| Half-width square | 4.08763 | 4.25911 | 7.57100 | 6.16146 |
| Point | 5.37379 | 5.57272 | 7.37468 | 6.31892 |

Smaller footprints substantially increase fine-scale variation in these
outputs. That statistic includes detail and artifacts as well as noise; ring
transition widths are descriptive, not calibrated resolution. These are
global-alignment controls, so they cannot exclude a role for a different
local recombination or reconstruction kernel in AutoStakkert.

At exact integer placement, all three tested kernels leave an isolated input
impulse as an isolated output impulse. They cannot produce the broad Gaussian
response previously measured in the supplied identical-frame AS control.
That earlier [transfer measurement](saturn-stack-transfer-probe.md) identifies
an effective spatial response, without identifying its internal source.

Switching from bilinear resampling to native-size unit-square deposition is
therefore not a distinct noise-reduction method in this translation experiment.
The native deposition hypothesis needs a specific different kernel or local
recombination mechanism to explain the observed AS result.

## Reproduction

```sh
PYTHONPATH=. OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python tools/native_drizzle_probe.py --out out/saturn-native-drizzle
```

Run `tools/planetarytools_sharpen_experiment.py` in the PlanetaryTools Python
environment on that directory's PNGs, then run
`PYTHONPATH=. .venv/bin/python tools/analyse_native_drizzle.py`.
Use a fresh output directory for a new replay and adjust the analyzer's root.
The [numerical record](../results/registration/native-mono-drizzle.json)
retains input/model hashes, exact baseline parity and sharpening provenance.
The focused deposition tests cover fractional and integer translation,
boundary coverage and conservation of an interior pixel's total contribution.
