# Saturn: is global alignment following the grain?

2026-09-27. Requested test: increase smoothing only for measuring alignment,
stack the original raw frames, then sharpen using the user's actual
PlanetaryTools recipe: wavelet 27/0/0/0 followed by contrast-adaptive
deconvolution amount 15.6.

**Result: stronger global alignment-proxy smoothing does not remove the excess
grain.** It changes the measured shifts but gives only small changes in the
sharpened fine-scale variation. This substantially weakens global grain locking
as the main explanation for the AS/PR difference in this capture.

## Exact sharpening reproduction

The existing implementation in `../PlanetaryTools/planetary-app` was invoked
directly, including its normal image loader, linear/perceptual conversions,
filter registry post-processing and 16-bit PNG exporter. No PlanetaryTools
source was modified. Its checkout was at `aedf4968d18ef55a3a5fc3c8de6593be642c7f26`;
the experiment also records hashes of the relevant implementation files.

- Wavelet: fine 27, medium/coarse/chunky zero, luminance false, auto false.
- Adaptive deconvolution: amount 15.6, contrast adaptive true, luminance true,
  auto false. The loader expands these monochrome sources to neutral RGB.
- Normal defaults: clamp high enabled, clamp low disabled.
- Both operations are performed in sequence in memory; export is 16-bit PNG.

The replay reproduces all three supplied sharpened files **pixel-for-pixel and
byte-for-byte**: PR local, PR global-only, and AS F5738 manual 64-frame reference.
This verifies the complete workflow against the user's results rather than
assuming that matching the numeric settings alone is sufficient.

## Stacking controls

All variants use the exact original 5,738 retained raw frames, their same
quality weights, and the original production 64-frame reference array. The
reference is held fixed to isolate alignment measurement. Local corrections
are disabled in every variant; the earlier experiment showed that doing this
barely changes the grain relative to the normal local stack.

The existing amplitude-correlation estimator is evaluated with Gaussian proxy
sigmas 1.5 (production default), 3, 6 and 12 pixels. Its Fourier weighting is
the product of the two proxy Gaussian responses. Only the estimated translation
is supplied to the unchanged production raw backprojector. Each original raw
frame is interpolated and accumulated normally; no proxy pixels enter the sums
and no new filtering is applied to the reconstructed image.

The cached input identity and selected frame IDs/quality values were verified.
The sigma-1.5 run reproduces both the earlier shifts and the earlier floating
global-only stack **exactly**, and its sharpened PNG is byte-identical to the
user's supplied global-only sharpened PNG. The complete four-variant CUDA
experiment took 96.9 seconds on the RTX 5070 (eight CPU threads).

## Measured results after sharpening

The same upper/lower disc patches and robust sigma-two high-pass statistic as
the earlier report are used. Values are percentage variation relative to local
brightness, and include real detail and artifacts as well as random variation.

| Alignment-proxy sigma | Upper patch | Lower patch |
|---|---:|---:|
| 1.5 pixels, global-only baseline | 2.9577% | 2.9832% |
| 3 pixels | 2.8789% | 2.9101% |
| 6 pixels | 2.8593% | 2.8403% |
| 12 pixels | 2.9156% | 2.9015% |
| Original PR local alignment | 2.9668% | 2.9751% |
| AS F5738, manual 64-frame reference | 0.5732% | 0.6419% |

Sigma six gives the smallest values: reductions of 3.3% and 4.8% relative to the
matched global-only baseline. They remain 5.0 and 4.4 times the AS measurements.
The separate Laplacian-based finest-scale statistic also remains almost
unchanged. Visual inspection of sigma 1.5 and 12 sharpened outputs shows the
same prominent grain; there is no clear improvement resembling AS.

The controls demonstrably changed alignment: for sigma 12 the median translation
change magnitude is 0.164 pixels, the 99th percentile 0.533 pixels, and the maximum
0.908 pixels. Thus the negative result is not an unused setting or identical
shift calculation. The half-stack differences also show no substantial drop in
the non-common component.

## Interpretation and limits

The hypothesis was plausible, but this experiment does not support global
alignment chasing fine grain as the main cause of the large discrepancy. It
does not prove that all alignment-induced bias is zero. The production reference
was deliberately held fixed, and this is a test of global translation, not an
exhaustive test of every local-registration implementation.

No alignment default should be changed on the strength of these small differences.
The user prefers investigation of the unfiltered stacking mechanism; no image
filter or smoothing stage has been added to production. The origin of AS's much
lower fine-scale variation remains unresolved. In particular, these results do
not identify sensor noise or establish that AS uses a smoothing filter.

## Reproduction and outputs

```sh
PYTHONPATH=. .venv/bin/python tools/alignment_proxy_experiment.py \
  --capture 2024-09-27-1154_3-CK-R-Sat.ser \
  --experiment out/saturn-controlled-alignment \
  --out out/saturn-alignment-proxies

PYTHONDONTWRITEBYTECODE=1 \
  ../PlanetaryTools/planetary-app/.venv/bin/python \
  tools/planetarytools_sharpen_experiment.py \
  --planetary-tools ../PlanetaryTools/planetary-app \
  --out out/saturn-alignment-proxies/sharpened \
  --verify out/saturn-alignment-proxies/global_sigma1.5.png stacking/global_subpixels.png \
  out/saturn-alignment-proxies/global_sigma1.5.png \
  out/saturn-alignment-proxies/global_sigma3.png \
  out/saturn-alignment-proxies/global_sigma6.png \
  out/saturn-alignment-proxies/global_sigma12.png
```

Output directories must be new. Raw floating snapshots, coverage, half stacks,
translations and 16-bit PNGs are retained. The sharpening directory additionally
contains floating arrays after each of the two steps. Reproduction of the three
existing user images is recorded under `out/saturn-sharpening-reproduction/`.

The tracked result record is `results/real-data/saturn-alignment-proxies.json`.
Patch metrics use `measurements()` in `tools/alignment_noise_experiment.py`.
All original input files and unrelated repository edits remain untouched.
