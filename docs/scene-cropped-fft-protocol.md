# Prospective full-count cropped-FFT profile

Run after the active reconstruction study has finished and all small CPU/CUDA
cropped-FFT controls pass. This profile fits no scene. It tests the exact-padding
candidate and measures resources; it cannot authorize a solver or scientific gate.

Use seed 1001, Dr0=4, the feature crop, all 500 observations, native cells,
margin 64, observed variance and SUM prior 0.0003. Form a nonnegative probe from
observed mean brightness divided by bin-area, multiplied by `1+0.25*sign`, where
sign is a fixed independent Rademacher array from NumPy generator seed 7171.
This adds high-frequency content without reading latent truth.

In fixed order compare full-padding retain-first 3 GiB, exact-crop retain-first
3 GiB, and exact-crop retain-first 4 GiB. Keep precision float64/complex128 and
at least 1 GiB free above each cache budget before setup. Do not retry at an
undeclared capacity or change precision. Record actual FFT dimensions, cache
hits/resident bytes, three normal-product timings, forward/objective timings,
process RSS, Torch allocated/reserved peaks and device free/total memory.

The ordered 8-worker independent CPU reference supplies the linear term, normal
product, objective and every frame's forward prediction. Require relative errors
at most 1e-10 for all four categories, including the maximum per-frame forward
error. Use two BLAS threads and one FFT thread per independent worker. Verify
source/input hashes before and after. Total budget 600 seconds, checked between
operations. One scientific study runs at a time. Local sequential measurements
are not universal speed claims. Preserve insufficient-memory or failed parity
outcomes. Any later solver adoption needs its own declared source identity,
resource budget and unchanged independent convergence checks.
