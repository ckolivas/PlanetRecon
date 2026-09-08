# Full-count cache retention profile

Prospective protocol, 2026-09-09. Repeated sequential 500-frame sweeps miss every
PSF in the current undersized LRU cache. The native 1024 by 912 scene and 512 by
512 PSFs require 8,859,648,000 bytes for all complex128 spectra, exceeding the
7,329,021,952 free bytes measured while the endpoint study was active. Do not
attempt an unbounded whole-sequence allocation or change numerical precision.

Implement one experimental admission policy: retain the first spectra that fit
and evaluate the remaining spectra transiently. Keep arithmetic, frame order,
pixel integration, PSFs, objective and precision identical. This is an audit
backend only; the default LRU implementation and production application stay as
they are. Test exact CPU results, cache bounds, sequential hits and clearing.

After the endpoint study completes, run one profile on seed 1001, D/r0=4, feature,
all 500 observations, native cells, margin64 and SUM prior 0.0003. In fixed order
compare LRU256MiB, LRU3GiB and retain-first3GiB. Keep at least1GiB of free device
headroom above the selected cache budget before constructing each mode. Record
insufficient memory or allocation failure; do not reduce precision or retry at
an undeclared capacity. These limits cover retained spectra, not total memory.

Use two BLAS threads and eight ordered independent CPU frame workers, with one
FFT thread each. Before GPU modes, compute independent CPU linear term, normal
product and objective on a constant nonnegative probe set from observed mean
brightness. Each GPU mode constructs the same quadratic, runs three normal
products and one objective, recording per-operation times, cache hits/misses,
retained bytes, process RSS and Torch peak allocated/reserved memory. All product,
linear and objective relative errors must be <=1e-10. Check source/input identity
before and after. Total budget600s, checked between operations.

No reconstruction is performed. Report timings as local sequential measurements,
not universal speedups. A successful retention profile alone does not qualify a
solver, authorize larger fit deadlines, select a prior or change scientific gates.
Any later adoption requires a new solver-study identity and explicit resource
budget. The completed 300-second endpoint failures remain unchanged.
