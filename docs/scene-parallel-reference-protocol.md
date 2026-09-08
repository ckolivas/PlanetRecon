# Bounded parallel independent CPU verifier

Prospective protocol, 2026-09-08. The full-count profile measured a 102-second
independent CPU certificate at 500 frames. Test parallel evaluation of the
existing spatial CPU frame operators; do not use the shared CUDA Fourier
implementation as its own reference.

Use the same seed1001, D/r0=4, feature crop, all500 observations, native cells,
margin64, observed scalar variances, SUM prior.0003 and observed-derived constant
probe as the full-count profile. Compare one versus eight frame workers for
normal and weighted adjoint operations. Each worker FFT uses one thread; cap
BLAS at two threads. Consume frame results in original order, with at most eight
pending futures. Report forward-workspace/RSS limits separately from this bound.

Require bit-for-bit equal ordered normal and adjoint arrays on this CPU/runtime.
Additionally compare the resulting relative distance bound with the earlier
serial profile to1e-10 relative tolerance. This constant probe is not a solution.
The full-grid test follows small mono/RGB/CFA, ordered-sum, solver/certificate and
worker-failure checks. Record source/input identities, actual wall time, RSS and
pending-frame high-water mark. Limit the profile to600 seconds between operations.

The profile may overlap the two endpoint studies: eight FFT workers plus two
BLAS threads here and four CPU threads there stay within the requested32-thread
allowance. Workload-dependent timings do not establish an isolated speedup.
No running study is changed to use the new verifier; migration requires a new
protocol identity and preserves previous checkpoints/results. Q3 remains closed.
