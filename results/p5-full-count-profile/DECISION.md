# Full-count resource and operator parity profile

2026-09-08. **11, 100 and all500-frame probes pass independent CPU parity.**
Largest relative difference among normal operator, linear term and objective
is8.84e-16, below the declared1e-10 tolerance. The probe is a fixed nonnegative
constant scene derived from observed counts; it is not a converged reconstruction.

| Frames | CUDA normal, two passes (s) | CPU normal (s) | CPU certificate (s) | CUDA setup (s) | CPU setup (s) |
|---|---|---|---|---|---|
|11|0.0342 / 0.0368|1.284|1.329|0.081|1.866|
|100|0.289 / 0.287|11.265|11.516|0.534|16.853|
|500|2.428 / 2.359|90.486|101.851|2.664|122.102|

Full500-frame data/PSF preparation took31.02s; the whole profile took455.64s.
These timings were observed under the current host workload, not on an isolated
benchmark machine. CPU cost grew faster than frame count in this run; do not
extrapolate it as a fixed per-frame cost. CUDA remains useful despite cache churn.

The256MiB retained-spectrum cache holds all11 spectra. At100 and500 frames, both
normal passes miss every spectrum (200/1000 misses respectively). A larger
resident GPU cache is not justified on this host: only about2.4GiB was free at
the preliminary check. No other applications were stopped or evicted.
Peak Torch allocated memory was374,439,936bytes and process high-water RSS
3,805,913,088bytes. Cache allocation, process RSS and total GPU memory are distinct.

[Report](report.json), [protocol](protocol.json) and per-count records preserve
timings, certificates, runtime/input/source identities and cache counters;
[checksums](checksums.json) cover archived JSON. Source and inputs stayed unchanged.
The existing exact operator is retained; no approximation or larger cache was
introduced as a consequence of this measurement.

Next: one bounded numerical pilot at the frozen5%/100% selected-frame endpoints.
At the measured all-frame cost, a complete many-thousand-iteration family is
expensive. Preserve time-limited convergence probes before selecting a different
solver or execution strategy. Scientific qualification and Q3 remain outstanding.
