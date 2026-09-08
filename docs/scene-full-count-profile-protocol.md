# Full-count optical resource profile

Prospective protocol, 2026-09-08. Measure one development feature crop (seed
1001, D/r0=4), using 11, 100 and all 500 observed frames. Use evenly spaced
indices including the endpoints for this resource profile; these are not quality
selections or a scientific assessment. Native cells, margin64, scalar variance
estimated from each observed frame, fixed SUM ridge .0003, zero prior mean.

Use the existing shared CUDA evaluator with a 256 MiB retained-spectrum cache,
two CPU threads and no other study workers. Record data/PSF setup, CPU and CUDA
problem setup, two normal-operation timings, independent CPU certificate cost,
gradient/objective parity, cache hits/misses, peak process RSS and Torch allocated
and reserved peaks. Read available GPU memory before work; do not evict or stop
other applications to accommodate this profile. No full resident 500-frame GPU
cache is attempted. The cache bound is not a total RAM/VRAM ceiling.

Require relative normal, linear term and objective differences <=1e-10 against
the independent CPU operator. Evaluate a fixed nonnegative constant scene derived
from observed counts, not truth. Its certificate is a parity and timing probe;
it is not expected to pass convergence. Do not report it as a reconstruction.

Allow 900 seconds total, checked between expensive operations. Preserve each
completed count and write an incomplete report on an exception or exhausted
budget. A single operation can cross the deadline. Hash input and all numerical
dependencies before/after. No prior tuning, new scientific pass or Q3 authorization
follows from this profile. Select the next resource strategy from measured cost
and occupancy; any optimization requires independent parity tests and a new run.
