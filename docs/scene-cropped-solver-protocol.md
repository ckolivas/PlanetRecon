# Prospective reference/diagonal Newton endpoints with exact-crop FFTs

Before inspecting these fitted outputs, declare two separate invocations of the
same runner: diagonal Newton-CG v2 first, then projected acceleration v4 reference.
Each invocation checks the original 25- and 500-frame endpoints for seed 1001,
Dr0=4, feature, at independently started caps 750 and 1500. Preserve native cells,
margin 64, observed variance, SUM prior 0.0003, independent CPU relative distance
threshold 1e-5, and latent/detector doubled-cap stability threshold 1e-4.

The new exact-crop FFT backend and 4 GiB retain-first spectra are qualified by
`results/p5-cropped-fft/report.json`; verify the report/input and numerical source
identities before use. Require 1 GiB additional free GPU memory before setup.
Keep 2 BLAS threads and the independent ordered 8-worker CPU verifier with one
FFT thread each. No precision change or approximate-inverse change is involved.

## Execution budget and rationale

Each 25-frame fit retains 300 seconds. Each 500-frame fit gets **900 seconds** in
this new experiment. The measured 0.390903 seconds per normal product implies
about 587 seconds for 1500 products before optimizer, checkpoint and verification
work. Reference iteration caps also incur periodic certificate products. The old
300-second ceiling cannot exercise both declared caps, even after the compute
improvement. The new ceiling adds explicit overhead allowance. It does not change
a convergence tolerance, overwrite an old deadline failure, or guarantee a pass.

Each method invocation has a total budget of 2400 seconds for four fits, setup
and independent checks. Budgets are checked between operations; current transforms
and final checks complete. Methods run sequentially. These are local numerical
studies, not isolated speed benchmarks. The cap means iterations for reference
and inner/search/refresh Hessian products for Newton; do not equate those units.

All terminal scenes, feasible accepted traces, inner/unaccepted progress and
fresh independent certificates are retained. Completed-stage reuse requires an
identical protocol/source/input identity. Reference iterate/momentum checkpoints
are retained for diagnosis, but completed wall-limited stages remain completed
incomplete results; `--resume` does not silently add time to them. Each cap starts
from zero in its separate namespace. No internal Newton resume is claimed.

A fraction passes only if both caps meet the independent distance criterion,
feasibility and image stability. A method passes this endpoint study only if both
fractions pass. Retain partial and failed outcomes. Neither a pass here nor a
backend speedup qualifies the full 60-selection family, scientific prior, Q2/Q3
or production adoption. Reassess the next step from the completed evidence.
