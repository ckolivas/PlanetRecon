# Prospective 25-frame window / projection-phase ablation

Declared before any fitted output from these candidates is inspected. This is
one development case, not the full numerical family or scientific qualification.

Use seed 1001, Dr0=4, feature crop and the exact observed 5% selection (25 of 500
frames) from the frozen full-selection manifest. Keep native cell factor one,
64-pixel margin, observed detector variance, SUM prior 0.0003 (mean ridge 1.2e-5),
no smoothness and the original independent CPU distance threshold 1e-5.

Compare two modes, in this order:

* Existing safeguarded Newton-CG v2 with the window-average positive inverse.
* Gradient-projection/CG v1 with the same window-average positive inverse.

For each mode independently start from zero at caps 750 and 1500 Hessian products.
Keep 32 inner products, 0.1 relative inner stopping, twelve projected line-search
trials and sufficient decrease 1e-4. The new mode adds its fixed eight-step
projection phase and per-cycle gradient refresh as documented in
[the candidate contract](scene-gradient-projection.md). Both use the unchanged
exact objective and a fresh independent CPU certificate.

Each fit has a 300-second wall budget, including optimizer finalization once it
returns; transforms/checks already in progress are not interrupted. Independent
CPU verification follows and is retained even at the deadline. Total study budget
is 1200 seconds across four fits and setup, checked between stages. Budget stops
are incomplete, never numerical success. Completed stages can be reused under
identical source, input, selection and protocol; no exact internal optimizer
resume is claimed. All accepted and unaccepted inner traces remain recorded.

Use qualified CUDA retain-first spectra (3 GiB, at least 1 GiB additional free
headroom), 2 BLAS threads and the ordered 8-worker independent CPU verifier with
one FFT thread per worker. One scientific study runs at a time; normal desktop
activity may overlap. This ablation is not an isolated speed benchmark.

A mode passes only if both caps satisfy the original fresh independent bound
and their latent and detector image changes are each at most 1e-4. Preserve
failed modes separately. The new mode may progress to a separately declared
500-frame study only if its own 25-frame criteria pass with unchanged source and
inputs; an incomplete control mode must not be relabeled valid. Passing this
case does not select a scientific prior or authorize production/Q3.
