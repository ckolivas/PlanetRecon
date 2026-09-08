# Reference selected-frame endpoint pilot

2026-09-08. **The25-frame (5%) endpoint passes; the500-frame endpoint is
incomplete under the declared wall budgets.** Both endpoints were attempted and
their input/source identities stayed unchanged. Neither a larger iteration-cap
label nor image stability can override an exhausted wall budget.

The [prospective protocol](../../docs/scene-full-selection-pilot-protocol.md)
fixes the observed selection manifest, native cells, margin64, scalar observed
variances and SUM prior.0003. The mean ridge is.0003/25 or.0003/500. Each of the
750/1500-cap fits starts from zero with a300s iterative wall budget; the pilot
allows1200s with iteration/stage-boundary deadline checks.

Both5% fits certify at310 iterations, with independent CPU relative distance
bounds9.96e-6 against the1e-5 requirement. Their latent/detector images are identical.
This endpoint took138.19s including setup, checkpoints and independent checks.

At100%, the750-cap attempt stopped at142 iterations and the1500-cap attempt at
107 iterations. Their independent CPU relative distance bounds are0.14992 and
0.15344. These are conservative error bounds, not measured image errors or
evidence that information is physically unrecoverable. Different iteration counts
under the same wall limit reflect the recorded overlapping workload; they are
not a convergence comparison at two actual completed iteration budgets.

The full pilot took1178.14s. Individual solver wall times include final work
beyond the iteration deadline (306.62/308.10s); independent CPU checks then add
their own cost. No deadline, tolerance or cap was silently increased. The all-frame
fits are retained in `incomplete-100-*.json`, with exact iterate/momentum states
remaining under ignored local `out/p2-selection-endpoints-reference/iterations`.
The failed stages are not stored as completed reusable solutions.

[Report](report.json), [protocol](protocol.json), [checksums](checksums.json) and
the workload record preserve the result and its scope. No truth pixels were
used for reconstruction, and no new scientific pass or Q3 authorization follows.
Next compare the declared single alternative on the same objective; do not
multiply this conditioning failure across the full60-selection family yet.
