# Same-objective L-BFGS-B endpoint pilot

2026-09-08. **The 25-frame endpoint passes; the 500-frame endpoint remains
incomplete.** This single alternative has not resolved full-frame convergence
within the declared resource budget and is not adopted as a production solver.

The [prospective comparison](../../docs/scene-lbfgsb-comparison-protocol.md) keeps
the exact same observations, selections, native scene, margin, variance estimates
and SUM prior as the reference. Variable scaling changes coordinates only.
The fused weighted residual objective/gradient avoids subtracting large quadratic
constants and passes independent dense and CPU/CUDA controls. The final independent
CPU certificate decides qualification regardless of SciPy's stopping flags.

At 5%, both caps certify after 195 iterations / 206 function-gradient evaluations,
versus 310 reference iterations. Independent relative distance bounds are 9.72e-6,
below 1e-5, and both latent/detector images are identical between caps. Fewer
iterations do not by themselves imply lower cost: the fits took 77.11/80.20s,
and their workload differed from the reference. No isolated speedup is claimed.

At 100%, the 750/1500-cap fits hit their 300s iterative deadlines after 70/116
iterations and 77/124 function-gradient evaluations. Independent relative distance
bounds are 0.36065/0.40111, far above the required 1e-5. Their latent change is
1.500% and detector change 0.816%, both above the 1e-4 relative stability threshold.
These are incomplete numerical fits; the bounds are not measured image errors.
The study took 1163.04s including preparation and independent CPU checks.

[Report](report.json), [protocol](protocol.json), per-fraction records and
[checksums](checksums.json) preserve both passing and failing outcomes. Source,
manifest and input identities stayed unchanged. Completed stage payloads are
retained locally, including incomplete fits. Stage resume reuses their recorded
outcomes; it does not silently retry them. Internal L-BFGS history is not serialized,
so this solver does not claim exact iteration resume.

Next examine conditioning and convergence traces under the frozen objective,
using the qualified parallel CPU verifier in a newly declared study. Do not
increase every budget, relax the certificate or expand this failure into the
full family. Prior/likelihood/phase and scientific qualification remain open;
no Gate-1/Q2/Q3 or production claim follows.
