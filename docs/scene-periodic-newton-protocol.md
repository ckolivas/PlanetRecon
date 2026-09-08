# Periodic-inverse constrained endpoint experiment

Prospective protocol, 2026-09-09, after the four frozen probes completed. At32
products the independent full linear residual ratio improved from1.9818 to0.0292,
and the fixed free-variable ratio from1.6232 to0.1717. This supports one bounded
constrained-update experiment; those probes did not qualify the saved scene.

Retain the safeguarded Newton-CG procedure and objective in
[the original constrained protocol](scene-projected-newton-protocol.md).
Change only the inner preconditioner to the positive periodic inverse described
in [the frozen-probe protocol](scene-periodic-probe-protocol.md), and adopt the
qualified retain-first3GiB CUDA cache with at least1GiB additional free headroom.
The reduced application is P C^-1 P on the current free set. The physical
Hessian, nonnegative projection, descent test, majorizer fallback and independent
global-ridge certificate remain authoritative. Do not use the periodic model to
score an image or compute its final certificate.

Dense constrained controls with wrong initial active sets and RGB/Bayer sampling,
invalid inverses, partial-product interruption, and independent CPU/CUDA checks
must pass first. Record every inner residual/product event atomically, including
unaccepted directions interrupted by a deadline, as well as accepted updates.
The new implementation identity is projected_newton_cg_v2. Historical outcomes
and exact-input records remain unchanged; no exact internal iteration resume is
claimed, only completed-stage reuse with identical identities.

Use the same observed selection manifest, seed1001, D/r0=4, feature crop, 25/500
frames, native cells, margin64, observed variance and SUM prior0.0003. Both fits
start independently at zero. Retain product caps750/1500, at most32 inner products,
0.1 inner residual target,300s per-fit deadline and1200s invocation deadline.
Inverse construction and independent CPU verification count in invocation time;
initial/final gradient checks are reported separately from inner/search/refresh
products. An already running operation may cross a deadline. Two BLAS threads,
eight ordered CPU workers and one FFT thread per worker remain the limits.

Require feasible independent CPU relative distance bounds <=1e-5 for both fits
and latent/detector changes <=1e-4. Compare certified objectives with the existing
reference; incomplete results are not accuracy or speed baselines. Report any
failure and do not increase budgets, relax tolerances or tune the prior in
response. A one-case endpoint pass would still not qualify the60-selection
family, scientific quality, production integration or Gate-1/Q2/Q3.
