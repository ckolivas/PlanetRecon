# Safeguarded constrained curvature comparison

Prospective protocol, 2026-09-09. Frozen-iterate diagnostics show that three
diagonal linear probes cannot qualify the saved 500-frame solution. Test a
feasible constrained update on the same objective, retaining the original
certificate. This is a numerical experiment; no physical solver adoption follows.

For x >= 0 and g = Hx-b, the free set is x > 0 or g < 0. Solve the principal
system H_FF d_F = -g_F approximately, with other components zero. Use PCG with
the existing positive majorizer as a preconditioner, at most 32 inner products
and a 0.1 relative inner residual target. This target does not certify the scene.
Release any zero coordinate with a negative gradient on every outer update.

Try projected candidates max(x + 2^-j d, 0) for j = 0,...,11. For s = candidate-x,
compute the exact quadratic change g's + s'Hs/2 and accept only if g's < 0 and
the change <= 1e-4 g's. If rejected, use s = max(x-g/M,0)-x, where diag(M) >= H
is the existing proved majorizer. Projection optimality gives g's <= -s'Ms;
therefore the fallback change <= g's/2 < 0 for a nonzero step. Reject failed
descent rather than accepting an optimizer success flag. Refresh gradients every
10 accepted steps and recompute the authoritative certificate on termination.

Before large data: independent dense constrained optima with incorrect starting
active sets; mono/RGB and CFA forward operators; feasibility and objective descent;
reduced principal-system solves; deliberately rejected Newton directions; expired
deadlines and product caps. Add CPU/CUDA agreement controls on small scenes.

Use the frozen observed selection manifest, seed 1001, D/r0=4, feature crop,
fractions 5% (25 frames) and 100% (500 frames), native cells, margin 64, observed
variance and SUM prior 0.0003. Each fit starts at zero. Two independent fits use
750 and 1500 inner/line-search/refresh Hessian products. These counts are not
comparable to the old solvers' outer iteration caps. Initialization and final
gradient/objective checks are separately reported. Each fit has the unchanged
300-second deadline; total invocation budget 1200 seconds including setup and
verification. Operations/checks already running may cross a deadline. Do not
raise budgets or tolerances automatically after an incomplete outcome.

Use one CUDA study, retained spectra cache 256 MiB, two BLAS threads and an
independent CPU verifier with eight ordered frame workers and one FFT thread
each. Retain every accepted iteration's product count, step kind, exact quadratic
change, active fraction and distance bound atomically. Keep terminal scenes and
all incomplete results locally with exact input/runtime/source identities.
Completed-stage reuse is supported; exact internal PCG resume is not claimed.

Numerical acceptance requires feasible independent CPU relative distance bounds
<= 1e-5 for both fits and latent/detector relative changes <= 1e-4. Compare only
the same objective with prior endpoint evidence. An incomplete 500-frame outcome
does not authorize full-family expansion, prior tuning, threshold relaxation,
production adoption or Gate-1/Q2/Q3 claims. Preserve all failed evidence.
