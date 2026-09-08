# One alternative optimizer on the unchanged scene objective

Prospective full-grid protocol, 2026-09-08. Compare diagonally scaled L-BFGS-B
against the existing projected-acceleration reference on the frozen5%/100%
selected-frame endpoints (seed1001,D/r0=4,feature). The earlier weak-prior pilot
needed1340 reference iterations; the full-count profile measures2.4s per500-frame
normal operation. P2 permits one alternative selected from conditioning/cost.

Keep the exact same observations, native cells, margin64, positive SUM ridge
.0003, scalar observed variances and zero initialization. Scale variables by
sqrt(diagonal majorizer); this changes coordinates, not the objective. Use the
installed SciPy L-BFGS-B with8 history pairs and20 line-search steps, explicit
analytic gradients and no function-decrease or projected-gradient tolerance as
a qualification criterion. Require the same1e-5 independent CPU distance bound,
feasibility and1e-4 latent/detector budget stability at separate750/1500 caps.

Evaluate objective and gradient from weighted residuals in one exact shared FFT
pass. This avoids cancellation from subtracting large quadratic constants. It
must match independent CPU values/gradients, including masks, RGB/CFA and cell
area scaling. Small active-constraint dense controls and hardware checks precede
the full-grid run. Optimizer success or a stable image cannot override a failed
certificate; an early function/line-search stop remains incomplete.

Use one CUDA worker, two CPU threads, a256MiB retained-spectrum cache,300s per fit
and1200s per pilot, as for the reference endpoints. Each cap starts from zero.
Retain actual iterations, function/gradient evaluations, wall/RSS/cache metrics,
and all incomplete outcomes. Do not auto-tune history, caps or tolerances from
the result. Timings are workload-dependent observations, not an isolated speed
benchmark; compare objective and independent certificates first.

L-BFGS internal history is not serialized. Only completed-stage resume is supported;
resuming does not retry a recorded incomplete fit. No exact iteration-resume or
production adoption is claimed. The reference solver retains its existing exact
iterate/momentum resume. Any full-family adoption needs further qualification.
No prior optimum, scientific gate pass or Q3 authorization follows.
