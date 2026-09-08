# Prospective complete observed-selection numerical family

Execution is gated on a valid unchanged `p2-reference-stability` report. Until
that prerequisite passes, the runner must refuse every case. No outcome is
assumed by implementing this protocol in advance.

Use the frozen 12-case manifest in its existing order: seeds 1001, 1002, 1003;
Dr0 values 4 and 8; feature and bland crops. Each case contains the fixed observed
5/10/25/50/100% selections (25/50/125/250/500 frames). Run all five fractions for
each case, with independently initialized reference v4 caps 1500 and 3000.
Do not change selected indices, the native-cell domain with margin 64, observed
variance, SUM prior 0.0003, distance threshold 1e-5 or image stability threshold
1e-4. There is no prior selection or scientific-gate claim in this protocol.

Use the qualified exact-crop CUDA backend, 4 GiB retain-first spectra, at least
1 GiB additional free device headroom, two BLAS threads and the independent
ordered eight-worker CPU verifier (one FFT thread per worker). Both cap outputs
receive fresh independent CPU certificates. Record latent/detector changes.
Every fraction requires both certified feasible fits and both stability checks.
A case requires all five fractions. Complete-family qualification requires all
12 cases; a successful case is not a family pass.

Invoke one case at a time, in manifest order, in its own output directory. Each
case has a 3600-second total budget. Per-fit limits for the five fractions are
300, 300, 600, 900 and 900 seconds respectively. These prospective allowances
reflect the measured full-count normal cost (0.391 seconds) and the certified
reference run (1410 iterations in 685 seconds). They are execution ceilings, not
promises of convergence; failed or missing stages remain incomplete. The
full-count limit is the same as the prerequisite stability study. Record actual
iteration/product work separately; constant cap numbers are not constant cost
across selected-frame counts.

Each fit has its own zero-start stage and reference iterate/momentum checkpoint.
Completed-stage reuse requires an identical case protocol, source, input and
runtime. A completed wall-limited fit is a retained failed result, not permission
to add time on resume. Current operations and final checks complete after a
budget deadline. No simultaneous scientific study or automatic budget increase.

Preserve the older endpoint studies and their failed lower caps. These new case
runs form one consistent 1500/3000 family identity; they do not silently import
historical pilot stages. Verify the prerequisite report/protocol hashes and
numerical source identities before and after each case. Stop for a new protocol
if implementation, source, numerical criteria or execution budgets change.
