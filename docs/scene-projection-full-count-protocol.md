# Prospective 500-frame projection/CG endpoint

Proceed only after the 25-frame projection candidate passes both original caps
with independent CPU certification and doubled-budget image stability. The
committed ablation report records that prerequisite; its numerical source and
package hashes must still match, except for the explicitly refactored study
runner. The earlier ablation runner version cannot resume under the new identity.

Use the exact seed-1001, Dr0=4, feature-crop 100% observed selection: all 500 frames.
Keep native cells, margin 64, observed variances and SUM prior 0.0003 (mean ridge
6e-7). Use gradient-projection/CG v1 with the same window inverse and fixed
parameters qualified in [the 25-frame ablation](scene-projection-ablation-protocol.md).
No new solver or inverse tuning is permitted within this study.

Independently start from zero at 750 and 1500 Hessian products. Each fit retains
its 300-second limit; the total study budget is 900 seconds for two fits, setup
and independent verification. Current transforms and final checks are retained
after a deadline, never forcibly interrupted. CUDA uses retain-first spectra
limited to 3 GiB with 1 GiB additional free headroom, 2 BLAS threads, and the
ordered 8-worker independent CPU verifier with one FFT thread per worker.
One scientific study runs at a time; ordinary desktop activity can overlap.

Require both fresh independent CPU relative distance bounds at most 1e-5,
feasibility, and latent/detector cap changes at most 1e-4. Do not count identical
wall-limited outputs as convergence. Preserve all inner/accepted traces and
incomplete outcomes. Completed-stage reuse requires an identical new identity;
there is no exact internal optimizer resume. No budget or tolerance increase is
inferred from a failure. This remains a single numerical endpoint; the full
selection family and scientific qualification are separate requirements.
