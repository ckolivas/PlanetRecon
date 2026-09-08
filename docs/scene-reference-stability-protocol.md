# Prospective 1500/3000 full-count reference stability check

The completed 750/1500 cropped-reference experiment remains incomplete. Its
1500-cap fit independently meets 1e-5 at 1410 iterations; its 750-cap fit does
not. This new experiment checks stability at the numerically adequate lower cap.
It does not relabel that earlier experiment or weaken any accuracy criterion.

Freeze the existing 500-frame stage at
`out/p2-cropped-reference/stages/100/budget-1500.npz`, SHA-256
`abbdcdaad5a4df0099a3fad842e8c9992a8e157531c9a0a0229177b1f11e18aa`.
Verify its original protocol/selection identity, commit manifest, image checksum,
reference v4 metadata, cap 1500 and 1410 completed iterations. The stage is reused
as the lower comparison image, not rerun or used to initialize the upper solve.
Bind the original report, protocol, stage files, input and manifest hashes.

Use the same seed-1001, Dr0=4, feature crop, all 500 observed frames, native cells,
margin 64, observed variances, SUM prior 0.0003, exact-crop CUDA backend and 4 GiB
retention with 1 GiB additional free headroom. Verify original numerical source
and package identities. Two BLAS threads and eight ordered independent CPU frame
workers (one FFT thread each) remain in effect.

Recompute an independent CPU certificate on the frozen lower image. Run a new
reference v4 fit from zero at cap 3000 with the same stopping tolerance and a
900-second fit limit. The previous certified run took 685 seconds at 1410
iterations, so this allowance is retained; the larger cap does not guarantee
success. Total new-study budget 1200 seconds includes setup and independent
verification. Current operations/final checks complete after a deadline.

Require both fresh independent CPU bounds at most 1e-5 and feasibility, plus
latent and detector image changes at most 1e-4. Preserve every incomplete result.
The upper fit has its own checkpoints/stage namespace; completed-stage reuse is
exact-identity only and does not add time to a completed wall-limited stage.
Report lower-cap reuse explicitly, without counting its old runtime as new work.
A pass qualifies this one full-count endpoint at these caps; the full selection
family, prior/sensitivity work and scientific gates remain separate requirements.
