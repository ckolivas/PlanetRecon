# Full-count reference stability passes at the declared 1500/3000 caps

The fixed seed-1001, Dr0=4 feature crop with all 500 selected frames now passes
both independent CPU distance certificates and the unchanged image-stability
criteria. Source, input and prerequisite identities remained unchanged.

The lower endpoint reuses the checksum-verified, certified 1500-cap scene from
the earlier cropped-reference study. Its CPU certificate was recomputed. The
3000-cap endpoint started independently from zero and converged at iteration
1410; it did not start from the lower scene. Both fresh independent relative
solution-error bounds are 9.859417126271778e-6, below 1e-5. Both scenes are
feasible. Latent and detector relative changes are exactly zero, below 1e-4.
The new upper optimizer took 617.81 seconds; the complete check took 717.17
seconds. The reused lower fit's original runtime is not counted as new work.

This qualifies numerical stability for one known-transfer objective with SUM
prior 0.0003, native cells and margin 64. The original 750/1500 study remains
incomplete because its 750-cap fit and image stability failed. Neither that
failure nor the alternative Newton failures are overwritten by this result.

Next execute the separately declared 12-case, 60-selection family with fresh
1500/3000 cap fits under the qualified backend. Every selection requires both
independent certificates and both image-stability checks. This endpoint does
not qualify the complete family, select the scientific prior, establish blind
reconstruction accuracy, authorize Q3 or change production defaults.
