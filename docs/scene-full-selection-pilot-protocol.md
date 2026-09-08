# Full-selection endpoint numerical pilot

Prospective protocol, 2026-09-08, after the completed full-count resource profile.
Use seed1001, D/r0=4, feature crop, at the 5% and 100% endpoints of the frozen
observed selection manifest. All five fractions remain required in the eventual
60-selection family; this pilot is only two selections (25 and 500 frames).

Freeze native cells, margin64, scalar observed variance, SUM ridge .0003, zero
initialization and the existing projected-acceleration solver. Use separate
750/1500 caps, unchanged independent CPU scene-distance tolerance1e-5 and
latent/detector budget stability1e-4. No scientific prior selection is implied.

The measured 500-frame normal operation costs2.4s and its independent CPU check
102s under the recorded workload. Bound each iterative fit to300s and the pilot
to1200s, with one CUDA worker, two CPU threads and a256MiB retained-spectrum cache.
Setup and final checks can cross the deadline because they are indivisible
operations. This intentionally bounds the conditioning probe: the all-frame fit
may remain incomplete. Record actual iterations, certificates and cache/RSS.
Do not silently raise caps, relax tolerance or multiply failures into the family.

Retain exact iterate/momentum checkpoints at100-iteration intervals and completed
stage commits. Resume only unchanged input, manifest, code, runtime and protocol;
retain each incomplete attempt separately. Two caps always have independent
states. A timed-out fit is never stored as a completed reusable stage.

Pass this pilot only if both endpoints pass both budgets and image stability.
Missing or timed-out fits are incomplete, not evidence of a physical limit.
Use conditioning and resource evidence to select at most one alternative solver
under P2 if needed, preserving exactly the same objective and CPU certificate.
No production integration or Gate-1/Q2/Q3 authorization follows from this pilot.
