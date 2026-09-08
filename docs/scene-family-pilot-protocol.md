# Complete development numerical pilot, 11 frames per case

Prospective protocol, 2026-09-08. Extend the already qualified 11-frame numerical
control to all three development seeds, D/r0=4 and 8, and feature/bland crops:
12 required cases. This is a complete development PILOT matrix, not the 500-frame
family with all selection fractions, scientific gap tables or Q2/Q3.

Freeze the previously tested SUM prior 0.0003, native cells, exact 64-detector-pixel
influence domain, zero initialization, observed-data scalar variances, and training
frames [0,55,111,166,222,249,277,333,388,444,499]. This is a numerical reference
objective, not a claim that this coefficient is an optimal scientific prior.
The weak-prior studies and their unbracketed endpoint remain separate evidence.

Require the unchanged 1e-5 scene-distance certificate, independently recomputed
on CPU, at both 750/1500 iteration caps. Require latent and detector relative
budget changes ≤1e-4. The existing 11-frame case needed 220 iterations; the caps
allow a fixed regime/crop conditioning margin and doubled-budget comparison.
Use at most two CUDA worker processes, each with two CPU threads and a 256 MiB
retained-PSF-spectrum cache. These are cache/thread bounds, not total process or
VRAM ceilings. Set 900 s per case and 3600 s for the family, checked between
iterations/stages. Include setup, checkpoint I/O and CPU checks in wall/RSS reports.

Checkpoint each solver's iterate and momentum and commit every completed stage.
Resume only exact input/protocol/runtime/code/solver identities; doubled-budget
fits have separate states. Preserve failures and complete case coverage. A case
or budget failure makes the family incomplete; do not tune or raise all budgets
in response. No assessment or truth pixels are read, and no scientific opportunity
or production-quality decision follows from numerical convergence alone.
