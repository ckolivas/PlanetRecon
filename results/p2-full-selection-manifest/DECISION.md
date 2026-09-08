# Full-development observed selection matrix

2026-09-08. The manifest contains every combination of three development seeds,
two seeing regimes, two crops and five fractions: **12 cases / 60 selections**.
Each selection retains exactly 25, 50, 125, 250 or 500 of the 500 observed frames.
Indices are unique, chronological and nested within a case. The scalar noise
estimate and SUM-prior normalization remain explicit; noisy electron sums are
not presented as true photon counts.

The [prospective protocol](../../docs/scene-selection-manifest-protocol.md) uses
only raw observed-frame Laplacian scores. It does not read truth/expected pixels,
register frames, or use optical quality scores. This differs explicitly from the
legacy registered-ranking experiment. It freezes a numerical-study selection;
it is not a reconstruction pass, a prior optimum, or a final assessment partition.

[Report](report.json), per-case records and [protocol](protocol.json) retain all
scores, selected indices and source/input identities. [Checksums](checksums.json)
protect the archived JSON. No images are distributed with this manifest.
Source and input hashes were unchanged throughout generation.

Tests cover deterministic ties, nested fractions, prior/count scaling, selection
without truth datasets, nonfinite input rejection and changed selection rejection.
All 60 numerical selection cases still require their own budget/certificate
checks; the next step is one bounded case across these fractions before multiplying
its cost across the remaining development family. Q3 remains unauthorized.
