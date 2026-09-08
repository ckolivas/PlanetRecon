# Frozen full-count conditioning diagnostic

The declared identity and existing-majorizer probes both completed32 products
on the preserved iteration142,500-frame reference checkpoint. Source, input,
manifest and checkpoint hashes stayed unchanged. Total wall time334.69s; no
scene update or historical convergence reclassification occurred.

| Scaling | Terminal recursive residual / initial | CPU/CUDA product relative error | Checked relative distance bound |
|---|---:|---:|---:|
| Identity | 2.06754 | 7.09e-15 | 0.149922 |
| Existing majorizer | 1.98179 | 3.43e-14 | 0.149922 |

Neither correction tightened the original0.149922 bound. Euclidean CG residuals
need not decrease monotonically, so their increase alone is not divergence.
Trace Rayleigh quotients are directional measurements, not certified extreme
eigenvalues. Initial CPU/CUDA gradient relative error3.87e-11 passes1e-10.
The active exact-zero fraction is33.49%.

The valid global upper Hessian bound is0.0143578 and ridge is6e-7. The resulting
23929.7 condition-number upper bound is not a measured condition number. The
[algebraic interpretation](../../docs/scene-conditioning-interpretation.md)
shows that even an exact inner solve cannot report less than0.000969164 using
this fixed-normal-cone energy/global-ridge certificate. This exceeds1e-5, but is
**not a lower bound on actual scene error** or on other certificate families.
It rules out relying only on a better inner correction to certify this saved x.

The independent eight-worker CPU verifier was adopted under a new study identity.
Dense constrained controls, checkpoint/provenance checks and interpretation tests
preceded this decision. Correction arrays remain in ignored local output; archived
JSON contains metadata and traces, without image payloads. All500-frame fits stay
incomplete, scientific prior selection stays unresolved and Q3 remains closed.

Next bounded work: compare a dense-verified true Hessian-diagonal preconditioner
on the same frozen iterate. A subsequent optimizer or certificate change requires
its own mathematical controls and prospective protocol; do not raise budgets or
relax tolerances automatically.
