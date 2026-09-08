# Observed-data prior pilot

All three fixed-grid ridge candidates passed 300/600 iteration budgets and
independent CPU solution certificates. The full native optical scene and zero
initialization were used, with no truth-supplied margins, expected-image values
or truth-derived variance maps. Variances are max(observed frame mean,0)+read².

| Native SUM-likelihood ridge | Iterations per solve | Selection score |
|---|---:|---:|
| 0.0003 | 110 | 3.23533 |
| 0.003 | 30 | 13.39233 |
| 0.03 | 10 | 300.12127 |

Training frames are 0/249/499; selection frames are 125/374. Only after choosing
0.0003 by held-out raw residual did the runner load development-assessment frames
62/187/312/437, which scored 2.56987. Score units are mean squared raw residual
per estimated scalar variance. The mean-likelihood implementation divides each
SUM prior by three training frames; coefficients are not silently reused across
photon counts or sampling areas.

The selected candidate is the weak-prior boundary of the prospectively declared
grid. This is not a demonstrated optimum or production prior. Freeze this value
for the declared domain/sampling/photon pilot controls; a wider prior scan needs
a separate prospective protocol. This single seed/regime/crop is development
selection, not untouched final scientific assessment or a Gate-1/Q2 pass.
The whole study took 27.85 s on the qualified CUDA path, including CPU checks.
