# Bounded weak-prior extension

All three candidates passed both 1500/3000 caps and independent CPU certificates.
Training, full native domain, observed scalar variances, zero start and selection
rules were unchanged. No truth or expected images entered the fits or scores.

| SUM ridge | Iterations per solve | Selection score |
|---|---:|---:|
| 0.000003 | 1340 | 1.48699 |
| 0.00003 | 390 | 2.10371 |
| 0.0003 | 110 | 3.23533 |

The selected weak endpoint scored 1.71689 on the separate development-assessment
frames. Lower prior strength reduces the remaining held-out error, but this scan
still does not bracket an optimum. As predeclared, no automatic further grid
extension follows. The stronger endpoint exactly reproduces the earlier selection
score and iteration count, preserving the control across runs. Total wall time
was 203.30 s, with the slowest candidate requiring 1340 iterations per solve.

The scientific prior remains unselected for production. The previously declared
0.0003 SUM-prior, 11-frame setting can still be used as a fixed numerical control
for broader development coverage; that does not promote it to an optimal prior
or merge the outcomes of the two selection studies.
