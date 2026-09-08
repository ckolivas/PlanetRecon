# Diagonal-scaling pilot at the original 100/200 iteration caps

The objective, scene, variance maps, data, zero initialization, ridge and numerical
tolerances are unchanged from v1. The diagonal majorizer is independently checked
against the dense Hessian and nonnegative least-squares oracle.

Scalar cases now certify at 30 iterations in both budgets. The noiseless spatial
case certifies at 120 iterations in the larger budget. The noisy spatial case is
still incomplete at 200, with a relative bound of 1.95e-5 versus the 1e-5 criterion.
Neither spatial case passes the two-budget pilot because the 100-iteration solve
is incomplete. Their improvements do not retroactively certify earlier runs.

Observed residual contraction after iteration 100 is about 0.7 per ten steps,
supporting a targeted 300/600 comparison for spatial cases. The extension uses
600 s per case, based on measured iteration cost; it changes no objective or
acceptance tolerance. Scalar controls need no larger rerun. This is one bounded
numerical extension after a diagnosed conditioning fix, not full scientific
family qualification or a change to production variance weighting.
