# Cropped-backend endpoint comparison

Both methods pass the 25-frame endpoint at both original caps. At 500 frames,
the reference's 1500-cap scene alone meets the independent 1e-5 distance target;
the 750-cap reference and both Newton caps fail. Both methods also fail the
original full-count doubled-cap image stability test. The comparison therefore
promotes neither method to full endpoint-family qualification.

The original source/input, selection, prior, domain and criteria match. The
runner's reporting-only reference cap-field correction does not alter the
numerical solver or backend. Two 25-frame pairs are independently certified and
their objectives are consistent with the summed gap bounds plus the declared
roundoff allowance. Incomplete pairs are not treated as accuracy baselines.
[Independent bounds and separate progress axes](comparison.png).

The normal-call accounting is checked against actual solver calls, including
interruption and reference resume, and verifies the solver source hashes before
analysis. For the full-count caps, reference uses 827/1553 optimizer normal
products and Newton uses 752/1502. These include initial/final gradients, exclude
constructor setup, objective forward calls and independent CPU verification,
and do not imply equal wall time or accuracy. The reference's higher cap stops
at its accuracy target; the failed Newton runs exhaust their product budgets.

Choose the certified reference stage for the next explicitly declared
1500/3000 stability check. Preserve these original cap failures and all prior
experiments. No prior choice, full selection-family qualification, Q2/Q3 or
production adoption follows from this one-case comparison.
