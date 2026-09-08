# One bounded extension of the development prior scan

Declared 2026-09-08 after the original scan, boundary, sampling and photon pilots.
The original candidate grid selected its weakest endpoint. The next hypothesis
is that the remaining held-out residual is prior bias, rather than a numerical
error or an inadequate extended boundary. Test weaker SUM strengths
[0.000003,0.00003,0.0003], retaining the already tested endpoint as a control.

Keep the original three training frames, selection/assessment separation,
full native optical scene, observed scalar variances, zero initialization,
float64 CUDA path and independent CPU certificate. Do not change the 1e-5
solution-distance or 1e-4 budget-stability tolerances. Use 1500/3000 iteration
caps and 300 s per solve within 1800 s total. The existing weakest fit needed
110 iterations; a 100× smaller strong-convexity lower bound motivates allowing
roughly 10× more iterations, with a fixed margin. This is a cost hypothesis,
not a promised convergence rate for active constraints.

Retain every failed candidate. Select only if all three pass both budgets.
If the weaker endpoint wins again, report that the optimum is still unbracketed;
do not automatically extend the grid again in this study. Assessment remains
separate from selection but is development data, not untouched final evaluation.
The previous scan and its selected settings remain historical evidence.
