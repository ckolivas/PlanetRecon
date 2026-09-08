# Bounded development assessment decision

The predeclared reduced-grid audit completed all three development seeds,
both D/r0 regimes (4 and 8), both crops, both initializations and all three
15/35/60-mode stages. It used eight frames per case, one outer iteration per
stage and two phase iterations. Wall time was 139.87 seconds on CPU. Reproduce
with `.venv/bin/python -m planetrecon.science_audit --out NEW_DIRECTORY`.

The seeded partition assigns six frames to object training, one to initialization
selection and one to assessment before any fitting. The assessment object is
the selected training-only reconstruction; it is never refitted using assessment
pixels. Assessment phases start from the external shift calibration and are
profiled once with that object frozen. This measures transfer to unused frames
conditional on fitted nuisance phases. It is not an unfitted predictive likelihood
or evidence from an independent capture, and time correlation remains possible.

All 12 reconstruction and assessment statuses remain **incomplete**. The tiny
iteration budgets do not converge the atmospheric fits, and the known-transfer
controls retain their own convergence requirements. Assessment loss cannot select
an initialization or change closure. Synthetic-truth closure still uses the
corresponding all-frame reconstruction and has a separate role.

| D/r0 | Crop | Median diagnostic closure |
| --- | --- | --- |
| 4 | feature | 0.991504 |
| 4 | bland | 1.015124 |
| 8 | feature | 0.987294 |
| 8 | bland | 0.967512 |

These numbers are from incomplete reduced-grid fits and **do not pass Q2**, even
when above the numerical closure threshold. They cannot be substituted for the
historical full-resolution evaluation family. E1/E2a0 maximum image disagreement
is 4.478e-11. Per-frame optimizer termination, all partition indices and residuals
are retained in the case reports; the manifest binds each file by SHA-256.

This completes implementation and bounded exercising of the separate assessment
path. Remaining scientific work includes full-support object convergence, TV and
physical-screen constraint/budget qualification, low-frequency ranking/gap
families, residual noise calibration and resource forecasting. Only corrected,
converged, complete affected families can establish new Gate-1/Q2 outcomes.
Q3 and the production atmospheric-solver claim remain gated. The validated
baseline and tagged native build workflow remain available independently.
