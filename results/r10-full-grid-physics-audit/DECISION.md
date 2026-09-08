# Full-grid physical operator convergence controls

All 60 declared checks pass across development seeds 1001–1003, D/r0 4/8,
and feature/bland crops at the production 128x128 detector crop size.
The controls double detector padding (64/128 pixels), pupil sampling (64/128),
and exposure quadrature (8/16), retaining the original tolerances. Low-frequency
Strehl and tilt moments compare subharmonic levels 4/5 on 16 independent screens
for each seed/regime. Shared-field pupil-grid centroid checks also pass.

The run took 174.42 s and reached a cumulative process peak RSS of 621.86 MiB
on Linux. This is the peak of this control process, not a forecast for a full
500-frame atmospheric reconstruction. No GPU measurements are implied.

These results establish the declared physical discretization controls across
both crops. They do not establish complete G1/G2 reconstruction-gap stability,
global atmospheric identifiability, independent-capture quality or Q2 acceptance.
Reproduce with `.venv/bin/python -m planetrecon.physics_audit --out NEW_DIRECTORY`.
The protocol is written before execution and no historical reports are replaced.
