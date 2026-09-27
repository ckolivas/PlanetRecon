# Automatic Saturn radii: mismatched inner edges

The Saturn R capture failed automatic geometry discovery because each ansa
independently chose its first positive radial-gradient peak. A fainter interior
feature was detected on just one side, so the estimator compared different
structures. All three aligned 32-frame windows failed the consistency gate,
leaving the globe and both ring radii blank and preventing Stack from running.

At the 256-pixel analysis width, the rejected pairs were approximately
54.34/45.54, 46.45/53.25 and 53.88/46.48 pixels. Each window also contained a
matching boundary near radius 54 on both sides. The estimator now pairs radial
peaks before selecting the innermost common edge. Existing contrast and peak
prominence requirements remain; pairs must agree within the larger of 1.5
analysis pixels or 2% of the outer radius. A missing or inconsistent boundary
still produces an unresolved result. No planetary radius ratios are assumed.

The matched pairs are 54.34/54.94, 54.25/53.25 and 53.88/54.48. The full limb
fits now agree across all three windows. Versioned Saturn geometry records
cause old estimates to refresh through the existing Stack preprocessing flow,
while leaving cached quality measurements usable and manual overrides intact.

Verification used the full `2024-09-27-1154_3-CK-R-Sat.ser` capture, starting
with blank globe/ring radii, field rate and viewing-latitude controls. Fresh
preprocessing and GUI prefilling produced:

| Setting | Measured value |
|---|---:|
| Globe equatorial radius | 96.9444 px |
| Visible inner-ring boundary | 147.3051 px |
| Outer-ring boundary | 221.3032 px |
| Field rotation rate | -6.3687e-8 rad/s |

The resulting configuration passed cache reload and completed a CUDA circular-AP
stack of the best 1% of screened frames: 255 used, with the other 27,434 excluded
by the selected masks. No manual radius values were supplied. The viewing
latitude came from the existing capture-UTC ephemeris cache. Configuration,
geometry, logs, raw snapshot and exports are in `out/saturn-radius-fix/`.
This verifies automatic geometry and execution, not an improvement in image
quality over other stackers.

Tests: 105 geometry/cache/UI regressions passed (two CUDA skips), and a separate
CPU/CUDA Saturn, local-alignment, refusal and checkpoint run passed 85 tests.
The small recorded-profile fixture reproduces all three original failures;
additional tests retain refusal for unmatched, one-sided and weak boundaries.
