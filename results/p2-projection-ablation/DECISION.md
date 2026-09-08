# Projection phase passes the 25-frame ablation

The candidate qualifies at this endpoint; the window-Newton control fails its
smaller cap, so the complete comparison retains overall `incomplete` status.
All four fits completed in 283.46 seconds with unchanged source/input identities.
The same observed 25 frames, window inverse, exact objective and independent
CPU distance criterion were used for both modes.

| Mode | Product cap | Products used | Accepted updates | Independent distance bound | Outcome |
|---|---:|---:|---:|---:|---|
| Window Newton | 750 | 750 | 24 | 1.544855e-4 | Incomplete |
| Window Newton | 1500 | 1072 | 34 | 8.548345e-6 | Certified |
| Window projection/CG | 750 | 556 | 100 | 9.008167e-6 | Certified |
| Window projection/CG | 1500 | 556 | 100 | 9.008167e-6 | Certified |

The new mode took 88 gradient-projection and 12 face-Newton updates in each run.
Its independently started cap outputs are identical in both latent and detector
coordinates, satisfying the original 1e-4 stability requirement. Its optimizer
wall times were 43.50/43.88 seconds. The window-Newton control took 69.73/94.24
seconds and has latent/detector changes 9.23381e-5/4.57365e-6; stability alone does
not rescue the smaller-cap certificate failure. Product counts include gradient
refresh, projection, inner and line-search work; initial/final checks are separate.
These local timings are not isolated or universal performance claims.

Compared with the original periodic fit, window weighting itself improves the
existing Newton method, but both original product caps pass only after adding
the projection phase in this ablation. This supports further testing of the new
candidate on this objective. It does not prove that either change is optimal,
that the approximate inverse is exact, or that the 500-frame problem converges.
[Accepted traces and final independent certificates](comparison.png) preserve
nonmonotone stationarity bounds despite feasible objective descent.

The predeclared prerequisite for a separate 500-frame experiment is met by
`window_projection`. Keep the failed control and all earlier outcomes. Next use
the same candidate, inputs, prior, numerical thresholds and individual 300-second
fit limit at 500 frames, with a fresh prospective protocol. Full selection-family,
prior/likelihood sensitivity and scientific gates remain open; production and
Q3 remain unauthorized by this evidence.
