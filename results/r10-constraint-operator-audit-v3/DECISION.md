# Bounded W01 operator and constraint decision

Reproduce with `.venv/bin/python -m planetrecon.constraint_audit --out NEW_DIRECTORY`.
Protocols are written before execution; source identities and per-solve diagnostics
are retained. These are 8x8 known-transfer quadratic problems, not a scientific
seed family or an atmospheric solver qualification.

All 16 composed forward/adjoint comparisons pass the declared 1e-11 relative
tolerance against direct spatial/Fourier-series matrices. The maximum error is
9.330e-16 across rectangular scenes, odd/even kernels, two bin factors, crop
positions, flux scaling and integer/fractional registration. E1/E2a0 agreement
is at most 9.705e-10. The dense SLSQP oracle succeeds in all six cases.

The archived v2 run isolated a certification bottleneck: two band-limited solutions
agreed with the independent oracle within 7e-10 in relative image norm, while a
cold final Dykstra projection failed its convergence check. Version 1.4 reuses
the latest gradient-step projection duals, rebased onto the stationarity target;
it retains the same projection tolerances and feasibility requirements.

In v3 all three band-limited cases pass from both initializations at budget 128.
The maximum oracle image discrepancy for those six solves is 5.839e-6; the largest
relative objective gap is 2.330e-11. The positive cosine initialization replaces
v2's constant, whose DC difference is erased by the first gradient step.

The full-support controls remain incomplete even at 512 iterations. Their relative
image errors range from 1.826e-4 to 1.985e-3 and relative objective gaps from
1.750e-9 to 9.721e-8. Small objective gaps do not override failed stationarity or
iterate tolerances. This separates a corrected projection-check bottleneck from
remaining accelerated-solver convergence work; it does not declare W01 complete.

The constant brightness noise control has zero scalar-variance mismatch. The
structured control has variance relative RMS 0.8521, inverse-weight relative RMS
0.6079 and variance dynamic range 8.225. These are controlled brightness-map
statistics, not real-capture residual calibration, and the likelihood is unchanged.

The initial pre-v2 12-case/512-iteration attempt was stopped after one case for
cost, before a family report existed. The completed v2 and v3 protocols and
results are preserved separately. Neither run authorizes Q3. TV, complete
physical-screen convergence/ranking families, independent assessment and
documented real captures remain separate requirements.
