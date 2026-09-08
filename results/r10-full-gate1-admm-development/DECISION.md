# First full-resolution quadratic candidate

All 12 development seed/regime/crop cases were executed at 500 frames and
128×128 resolution, using the unchanged Gate-1 quadratic and 10000/20000 ADMM
iteration caps. The input files were certified before the study, and package
and candidate source identities remained unchanged during execution.

Images and relative gaps are stable: maximum relative image change is
2.895e-5 and maximum absolute G1/G2/G3 change is 4.071e-5. All cases nevertheless
remain incomplete because at least one constrained solve misses a declared
stationarity or solution-error-bound check. Descriptive gap tables are retained
but cannot certify a Gate-1 decision. Total wall time is 531.35 seconds with four
two-thread workers, alongside input generation.

The first candidate incorrectly attempted to satisfy complex Fourier
stationarity after fractional registration on an even detector grid. The
production FISTA gradient takes the real inverse FFT, so its real-image
quadratic has Hermitian-averaged linear coefficients and reflected-averaged
curvature. Dividing before restricting to real images does not solve that same
quadratic. A dense spatial-matrix regression now covers the distinction; the
corrected candidate is rerun in a separate directory with unchanged tolerances.

The production estimator is not replaced by this experimental audit candidate.
This study does not qualify Q2/Q3 or fix the separately measured crop-forward
model mismatch. The first, much slower FISTA attempt is preserved in
`../r10-full-gate1-development` as incomplete, with no numerical conclusion.
