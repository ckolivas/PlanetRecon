# Conserving observation density in a periodic inverse

The first periodic inverse improves the saved near-solution linear probe, but
regresses the constrained25-frame experiment from zero. One concrete approximation
gap is the observation window: the initial symbol uses a detector sampling
density1/b^2 across the whole native scene, including its unobserved margin.

Let N be native spatial cell count, m the detector pixel count, and Q_k the
Fourier transform of the integrated PSF. A unit-norm Fourier mode has modulus
1/sqrt(N). When each valid detector footprint has complete PSF support inside
the native scene, its data-Hessian Fourier diagonal is

    sum_k flux_k^2 * abs(Q_k)^2 * sum_detector(weights_k) / N.

The original periodic data symbol instead uses sum(weights)/(b^2*m). Multiplying
that data term by the geometry-derived window fraction b^2*m/N conserves the
total observation weight. This factor is not fitted to reconstruction accuracy.
For native1024 by912, detector128 by128 and b=4, the fraction is about0.2807.

`WindowAveragedPreconditioner` implements this candidate for common detector
sampling. It retains the original ridge and periodic smoothness terms. The result
is a convex combination of the old positive symbol and its positive regularizer,
so the inverse stays positive on full and reduced subspaces. Exact Fourier-
diagonal agreement applies to the data term with complete PSF footprints and
zero smoothness; finite boundary truncation and the original nonperiodic
smoothness still require separate controls. Sampling aliases and cross-frequency
coupling are not removed from the exact physical Hessian.

Three dense controls pass: agreement with the exact finite-detector Fourier
diagonal for interior PSF footprints and nonuniform weights; unchanged results
for full detector coverage; and positive reduced RGB/Bayer inverse eigenvalues.
This does not establish better full-size solves or authorize production adoption.

Before another reconstruction experiment, compare diagonal, original periodic
and window-aware inverses on early and near-solution states for both25 and500
frames. Validate all state/input identities and independently recompute reduced
residuals. Include finite-boundary and active-mask sensitivity; the single
near-solution500-frame probe was not representative of the from-zero trajectory.
Predeclare product counts and runtime once this diagnostic is implemented.
Keep the300-second historical fit failures, prior and numerical tolerances intact.
