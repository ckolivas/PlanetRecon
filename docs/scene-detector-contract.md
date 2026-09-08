# Extended scene-to-detector contract, version 1

This is an experimental reference operator for P1/P2, separate from the legacy
crop-periodic Gate-1 estimators and the GUI baseline. It does not authorize Q2/Q3.

## Coordinates and units

A latent scene is an H×W optical-cell array, or H×W×3 RGB array. An entry is
integrated source flux in that cell at the declared reference epoch. Array
coordinates are +x right, +y down. Optical cells [by:by+b, bx:bx+b] form one
detector pixel; integration is a sum, not an average. `flux` converts scene units
to expected electrons for the whole exposure, including exposure duration once.

Each exposure sample applies a declared translation in optical pixels, then a
unit-energy optical PSF through zero-extended linear convolution. Translations
use a piecewise bilinear deposition of cell flux; the adjoint gathers exactly the
same weights. Fractional translations at this sampling are a declared numerical
approximation, not exact continuous optics. No periodic Fourier shift or complex
Nyquist coefficient is hidden in this operation. Large shifts lose flux beyond
the declared extended scene; that boundary loss must be included in sensitivity
checks. Full optical PSFs already containing pupil tilt require no extra shift.

Convolution uses the existing simulator's `same` origin: full-convolution indices
start at floor((kernel_size−1)/2). For an even kernel the PSF sample at n/2 appears
one optical cell after this origin. This is intentional compatibility with the
simulator, not an implicit change to the centred crop-Fourier convention.

Exposure quadrature weights are nonnegative and sum to one. Weighted sample
predictions are summed before detector integration and crop. A PSF per sample
allows correlated atmosphere/motion. A pre-averaged PSF is exact for a fixed
scene because image formation is linear. Time/pose estimation is upstream;
operators accept already computed reference-relative translations. Field/surface
warps and their phase derivatives are subsequent extensions.

The detector crop has an explicit (x,y) origin on the full binned scene grid and
a rectangular shape. A binary valid-pixel mask is applied after crop. Mono and RGB
are supported; raw CFA observes one RGB component according to the full detector
coordinate parity, pattern and declared detector parity offset. Cropping does not
reset CFA parity. Display interpolation is never part of a measurement operator.

## Boundary and prior

Every cell of the finite extended scene, including unobserved margins, is an
unknown in reconstruction. Beyond that finite domain the forward operator uses
zero flux. No simulation truth is used to fill unknown margins or initialize a
real reconstruction. Domain extent and any boundary prior belong to the study
protocol; changing them creates a new scientific question.

The initial constrained reference objective uses nonnegative scene cells and an
explicit positive ridge prior, optionally with nonperiodic adjacent-cell
smoothness. This removes unobserved null directions with a declared prior. It is
not the legacy crop-Fourier support constraint: the optical PSFs provide the
transfer cutoff, while latent cells remain unknown at the chosen sampling.
Prior/domain sensitivity and comparison with an optical-support representation
must precede any claim of full scientific qualification.

## Validation and acceptance

Independent spatial convolution, detector sums, coordinate impulses, constants,
structured and limb-crossing scenes, odd/even kernels, fractional translations,
all four Bayer patterns/parities, adjoint dot products and objective derivatives
must pass at float64 numerical accuracy (relative 1e-10 for bounded controls).
Interior flux is conserved for unit-energy PSFs; finite-domain/crop losses remain
physical losses, with no edge renormalization.

Physical controls must retain the original full optical PSFs (or regenerate them
from unchanged simulation dependencies), compare with independently stored means
and keep float32 serialization error separate from model bias. Cropped detector
PSFs cannot reconstruct the missing optical kernel or exterior scene.

For a sequence, report per-frame residuals divided by noise variance AND the
squared bias of the summed stack divided by its summed independent-noise
variance. The latter detects persistent sub-noise per-frame bias. Numerical
agreement with the stored mean is a consistency test, not a claim that simulated
optics are adequate for real captures. Full reconstruction accuracy/information
loss criteria must be frozen before evaluating an approximation or trimming a
usable field. No automatic crop trimming is authorized by this contract.

## Reference solver certificate

`tools/scene_quadratic.py` implements the explicit matrix-free objective above,
with likelihood averaged over frames. Weights are frozen inverse variances.
Its initial iterate is zero unless an explicit initialization is supplied; every
margin is included. A strictly positive ridge gives strong convexity at least
`ridge`. The gradient Lipschitz upper bound uses nonnegative operator row/column
sums plus the ridge and the nonperiodic difference bound (8×smoothness).

For a feasible scene, choose the normal-cone vector so the residual is the
gradient at positive cells and min(gradient,0) at exact zero cells. Strong
convexity gives distance-to-optimum ≤ norm(residual)/ridge and objective gap ≤
norm(residual)²/(2×ridge). These bounds apply to this discrete regularized
objective; they do not bound distance to the physical truth. The reported
relative bound divides by max(norm(scene),1). No optical-support or identifiability
claim follows from this certificate. Exhausted budgets remain incomplete.

The v2 solver uses a diagonal majorizer to avoid limiting every scene cell by the
largest spatial-noise curvature. For the nonnegative data Hessian H,
diag(H·1)−H is a graph Laplacian and is positive semidefinite. The nonperiodic
smoothness term is majorized by twice its diagonal degree. Adding the ridge
therefore bounds the full Hessian. Optimizing in coordinates sqrt(diagonal)×scene
preserves positivity and the objective. The certificate is still evaluated in the
original scene coordinates with the original ridge, independently of scaling.
Dense tests check both the majorizer's positive-semidefinite difference and
agreement of global/diagonal methods with the same constrained optimum.
