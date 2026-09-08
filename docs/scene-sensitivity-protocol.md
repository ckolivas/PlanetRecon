# Development prior, domain, sampling and photon protocol

Prospective protocol, 2026-09-08. This is development work, not final assessment or
Gate-1/Q2 authorization. Numerical failures remain incomplete and cannot select a
candidate. Existing simulations, certificates and historical results remain intact.

## Fixed scientific question

Determine whether the reconstructed detector field and held-out raw predictions
are sensitive to the latent boundary, cell sampling and prior strength, after
fixing image formation. Use the positive-ridge constrained objective and its
existing 1e-5 relative solution-distance certificate; compare two iteration caps
and require detector-image relative change ≤1e-4 before accepting a numerical fit.

Express the objective as a SUM of independent frame likelihoods plus a fixed
physical prior. In a mean-likelihood implementation divide the prior coefficient
by the training frame count. Doubling independent observations thus increases
information instead of silently doubling the prior. Duplicating data for a
normalization unit test is not independent scientific evidence.

Latent cells contain integrated electrons. For an area-integrated squared-density
prior, the coefficient on squared cell flux is inversely proportional to cell
area. A coarse cell spanning f×f native cells therefore uses the native coefficient
divided by f². Coarse flux is uniformly distributed onto native optical cells
before applying the SAME PSF/integration/crop operator. This explicitly tests a
piecewise-constant scene basis; it does not replace optical integration with a
rebinned PSF. Native, 2×2 and 4×4 cell representations are development candidates.

All finite-domain cells, including margins, are unknown. Compare the original
full scene with detector-crop margins 0, 32 and 64 detector pixels, intersected
with the original finite domain. Exterior is zero; no truth fills margins. At
64 pixels, a 512-cell optical PSF has no coupling to omitted scene cells; with
zero smoothness and positive zero-centred ridge, those omitted cells optimize to
zero. This gives an exact-domain control, not a heuristic permission to trim.

## Selection and assessment separation

Begin with development seed 1001, D/r0=4, feature crop as a bounded pilot, then
extend qualified candidates across all three development seeds, both regimes and
both crops. Training frames start at [0,249,499]; a larger training set, when
budgeted, is [0,55,111,166,222,249,277,333,388,444,499]. Selection uses [125,374].
Separate development assessment uses [62,187,312,437] AFTER selection. No final
scientific evaluation family is accessed or declared untouched by this study.

Select ridge only by mean held-out raw weighted squared residual, using fixed
observed-data scalar variances (max(frame mean,0)+read_noise²). Candidate native
SUM-likelihood ridge strengths are [0.0003,0.003,0.03]. This is a prospective scan,
not a selected production prior. A tie within 1e-8 relative score uses the larger
ridge. Reject the selection if any required candidate is numerically incomplete;
retain all fits, scores and failures. Assessment pixels cannot influence the
selected candidate. Truth may be read only for separately labelled AFTER-fit
simulation diagnostics; it never determines a fit, margin, initialization,
variance estimate or selection score.

## Bounded execution order

1. Measure baseline forward, adjoint and normal costs. Implement shared FFTs and
   bounded caches; verify CPU and, where available, CUDA float64 parity against
   the existing operator, objective, certificate and independent dense controls.
2. Run the three-frame full-domain ridge scan at measured budgets; freeze its
   selected ridge for the subsequent pilot domain/sampling controls.
3. Compare full/0/32/64 margins at native sampling, then 1/2/4 cell factors on the
   exact 64-margin domain. Cache exact compatible stages and retain all failures.
4. Add the larger training set and complete development family only after pilot
   cost and numerical conditioning justify a declared family budget.

Each execution records exact inputs, code/runtime identities, settings, actual
iterations, wall time and cache scope. No automatic repeated budget increases.
A failed fit or material boundary/sampling sensitivity yields an explicit next
hypothesis rather than a production setting. GPU float64 parity is numerical
qualification on the tested hardware, not general GPU or release acceptance.
