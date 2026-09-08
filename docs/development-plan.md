# Development plan

Updated 2026-09-09 with certified full-count reference stability and the full-family protocol.
This tracks completed implementation and the remaining qualification sequence.
It supersedes the ordering of unfinished work in the historical roadmap;
completed features and archived scientific results remain intact.

## Image-output priority

The objective is better reconstruction using more frames than conventional
quality-cutoff stacking. All screened frames remain the default. The optional
quality-range midpoint cutoff (strictly above capture worst/best midpoint) and
ranked frame-count selection are comparison controls to steer improvements,
not a replacement objective or a default rejection policy.

Next, address atmospheric deformation and blur so additional frames contribute
useful information. Compare the all-screened result against the optional cutoff
under the same reference and settings, without sharpening. A first local
registration pilot using a single noisy frame regressed on Jupiter (128 frames:
matched RMS 0.006628 global versus 0.009514 local); it was not adopted.
The [disjoint-frame template pilot](../results/real-data/jupiter-template-pilot.json)
now evaluates 64 independent high-quality template frames against the same 128
evaluation frames. Template-only global alignment improves RMS by just 0.064%;
template-based local warping still worsens RMS and fine-detail correlation.
Neither is integrated. Next examine local displacement reliability and separation
of brightness/blur changes from motion before another bounded image comparison.
Recorded SER exposure now supplies geometry midpoint timing automatically where
the capture header includes it; manual values retain priority. Require actual image improvement before integration; defer broad
qualification matrices and reporting work that do not change application output.

## Execution progress (2026-09-09)

- **P0:** stage checkpoints plus atomic iterate/momentum resume now cover the new
  solver and sensitivity/family runners. Exact identities bind input, protocol,
  runtime and source. Deadline failures retain resumable states and separate
  incomplete records. Historical runners still need migration where used.
- **P1:** the extended scene operator matches all 3,000 development frames.
  The finite optical influence domain with a 64-detector-pixel margin reproduces
  the full-domain constrained solution to rounding in the pilot, with an exact
  support argument for this zero-centred ridge objective. Using only the detector
  crop changes the result by 36.4%; arbitrary margin removal is unsafe.
- **P2:** dense numerical controls, noise pilots, observed-only prior selection,
  domain/cell/photon normalization studies, and the complete **12-case, 11-frame
  numerical pilot** pass their declared checks. All 24 family fits satisfy the
  independent CPU 1e-5 distance bound and doubled-budget image stability.
  This does not qualify the full 500-frame selection family or scientific quality.
  The single declared weak-prior extension again selected its weakest endpoint;
  the prior optimum is unbracketed and the extension has stopped as declared. The full
  observed selection manifest is now frozen; the 25-frame endpoint passes both
  original solvers. A safeguarded Newton-CG implementation now also passes the
  25-frame endpoint (13 updates, 293 products, independent bound 6.29459e-6).
  The original endpoint studies remain incomplete at 500 frames under their declared budgets.
  Newton-CG reaches nine updates and bound 0.167701 in each 300-second fit;
  identical terminal images across caps do not qualify an unconverged solve.
  Three frozen-iterate 32-product probes (identity, existing majorizer and true
  Hessian diagonal) now pass independent CPU/CUDA checks but do not tighten the
  saved iterate's error bound. An algebraic limit rules out qualifying that
  iterate using only this fixed-residual energy/global-ridge certificate family;
  it does not establish a lower bound on actual reconstruction error.
  A positive periodic inverse now improves independent fixed-state residuals
  (full 1.9818 to 0.0292; reduced 1.6232 to 0.1717), but its subsequent constrained
  experiment from zero regresses: neither 25-frame cap passes, and both 500-frame
  fits stop with bound 38.803. It is not adopted. Inner progress is now retained,
  including interrupted directions. A geometry-derived window-average inverse
  now has twelve independently checked state probes, but the later projection
  candidate still fails its full-count extension. The qualified cropped FFT
  backend now allows the reference to certify 500 frames at iteration 1410.
  A separate 1500/3000-cap stability study passes both fresh independent CPU
  certificates (9.859417e-6) with identical images. The original 750/1500 study
  remains incomplete. The complete 60-selection family is the next requirement;
  no scientific prior or production adoption follows from this one endpoint.
- **P5 early work:** shared CPU/CUDA scene FFTs, bounded retained PSF spectra and
  exact iteration resume are implemented. In the three-frame benchmark, solve
  time was 20.37 s for the reference and 1.51 s for shared CUDA, with a 3.97e-15
  relative solution difference. This is a measured pilot, not a whole-job or
  500-frame reconstruction performance guarantee. Full-count operator parity and
  bounded parallel independent CPU verification now pass, with archived resource
  profiles. Fixed cache admission at 3 GiB now preserves 181 PSF spectra across
  full sweeps and reduces local median product time from 1.35 to 1.20 seconds
  with unchanged arithmetic. Larger LRU alone still has zero reuse. Full spectra
  require 8.86 GB; available-memory headroom remains part of each new protocol.
  Translated exposure fast paths remain absent.
- **P7 intake:** all seven local SER files are now hashed and inventoried without
  image pixels or private filenames. IR642/L3 remain RGGB; three Saturn R/G/B
  files are mono. OSC Saturn and both Mars files contain duplicate timestamps,
  so the current timing contract reports no valid duration. Independent group
  provenance and distribution permissions remain unknown; true mono Mars is absent.
  Bounded raw-pixel checks found distinct frames in all 64 sampled equal-timestamp
  pairs per affected capture. Timestamp duplication alone cannot justify deleting
  a frame; no timing repair or cadence inference follows from these samples.
- **P3/P4/P6–P8:** likelihood/phase qualification, scientific requalification,
  production integration, independent captures and release refresh remain.
  Q3 is not authorized. Windows/macOS runtime tests remain excluded.

See [current status](status.md), [operator evidence](../results/p1-scene-detector-full/DECISION.md),
[family evidence](../results/p2-development-numerical-pilot/DECISION.md),
[prior endpoint decision](../results/p2-prior-extension/DECISION.md) and
[performance evidence](../results/p5-shared-fft/DECISION.md),
[full-count profile](../results/p5-full-count-profile/DECISION.md),
[CPU verifier](../results/p5-parallel-reference/DECISION.md) and
[endpoint comparison](../results/p2-selection-endpoints-comparison/DECISION.md),
[fixed-iterate certificate limits](../results/p2-frozen-conditioning/DECISION.md)
and [Hessian-diagonal comparison](../results/p2-frozen-jacobi/DECISION.md),
[Newton-CG endpoints](../results/p2-selection-endpoints-newton/DECISION.md),
[retained-cache profile](../results/p5-retained-cache/DECISION.md) and
[capture intake](capture-intake.md),
[coupled fixed-state probes](../results/p2-periodic-probe/DECISION.md) and
[constrained regression](../results/p2-selection-endpoints-periodic-newton/DECISION.md).

## Direction

**Correct image formation, establish numerical accuracy on that model, then
qualify blind reconstruction and its practical benefit.** Maintain a usable
baseline throughout. Application reliability, capture acquisition and release
preparation can proceed independently of a positive atmospheric result.

The latest [full-resolution evidence](../results/r10-full-grid-summary/DECISION.md)
shows why this order matters:

- All 30 development/evaluation simulation inputs are available and certified
  under their recorded method. That certification does not certify a new solver.
- Only 51 of 120 constrained solves pass at the larger corrected-candidate
  budget; every development case still contains an incomplete solve.
- Images and gaps are stable between budgets, but stationarity and error bounds
  do not all pass. Stability alone cannot certify convergence.
- Crop-forward discrepancy is large relative to detector noise, and the metric
  taper does not remove it. Per-pixel noise weighting worsens all 24 tested
  subset/crop reconstructions under this approximation.

The current GUI, RGB/Bayer baseline, cached preprocessing, best-frame reference,
SER timing, geometry/resume, scientific export, local CUDA support and native
tag-build automation are existing foundations. They need regression protection
and qualification where incomplete, not wholesale reimplementation.

## Current priority: application image quality

The user's latest direction prioritizes changes that improve reconstructed
application images. Best-frame percentage selection is implemented in the GUI
and CLI, reusing cached quality measurements. The unsharpened Jupiter comparison
at 25/50/100% by frame count found a 5.2% relative reduction in matched RMS at 50%, with slightly
lower fine-detail correlation; 25% worsened both metrics. Keep sharpening separate
and retain 100% as the default. Next implement and evaluate local registration
against the global-translation baseline, using real Jupiter and controlled
spatially varying motion. Adopt changes only when image detail improves without
introducing colour, boundary or noise artifacts; do not expand bookkeeping tools.

The second numerical case was interrupted without a terminal report; its four
completed selection records are preserved and its full-count endpoint remains
incomplete. Leave the remaining broad numerical matrix pending. Resume that work when it is needed
for a specific image-reconstruction candidate. Do not add standalone progress,
accounting or reporting tools as development milestones. The scientific
qualification requirements below remain requirements for advanced claims;
they do not block improvements to the existing capture-stacking application.

## Delivery order

| Phase | Work and deliverable | Completion condition |
|---|---|---|
| P0 | Consolidate current status and experiment controls | One current status matrix; immutable, resumable experiments with explicit budgets and dependencies |
| P1 | Shared scene-to-detector forward and adjoint | Independent numerical agreement and quantified model error across frames, boundaries, sampling and motion |
| P2 | Qualified known-transfer reconstruction | Feasibility, stationarity, independent controls and budget stability pass across the full development family |
| P3 | Noise and phase/exposure model qualification | A development-frozen likelihood and identifiable phase model with measured sensitivity |
| P4 | Full scientific requalification | Complete Gate-1 and scoped Q2 results, including valid negative outcomes and untouched assessment |
| P5 | Scalable CPU/GPU execution | Measured parity, bounded resources, checkpoint/recovery and practical runtime for qualified operators |
| P6 | Physical solver in the real-capture application | End-to-end raw-data inference with geometry, diagnostics and truthful result status |
| P7 | Independent capture and user-workflow acceptance | Predeclared real-data comparisons and supported operating limits demonstrated |
| P8 | Release completion | Updated native artifacts, Linux functional checks, documentation and tagged GitHub release requirements ready |

P1–P4 are the scientific dependency chain. Small operator/backend parity work in
P5 can begin after P1; large Q3 studies require P4. Corpus collection and baseline
application/release maintenance can proceed throughout. P6 depends on the
qualified components from P1–P5; advanced claims also require P7.

## P0 — Make the remaining work reproducible and bounded

Reconcile the README and roadmap into an implemented/qualified/experimental/
deferred matrix. Remove obsolete current-status claims, including old IR642
mono interpretation and already completed resume/build work, while retaining
dated historical evidence.

Make experiment execution checkpoint each crop, subset, initialization and
solver stage. Preserve partial results on cancellation, failure and time limits.
Resume only when input, configuration, operator, solver and protocol identities
match. Record wall time, actual iterations, memory and background workload.

Separate simulation provenance from reconstruction qualification without
weakening either. Retain full source identities and explicit dependency hashes;
reuse existing data only after compatibility checks. Never refresh old pass flags
merely by rewriting a fingerprint. Test rejection of changed physics, sampling,
calibration and solver contracts.

Each study must declare a question, cases, budgets, tolerances and a decision rule
before execution. Use measured pilot cost to set the family resource budget.
An exhausted budget produces an incomplete result and a specific next hypothesis,
not an automatic larger rerun. Commit each independently reviewable step.

## P1 — Correct the scene-to-detector model first

Extend the existing `operators.py` forward/adjoint work into the reconstruction
path. Specify coordinate origins, even-grid centring, pixel integration, flux
units and reference times explicitly. The model must include:

- An extended latent scene, padded linear optical convolution, detector
  integration, actual detector crop and valid-pixel masks.
- Exposure integration where required, with translation and later field/surface
  motion represented in the forward model.
- Detector-coordinate Bayer sampling for colour. Interpolated previews must
  remain separate from raw measurements used by a likelihood.

Choose the extended-scene representation and treatment of unobserved margins
explicitly. Compare reconstructing nuisance margins with a declared boundary
prior; do not fill them with simulation truth. Define optical support on the
chosen scene representation instead of carrying over crop-periodic assumptions.

Validate impulse, constant, structured and limb-crossing scenes; odd/even kernels;
fractional shifts and Nyquist coefficients; all CFA parities; and flux conservation.
Use independent spatial calculations, adjoint dot products and derivative checks.
Then evaluate varied physical frames across both crops and seeing regimes.

Measure both per-frame mismatch and its correlated contribution to an entire
stack. A residual smaller than single-frame noise may still dominate a long
stack. Freeze acceptable bias against reconstruction accuracy and information
loss before evaluation. An interior approximation is permissible only if it
passes this test with a declared usable field; trimming 32 pixels is not an
automatic solution.

**Deliverable:** a tested operator contract, reference implementation and model
error report, ready for known-transfer reconstruction.

## P2 — Qualify a solver for the corrected operator

Keep the crop-Fourier solver as a legacy control. Its diagonal inversion and
current ADMM distance certificate do not automatically apply to a padded,
cropped, spatially weighted operator.

Implement one matrix-free constrained reference solver using the shared forward
and adjoint. Compare at most one alternative initially, selected from measured
conditioning and projection cost. Establish any strong-convexity or duality
assumptions needed for an error certificate on this actual objective.

Start with independent small dense solutions, including active constraints and
multiple starts. Preserve the E1/E2a0 identical-objective control in its original
domain; add an appropriate independent control for the new operator. Diagnose
conditioning, scaling, projection accuracy and stopping separately.

Progress through a small full-resolution pilot, a development subset, then all
three development seeds, both regimes and both crops. Retain every required
selection fraction, photon budget and failed solve. Check feasibility,
stationarity, objective/error bounds, image accuracy and doubled-budget stability.
Publish the acceptance matrix before advancing to blind phase fitting.

Do not increase all budgets to compensate for an unexplained failure. Changes
to objective, scaling or tolerance require a new declared protocol and a new
comparison; previous incomplete results remain incomplete.

## P3 — Qualify noise, exposure and phase inference

Repeat scalar versus spatial variance studies on the corrected operator. Include
matched noiseless controls and realistic Poisson-plus-read noise. Compare a
frozen weighted quadratic with a justified signal-dependent likelihood where
needed, recording bias and uncertainty calibration. Simulation-derived variance
is an oracle diagnostic; real-data inference must estimate its inputs without
truth leakage. Do not change production weighting on current evidence.

Repeat phase-basis and finite-exposure ablations using the shared detector model.
Determine whether extra modes, exposure integration or spatial PSF variation are
supported by development evidence. Separate piston, flux, colour gain, global
shift and rotation ambiguities. Test measured shifts independently of known
pupil tilt, and retain uncertainty or an unresolved status where appropriate.

Verify nonzero-phase gradients, mode transitions, multiple starts and final
object/phase stationarity. Freeze architecture, priors, regularisation, phase
order and selection rules before any final assessment.
Any adopted likelihood or phase-model change must repeat the affected P1/P2
checks before the combined method can inherit their qualification.

## P4 — Requalify the science with a decision that can end the research loop

Run complete corrected development Gate-1 reconstruction/gap families, including
the practical-ranking and low-frequency augmentation/gap follow-up. Classify
opportunity only when denominators, metrics and numerical fits are valid.

Use seeds 2001–2012 as historical regression/replication data: their earlier
results have already been inspected. For a genuinely final test, reserve a
disjoint assessment family and confirm that it has not been used for tuning.
Keep chronological capture partitions and independent nights/cameras separate
where correlated frames would otherwise overstate independence.

Run scoped full-resolution Q2 only after the necessary numerical/model controls
pass. Preserve both initializations, model-selection and separate assessment
partitions, photon accounting, complete family coverage and invalid-case reasons.
Retain the existing 0.40 closure target where its original metric/opportunity
definition still applies. Any changed scientific question needs prospectively
declared criteria, not retrospective threshold adjustment.

Decision branches:

- **Valid benefit:** proceed to the corresponding Q3 and production claims.
- **Valid limited/no benefit:** publish the supported limits; retain the useful
  baseline and restrict the physical solver's claims to demonstrated conditions.
- **Incomplete/invalid:** identify the failed prerequisite and return to that
  phase. Do not claim either success or a physical information limit.

A scientifically sound negative result is a legitimate research outcome. It does
not imply that an unqualified MFBD feature is ready for release.

## P5 — Make qualified computations practical

Profile the corrected operator before selecting GPU work. Port dominant measured
costs behind a common CPU/CUDA interface, using the working project venv and
supported local RTX 5070 runtime. Verify forward/adjoint, gradient, objective,
solver-status and reconstruction parity before enabling acceleration by default.

Bound frame batches, scene/PSF workspaces, preview queues and checkpoints. Enforce
the declared resource scope: distinguish a CPU process ceiling and Torch
allocator cap from total GUI/process-tree RAM or all driver VRAM. Test exhaustion,
fallback, cancellation and resume without changing the objective or duplicating
frames. Keep total CPU parallelism within 32 threads.

Benchmark representative complete jobs, not gradient calls alone: wall time,
peak memory, time to first useful preview, cancellation latency and numerical
accuracy. Undertake Q3 sizes 1000/5000/20000 only when P4 authorizes that scope
and measured resource estimates support the runs.

## P6 — Integrate physical reconstruction with the existing application

Move qualified components from audit tools into an explicit engine API; remove
runtime monkeypatching from the production route. Preserve baseline selection
and expose physical inference with its applicable qualification status.

Connect calibration, mono/RGB/raw-CFA observations, quality selection, reference
epoch, field rotation, oblate-globe motion and Saturn globe/ring layers through
the same operator contracts. Include exposure motion, occlusion and moving-moon
masks as supported; test geometry and atmospheric errors separately before
combining them. Add local PSFs only when justified and separately validated.

Preserve the requested workflow: optional cached preprocessing, highest-quality
retained reference, SER timestamps, apparent flattening, nearest-neighbour Bayer
colour previews, tooltips and refreshed settings for every new run. Keep IR642
and L3 Mars as OSC. Improve rotation inference only with recoverability controls;
unresolved spin must remain explicit, with a manual override rather than a
plausible-looking invented rate. Apparent flattening must not be presented as a
uniquely measured intrinsic shape without the necessary geometry.

Provide actual intermediate images and stage progress, clear convergence/failure
status, owned-worker cleanup, compatible resume and scientific export metadata.
Test the complete open → preprocess → edit settings → run/cancel/resume → save
path on CPU and supported CUDA where applicable.

Sharpening remains a separate, explicit user-selected post-processing step.
Keep the unsharpened scientific result and its provenance available.

## P7 — Demonstrate useful results on independent captures

Begin sourcing and documenting captures now: additional nights/cameras, true
mono Mars, mono and OSC targets, field rotation, measurable surface rotation,
Saturn ring configurations, different seeing/noise and calibration conditions.
Record report/fixture permissions; do not distribute private pixels by default.

Use the supplied unsharpened Jupiter stack as an initial comparison, with matched
registration, scale, linear intensity and documented frame selection. Treat the
sharpened image only as a separate post-processing reference. An external stack
is a comparator, not ground truth; combine it with split-half consistency,
held-out raw residuals, coverage/colour checks and repeated independent captures.

Predeclare the acceptance protocol before tuning against the final corpus. The
product target is to match or improve the conventional unsharpened stack without
inventing detail, clipping channels or hiding poor coverage. Report failures and
supported limits by capture category, rather than relying on a single average.

Complete large-input and failure tests for the declared input formats. Retain
the existing SER and native AVI scope; compressed/OpenDML video support is a
separate optional extension unless made a release requirement.

## P8 — Complete release work within the requested scope

Refresh bundled CPU/CUDA artifacts after substantive engine changes. Verify
offline Linux execution without the development Python environment, scientific
encodings, GUI controls, cancellation/resume and supported GPU fallback.

Exercise and maintain the existing GitHub version-tag build matrix: Linux x64
CPU/CUDA, Windows x64 CPU, and macOS Intel/Apple Silicon CPU. Check versions,
source identity, bundled components, notices, SBOM, checksums and split CUDA
assets. Windows/macOS runtime testing stays excluded as requested; native build
success must not be described as runtime qualification.

Finish the quick-start guide, support matrix, known limitations, numerical status
explanations and reproducible release manifest. Obtain the owner's project
license decision before public distribution; use publisher signing/notarization
identities only if supplied, and label unsigned artifacts accurately. These
external choices need not block scientific implementation or local builds.

Prepare reviewable release candidates before the separate tagged-publication
step. This proposal does not itself create a tag or publish a release.

## First implementation sequence

1. Consolidate status and add per-stage durable experiment checkpoints.
2. Specify the extended-scene, detector and boundary contract, with independent
   failing fixtures for the measured mismatch.
3. Implement the shared forward/adjoint and qualify its flux, sampling and
   whole-sequence model error.
4. Add and certify the known-transfer constrained solver on that operator.
5. Repeat noise and phase/exposure sensitivity; freeze the scientific protocol.

Use separate commits for each reviewable implementation or completed evidence
step. Keep code fixes, failed experiments and subsequent refinements traceable.
Set runtime estimates after the P1/P2 pilot measurements; a calendar promise for
full qualification is not supported by the current evidence.

## Next implementation sequence after the full-count and endpoint work

1. Finish known-transfer conditioning qualification at full selected-frame counts.
   The frozen observed manifest contains all 60 development selections. The 5%
   endpoint passes with both reference and alternative solvers; complete all-frame
   numerical accuracy remains a separate requirement. Preserve every bounded
   failure and do not treat an optimizer flag as a certificate. The bounded
   diagnostic has now retained iteration/product traces and compared three
   diagonal scalings on the same frozen iterate. None tightened its bound, and
   the algebraic floor rules out fixing certification only with a more accurate
   inner solve. Safeguarded Newton-CG now passes dense optima, feasibility,
   descent, incorrect-active-set and CPU/CUDA controls; its declared endpoint
   experiment passes 25 frames but remains wall-limited at 500. A positive periodic
   inverse then improved the fixed-state full/reduced linear residuals, but its
   from-zero constrained experiment regressed at both 25 and 500 frames. It is not
   adopted. The early/late window diagnostic now passes all twelve independent
   probes; the window factor improves sampled Fourier energies and late reduced
   residuals, while projection can remove a substantial part of a unit correction.
   Neither fact proves a faster constrained fit. The completed ablation held
   the window inverse fixed and compared old Newton against the new gradient-
   projection/CG candidate at 25 frames and both original product caps. The latter
   passes 19 small independent CPU/CUDA controls and both caps at 556 products.
   Its full-count extension fails at both wall limits with bound 2.793714, worse
   than the earlier incomplete reference/diagonal bounds. Do not adopt it or
   treat identical cap outputs as convergence. See [the window/state decision](../results/p2-window-state-probe/DECISION.md),
   [the 25-frame ablation](../results/p2-projection-ablation/DECISION.md) and
   [the failed full-count extension](../results/p2-projection-full-count/DECISION.md).
   An approximate inverse must never replace the exact forward model or independent
   certificate. Inner and unaccepted-direction traces are
   now available to distinguish poor inner progress from outer budget exhaustion.
   Do not infer a solver winner or image accuracy from incomplete fits. Keep
   every alternative comparison, failed outcome and numerical threshold.
2. Use the measured resource evidence to choose subsequent execution budgets.
   Shared CUDA passes full 500-frame operator parity, while the small GPU cache
   churns at larger counts. A bounded parallel independent CPU verifier is now
   implemented to address expensive certificate checks. Its full-count arrays
   match serial CPU results bit-for-bit. The frozen-iterate studies have now
   adopted it under new identities while preserving historical states. Retain
   these worker and cache scopes in subsequent prospective execution budgets.
   The new 3 GiB fixed-admission cache passes all 500-frame parity and improves
   local per-product time by about 11%; adopt it only in a new study identity
   with the qualified 1 GiB headroom check. A larger LRU cache alone did not help.
   This modest gain does not justify an unqualified solver or blind budget rise. A new
   exact detector-crop FFT embedding now passes 36 CPU/CUDA controls, including
   independent fitted-scene certificates. Its bounds exclude circular aliases
   only from retained samples; they preserve the original nonperiodic model.
   Its [full-count profile passes](../results/p5-cropped-fft/DECISION.md), reducing
   local normal-product time from 1.052 to 0.391 seconds with 4 GiB retention
   (all spectra) and the declared 1 GiB additional headroom. The cropped 3 GiB
   mode also passes at 0.436 seconds. Adopt only in a new audit study identity.
   The [new comparison](../results/p2-cropped-comparison/DECISION.md) is complete:
   both methods pass 25 frames; diagonal Newton fails 500 frames at both actual
   product caps. Reference certifies 500 frames at iteration 1410 under the 1500
   cap. Its original 750/1500 stability remains failed. A separately declared
   [1500/3000 check](../results/p2-reference-stability/DECISION.md) now passes both
   fresh independent CPU bounds and image stability. Keep each historical failure.
   The failed projection candidate is not adopted.
3. Qualify the full 5/10/25/50/100% selection matrix across all seeds, seeing regimes
   and crops, with independent certificates and budget stability. The one-case
   endpoint comparison does not replace the complete family or settle the prior.
   Execute the [declared complete-family protocol](scene-selection-family-protocol.md)
   in manifest order, case indices 0 through 11. Each case runs all five fractions
   at fresh 1500/3000 caps with the qualified cropped FFT and independent CPU
   certificates. The runner gates execution on the archived stability check and
   binds source/input/runtime identities. Preserve every attempt, including failed
   fits and incomplete cases; require all 60 selections for a family pass.
   Case 0 passed all five selections; case 1 was interrupted after four passing
   selections (1/12 complete cases, 9/60 selections). Its 500-frame optimizer trace
   has no committed independent CPU certificate and is not a pass. Further matrix
   execution is deferred in favor of the application-image priority above. Use the
   [tested report checker](scene-selection-family-summary.md) to write an immutable
   cumulative summary. Continue with the next missing case in manifest order;
   do not rerun completed failures or extend their budgets implicitly.
4. Resolve prior/likelihood/phase and exposure sensitivity under P3, including
   identifiability and signal loss. Do not automatically extend the prior grid
   again or reuse the inspected pilot assessment as untouched final evaluation.
   Freeze a separate scientific assessment before Gate-1/Q2 requalification.
5. Continue P6–P8 within their dependencies: production integration and Linux
   bundle refresh, independent conventional-stack comparisons, then release
   preparation. Capture sourcing and owner license/signing choices remain external.

The initial full-count profile, observed selection manifest, endpoint reference
and alternative protocols are committed separately. The alternative supports
completed-stage resume only; the original solver retains exact iterate/momentum
resume. Overlapping timing measurements are not isolated speed benchmarks.
Each subsequent implementation and completed evidence step remains a separate
commit. Full convergence and scientific qualification are still required before
production adoption or Q3 claims.
