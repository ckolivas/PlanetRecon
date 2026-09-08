# Proposed development plan

Updated 2026-09-08, following commit `e3db720`. This proposes an updated sequence
for subsequent implementation; it does not mark the remaining work complete.
It supersedes the ordering of unfinished work in the historical roadmap;
completed features and archived scientific results remain intact.

## Execution progress (2026-09-08)

- **P0 foundations delivered:** current status matrix, exact-identity durable
  per-estimator checkpoints in the full Gate-1 runner, per-frame optical audits,
  and a read-only compatibility bridge for the 30 immutable archived inputs.
  Migration of other historical runners and within-solver restart remain.
- **P1 reference delivered:** extended optical scene, exposure translations,
  linear convolution, pixel sums, crop/mask and detector-coordinate CFA with exact
  adjoints. Independent spatial/derivative controls pass. All 3,000 full-grid
  development frames pass both per-frame and correlated-stack consistency checks.
  This resolves the measured crop-periodic mismatch; real optical adequacy,
  boundary-prior sensitivity and combined geometry/phase inference remain.
- **P2 reference delivered:** matrix-free nonnegative solver with a proved
  strong-convexity certificate and independent dense controls. A full 1152×1152
  three-frame scalar-weight pilot certifies both budgets. Spatial weighting
  exposed conditioning problems, retained as incomplete evidence; a same-objective
  diagonal-majorizer refinement is being qualified. Full selections/family remain.
- **P3–P8:** the dependency ordering and owner requirements below remain in force.
  A fixed-prior, one-case noise pilot is diagnostic; it does not freeze the
  likelihood, prior, phase model, boundary extent or scientific assessment.

See [current status](status.md), [operator evidence](../results/p1-scene-detector-full/DECISION.md)
and [initial solver pilot](../results/p2-scene-noise-v1/DECISION.md).

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
