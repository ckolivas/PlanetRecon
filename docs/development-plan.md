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

The [normalized local patch matcher](../results/real-data/jupiter-local-alignment.json)
is now available as an experimental option. It improves both paired image
metrics in 128/512-frame pilots and the full 3,341-screened-frame Jupiter run,
without extra rejection or sharpening. The full application gain is modest:
0.246% lower matched RMS and fine-detail correlation 0.96367 versus 0.96187.
Keep global alignment as the default. Preserve the rejected gradient and
single-frame-reference approaches; separating a blur direction alone did not
resolve their real-image regression. The adopted optional matcher excludes
one-dimensional texture and rejects collapsing/folding coordinate fields.
Thirty focused CPU/CUDA controls pass, alongside 1,039 default regression tests.

The [matched local midpoint comparison](../results/real-data/jupiter-local-midpoint.json)
is complete with identical source, template candidates and anchor: 1,645 frames
lower matched RMS to 0.00513695, but fine-detail correlation falls to 0.961536
versus 0.963670 for all 3,341 screened frames. Keep both controls optional and
the all-screened default. The [first local-energy weighting pilot](../results/real-data/jupiter-local-quality-pilot.json)
is also complete: eight-seed spatial-blur controls improve RMS by about 40%,
but the fixed 512-frame Jupiter comparison gains only 0.011% in RMS and loses
fine-detail correlation. It is not adopted. The [known-noise mono/Bayer controls](../results/real-data/local-quality-noise-controls.json)
now confirm the composite-filter noise budget within 0.4%. Subtracting known
noise improves spatial-blur control RMS by roughly 1–18% relative to the first
local weighting candidate. However, a same-colour finite-difference MAD estimator
mistakes sharp Bayer detail for noise, overestimating noise sigma by up to 2.5x.
The [bounded Jupiter noise sensitivity](../results/real-data/jupiter-local-noise-sensitivity.json)
is now complete: both half/full provisional noise subtraction worsen both paired
metrics; full subtraction increases RMS by 5.1% and loses 0.0533 fine-detail
correlation. Increasing quality smoothing also harms low-noise spatial-blur
controls. Neither approach is adopted; stop this weighting/smoothing sweep.
The [independent local-template refinement](../results/real-data/jupiter-local-template-refinement.json)
also completed: RMS improves 0.164%, but fine-detail correlation declines by
0.000141. It fails the declared two-metric gate and is not adopted; no full run
or additional template iterations are warranted.
The [colour-sampling controls](../results/real-data/sampling-footprint-controls.json)
are complete. An unconditional radius-0.75 triangular footprint reduces blur but
raises noise-only RMS about 34%, so it is rejected. A shared colour gate based on
agreement between two independent 32-frame templates retains radius-1 sampling
in flat/noise-only regions. Across mono and all four Bayer layouts it reduces
textured/mixed-scene error about 19–35%, preserves constant colours and direct
channel support, and leaves flat/noise-only controls unchanged. CPU/CUDA sampling
agrees within 1.4e-15. The [fixed Jupiter pilot](../results/real-data/jupiter-gated-sampling.json)
is complete: RMS worsens by 2.89% and fine-detail correlation falls by 0.0386.
The gate is not adopted; do not run a full-capture validation or sweep radii.
The [all-colour registration controls](../results/real-data/colour-registration-controls.json)
are complete: bilinear RGB luminance improves aggregate translation accuracy by
about 23% at Jupiter-like colour ratios and 31% with balanced colours. The weak
red/blue case is roughly neutral overall, with up to 1.8% regressions in individual
Bayer layouts, so this is not a universal replacement. Constant colours and
flat-template/noise controls pass. The [paired Jupiter luminance-proxy pilot](../results/real-data/jupiter-colour-registration-pilot.json)
is now complete with unchanged green-based stack weights, all 512 frames,
template candidates and best-frame origin. RMS improves 0.405% and fine-detail
correlation rises by 0.004536. The [full application validation](../results/real-data/jupiter-colour-registration-full.json)
is complete and the change is integrated into the existing experimental Bayer
local option. All 3,341 frames, cached weights, template candidates and anchor
1947 are preserved. Matched RMS improves a further 0.086% to 0.00529268 and
fine-detail correlation increases to 0.963964. The integrated output and coverage
match the full candidate bitwise. Global alignment remains unchanged; added
processing cost and the weak-red/blue limitation remain. Default regression:
1,046 passed, 77 skipped; focused CPU/CUDA and new resume controls: 37 passed.
The [matched scalar-weight comparison](../results/real-data/jupiter-scalar-weighting.json)
is complete: cached quality weighting beats equal weighting by about 1.5% in RMS
and improves fine-detail correlation. Both arms share every raw projection,
template and frame. Retain cached quality weighting. The [squared-quality controls](../results/real-data/squared-quality-controls.json)
improve variable-blur error by 6–7%, with a tiny uniform-scene noise penalty.
The [matched squared-quality pilot](../results/real-data/jupiter-squared-quality-pilot.json)
improves RMS by 1.2% and fine-detail correlation by 0.00149, retaining all 512
frames and sharing every raw projection. The linear arm reproduces the earlier
result bitwise. The [full 3,341-frame comparison](../results/real-data/jupiter-squared-quality-full.json)
also improves both metrics: RMS decreases 1.42% to 0.00521734 and detail
correlation rises to 0.964811. Original application sums reproduce bitwise;
all selected frames, template candidates and anchor 1947 are unchanged.
The optional **Stronger quality weighting (experimental)** control is now
integrated into local alignment, with GUI/CLI settings, unchanged cached
screening/template/anchor selection and exact resume checks. Default regression:
1,055 passed, 78 skipped; nine weighting controls pass with hardware enabled,
and the 37 existing local/colour CPU/CUDA controls pass. Linear checkpoint
identities remain compatible. The [mono Saturn pilot](../results/real-data/saturn-local-weighting-pilot.json)
is complete: 918 of a fixed 1,024-frame sample pass screening; all three arms
use the same 512 evaluation frames. Local alignment reduces RMS by 1.1%;
stronger weighting reduces it a further 16.9%, with improved fine-detail
correlation. Local arms share an independent 64-frame template and raw
projections. The stored 16-bit reference has usable detail despite its clipped
preview. Full preprocessing is complete: 25,476 of 27,689 frames are retained, with
reference frame 22,877. The [full paired protocol](../results/real-data/saturn-full-protocol.json)
is complete; its shadow sums match separate application runs bitwise on a small
control. The [full Saturn result](../results/real-data/saturn-local-weighting-full.json)
retains all 25,476 selected frames: local alignment reduces RMS by 0.82%, and
stronger weighting reduces it a further 20.23%, increasing detail correlation
from 0.990972 to 0.994502. All arms share the 24,087-pixel comparison interior.
Both optional improvements pass both metrics; keep current defaults and use
concrete residual evidence to guide any further bounded output improvement. Preserve linear-weight
defaults; do not sweep powers or introduce another rejection policy.
The Saturn setup follow-up now blocks an invalid equator-on prefill: a generic
globe-view assumption must not turn Saturn's rings edge-on. Discovery requires
a supplied signed latitude for Saturn surface-rate suggestions; cached and GUI
prefills exclude the old assumption and dependent rates. Switching modes clears
only matching automatic values and preserves user edits. Ordinary Saturn colour
stacking continues to use Motion model None without physical ring parameters.
Preserve global/all-screened defaults, ambiguity/fold guards and no sharpening.
Automatic field-angle estimation now refines the periodic correlation peak
between polar bins instead of rounding to 2.8125-degree increments. Independent
continuous-scene controls recover signed 0.5–4-degree rotations within 0.08
degrees. On a nine-frame sequence spanning one degree, unsharpened stack RMS
falls from 0.004612 to 0.000280 (known-rate control: 0.000269), retaining every
frame. Featureless discs still report unconstrained rotation. This removes a
measured quantization error; it does not establish real-capture motion accuracy.
Existing checkpoint identities bind the estimated angles and every frame pose,
preventing continuation when this refinement changes the fitted geometry.
Saturn runs with zero field/surface rates and no moon track now register camera
translation before classifying and accumulating globe/ring layers. Previously
all Saturn runs held the centre fixed, smearing even a static drifting scene.
Five-frame integer-drift controls recover the stationary image and both layer
coverage maps to numerical precision in mono and all four Bayer layouts, with
all frames retained. The unrestricted moving-layer trial failed existing region
isolation controls: visibility holes biased its generic correlation. It is not
adopted. A subsequent ring-only matcher now supports globe spin when field
rotation is zero and no moon track is supplied. It uses normalized correlation
over exposed rings and nearby sky, excluding the globe and smoothing footprint;
unobserved padding, low correlation and ambiguous peaks cannot set a drift.
Unconstrained frames keep the configured centre with a warning. Integer-drift
controls reproduce stationary rotating-scene images and both layer coverage maps
in mono and all four Bayer layouts. Noisy fractional shifts are recovered within
0.02 pixels; the five-frame noisy spin control reduces unsharpened RMS from
0.16194 to 0.03375 with every frame retained. Large globe-brightness changes,
repeating texture, shift limits and exact mono/Bayer resume are covered. At that
stage, field rotation and explicit detector moon tracks kept fixed centres. This does
not supply missing physical ring geometry or establish real-capture accuracy.
Default regression after the ring-only extension: 1,103 passed, 78 skipped.
The next field-rotation investigation identified an upstream bias: automatic
polar-angle fitting treated camera displacement as rotation. A seven-pixel
translation produced nearly eight degrees of false rotation in an independent
continuous-scene control. A simple translation-first trial failed the existing
fast-rotation and bright-spinning-Saturn controls and is not adopted. Instead,
four bounded angle/translation refinements align the sampled estimation images
to the reference centre before polar correlation or Saturn's ring-annulus masking.
Saturn translations use exposed rings; an unconstrained match retains the
configured centre. The existing faster-rotation and bright-spot controls pass.
Out-of-range translations cannot contribute to the rate; raw frames still use
one final combined warp. In the nine-frame drifting one-degree control, RMS
falls from 0.03635 to 0.000279, retaining all frames. A static drifting control
recovers the reference to rounding; mono/Bayer Saturn auto-field estimates match
the explicitly zero-rate image and coverage exactly. This corrects rate
estimation; it did not itself enable Saturn drift tracking with nonzero field rotation.
Default regression after the joint sample fit: 1,109 passed, 78 skipped.
The subsequent field-tracking extension rotates the exposed-ring reference into
each frame's orientation. Drift must improve its correlation beyond the measured
forward/back rotation interpolation mismatch. Asymmetric stationary controls
previously suggested false shifts up to 0.052 pixels; the gate retains zero.
Rotated unobserved pixels and the full smoothing footprint cannot contribute to
the template. Both rotation directions and near-zero rates recover stationary
stacks with all five frames in mono and every Bayer layout. The noisy asymmetric
ring/field/spin control reduces RMS from 0.19429 to 0.04382, retaining every frame
without sharpening. Existing exact layer-boundary controls, nonzero reference
anchors, shift limits and mono/Bayer field-motion resume pass. Moon tracks still
keep fixed centres; physical geometry and real-capture motion accuracy remain
separate requirements. Unchanged zero-field checkpoint identities are preserved;
older fixed-centre field-motion sums cannot be mixed with the new registration.
Default regression after field tracking: 1,132 passed, 78 skipped.
The globe-spin follow-up found the same kind of visibility bias outside Saturn:
zero-filled newly visible longitudes pulled whole-image camera registration by
up to 2.4 pixels in a stationary spinning-globe control. Surface/combined modes
with nonzero spin now use normalized matching on shared visible interior texture.
Both missing longitudes and the interpolated limb are excluded with their full
15-by-15 smoothing footprints. A first mask retaining the limb still biased
combined field/spin displacement by up to 0.18 pixels; it is not adopted.
Independent analytically sampled sphere controls cover both spin signs, field
rotation, integer/fractional drift, mono/RGB/all four Bayer layouts and noise.
The final maximum coordinate error is 0.071 pixels across the signed drift controls.
In the seven-frame noisy mono control, unsharpened RMS falls from 3.48881 to
0.21657; with combined field rotation it falls from 3.62800 to 0.22973. Both retain
all seven frames. Newly visible brightness changes cannot pull the shared surface;
nonzero anchors, shift limits and exact geometry resume pass. Unconstrained
surface matches retain the configured centre with a warning. Zero-spin and Saturn
tracking preserve their existing behavior; the normalized correlation is shared
without changing Saturn's masks or gates. Changed surface registration diagnostics
bind checkpoint identities, preventing reuse of older whole-image surface sums.
This demonstrates removal of a measured bias, not real-capture motion qualification
or automatic recovery of missing physical geometry. Private bounded logs and the
rejected nearby-sky candidate are retained in `out/globe-support/`.
Default regression after shared-surface tracking: 1,153 passed, 78 skipped;
85 focused globe/ring/tracking/resume controls pass.
These native green-proxy experiments do not change the separate area-mean
luminance quality score used in preprocessing. Do not assume noise caused the
real-image regression without evidence. Retain common colour weights, frame counts
and the best-frame anchor; do not add automatic sharpening or tune defaults from
one capture. Extend to local mono/other-planet captures where meaningful comparisons
are available. Keep experiments bounded and adopt only demonstrated output gains.
Do not expand reporting tools or the deferred broad numerical matrix.

Recorded SER exposure already supplies geometry midpoint timing where the
capture header includes it; manual values retain priority.

The next field-estimation fix preserves angular detail at different radii.
Previously the estimator averaged concentric rings before correlation, allowing
opposite radial contrasts to cancel. Seven independent continuous-scene controls
failed with the old estimator, including refusal of an observable one-degree
sequence. Corresponding rings now correlate separately after removal of their
individual mean brightness; their correlations are then combined. Signed
one/four-degree controls pass without weakening the texture threshold. In the
nine-frame positive one-degree control, the inferred rate is 0.00219350 rad/s
versus 0.00218166 known. Unsharpened interior RMS is 0.00005182 versus 0.00209330
with explicit zero correction and 0.00005015 with the known rate; all nine frames
remain. Existing drift, featureless-disc, Saturn and resume controls pass.
This is synthetic output evidence, not real-capture motion qualification or a
solution to spin/field ambiguity. Fitted diagnostics and poses continue to bind
checkpoint identity. Validation: 78 focused tests and 1,170 default regression
tests pass, with 78 skips; private logs are in `out/radial-field/`.

The cropped-field follow-up removes artificial rotation information from detector
padding. Previously even a uniform cropped frame supplied an apparently resolved
angle, and a nine-frame four-degree scene yielded only 0.00026791 rad/s versus
the known 0.00872665. Polar fitting now uses complete observed circles. In cropped
Field-mode runs, translation also uses shared observed interior texture, excluding
the rotated boundary and the sampled limb with their full smoothing footprints.
Restricting only the polar fit still left a direction-dependent drift bias;
retaining the interpolated limb left a 0.21-pixel fractional-shift error. Those
intermediate attempts are retained in `out/field-support/`. The final maximum
coordinate error is 0.0053 pixels in the six signed integer/fractional controls.
Positive four-degree mono stack RMS falls from 0.01750330 to 0.00034361, close to
the known-rate result 0.00034121. The negative-direction result is 0.00034714;
both retain all nine frames. Mono/RGB/all Bayer layouts improve against zero
correction and stay within 10% of their own known-rate error, retaining the
separate CFA interpolation floor. Flat unsupported estimates refuse to run;
explicit zero remains valid. Full footprint, nonzero-anchor and exact mono/Bayer
resume controls pass. This remains synthetic qualification, not a real-capture
motion accuracy claim. Updated registration diagnostics bind checkpoint identity.
Validation: 107 focused tests pass, followed by 1,200 default regression tests
with 78 skips (including the additional independent footprint control).

The angular-sampling follow-up fixes a separate false-rotation mechanism. A fixed
128-bin polar grid aliased resolved fine texture: an independent one-degree scene
was fitted as nearly six degrees. Angular sampling now scales with observed radius
to keep the outer sampling interval below one detector pixel. A denser grid alone
still selected weaker repeated peaks that happened to land on integer bins; the
periodic correlation is now interpolated eightfold before choosing its winner,
then refined locally. All 20 new controls failed with the old code; the grid-only
candidate passed 10, and the complete change passes all 20 across both rotation
directions and 128/256/512-pixel images. In the nine-frame 256-pixel control,
unsharpened RMS falls from 0.08079746 to 0.00644968, versus 0.00644813 with the
known rate. Every frame remains, and both rotation signs show the same gain.
This refines estimation only, with no raw-image resampling or sharpening added.
Synthetic evidence does not establish real-capture accuracy or remove genuinely
ambiguous motion. Private failures, the grid-only candidate and final metrics are
retained in `out/field-angular-sampling/`. Validation: 96 focused controls and
1,220 default regression tests pass, with 78 skips.

The shared-support drift follow-up fixes subpixel bias from tilted features.
Independent axis peak fits ignored the cross-axis curvature, producing errors
up to 0.43 pixels in analytically sampled tilted-Gaussian controls. Surface, ring
and cropped-field matchers now solve a joint two-dimensional quadratic peak.
It must have negative curvature in every direction and remain inside the fully
observed three-by-three neighbourhood; ambiguity and correlation guards remain.
Perfect integer matches retain exact coordinates. The maximum coordinate error
in the 15 signed tilt/offset controls is 0.042 pixels. In a nine-frame rotating
control, unsharpened RMS improves from 0.00499265 to 0.00419580 with all frames
retained; signed image controls also match the independently supplied camera
offsets. Fourteen of the original 19 new controls failed with the old axis fit.
The peak method now binds checkpoint identity, with an explicit old-checkpoint
refusal control; existing exact resume tests pass. These are synthetic motion
improvements, not real-capture qualification. Private logs are in
`out/coupled-drift/`. Validation: 94 existing motion/resume controls and 19 new
drift/image controls passed, followed by 1,240 default regression tests with 78
skips, including the additional checkpoint-identity control.

The drifting-crop follow-up corrects incomplete joint field/drift fits. Four fixed
refinement rounds left roughly 24–25% rotation-rate error in independent signed
nine-frame controls with fractional camera drift. Direct polar sampling alone
worsened those fits; retaining image recentering and adding rounds also left
substantial error. The adopted cropped-field path samples original estimation
planes about their fitted centres, restricts radial support to circles observed
in both frames, and requires successive joint updates below 0.001 detector pixels
(including rotation at the configured radius), with a 64-round cap. Unsettled
samples cannot authorize an automatic motion run. This does not certify physical
geometry or change manual rates, ordinary global stacking, Saturn or surface fits.
Signed mono RMS decreases from 0.00361815/0.00343317 to 0.00033951/0.00050679,
retaining all nine frames, versus known-rate RMS 0.00028461/0.00029263. These are
synthetic gains, not real-capture qualification. Mono/RGB and all four CFA layouts
recover rates within 4%; CFA outputs remain within 10% of the corresponding
known-rate colour reconstruction, accounting for its separate sampling floor.
Direct translated-support, nonconvergence refusal and old-checkpoint rejection
controls pass. Failed intermediate comparisons and test logs are preserved in
`out/translated-polar/`. All 12 drifting-crop image controls fail with the previous
implementation. The 23 new controls pass, followed by 1,263 default regression
tests with 78 skips.

The narrow-detail follow-up adds a retry for unresolved polar rotation fits.
The initial 24 radii can miss an observed roughly three-pixel-wide textured band,
incorrectly refusing automatic Field motion. The retry uses detector-scale radial
sampling with cubic interpolation of estimation pixels and preserves the original
limb exclusion, texture floor and correlation threshold. Initial resolved fits
are retained. Two broader replacements were rejected: dense bilinear sampling
left narrow-feature angle bias and failed colour image controls; dense cubic
sampling fixed those controls but regressed eight previously passing Bayer drift
cases. The retry has deliberately narrower scope and does not fix already-resolved
small-image angle biases observed in the full-replacement experiment.
New controls cover signed rotations on 384/576-pixel independent scenes, nine-frame
mono/RGB/RGGB stacks, circular radial brightness, independent noise and checkpoint
identity. All frames are retained in successful stacks, whose RMS is within 5%
of their corresponding known-rate reconstructions. These are synthetic controls,
not real-capture qualification. Both rejected implementations and their failures
are retained under `out/radial-sampling/`; do not adopt either as a blanket change.
All 14 new angle/stack controls fail with the previous estimator. The retry passes
158 focused controls and all 20 final new controls, followed by 1,283 default
regression tests with 78 skips.

The colour-motion follow-up addresses two linked failures. Green-only Bayer
tracking can miss usable red/blue detail, while a warped reference limb can
provide false green drift on a flat globe. Shared-surface support now requires
an observed reference pixel neighbourhood before warping, followed by the existing
source-side smoothing-footprint exclusion. An unresolved Bayer geometry/tracking
preflight retries RGB luminance; later unresolved green matches can retry colour
against the same reference. Existing peak/support gates still apply and failure
in every colour refuses the run. Raw CFA accumulation and green/cached scalar
weights are preserved. The policies bind checkpoint identity.
A blanket RGB replacement was rejected after eight Saturn Bayer drift regressions.
Colour retries without the reference-limb fix also failed image controls; they
accepted artificial green structure with large drift errors. The combined fix
passes signed red/blue globe motion on all four Bayer layouts, automatic field
rotation from red detail, late-frame colour tracking, flat-globe refusal, exact
resume and old-checkpoint rejection. Independent camera-shift image differences
remain below 0.15 at roughly 100-unit brightness. These are synthetic controls,
not real-capture qualification. Failed candidates and logs are retained in
`out/colour-motion/`. Work was isolated while a user GUI processing job was active.
The first full regression exposed two 32-pixel-wide legacy operator fixtures:
one has only 12 supported registration pixels (below the existing 32-pixel minimum),
and the other's competing peak is 0.98355 versus 0.99850, failing the existing
0.02 uniqueness margin. Their projection/coverage tests now supply their known
zero camera drift explicitly; application tracking keeps both refusal gates.
The failed regression is retained in `out/colour-motion/old-fixture-regression.log`.
Validation: 24 of 30 selected new controls fail with the previous implementation;
all 33 final new controls and 51 focused operator/colour checks pass. The final
default regression passes 1,316 tests with 78 skips.

The colour-preflight follow-up makes sampled and later tracking use the same
per-frame retry. Previously a green failure among the preflight samples switched
every frame to RGB, while the same failure later changed only that frame's match.
Preflight now receives each calibrated frame so it can retry colour locally;
unresolved geometry estimation still has its separate RGB retry. Resolved green
matches, quality weights, support/ambiguity checks and all selected frames are
preserved. The changed preflight policy binds checkpoint identity.
In two seven-frame independent drifting-globe controls, error versus supplied
camera shifts improves from 0.0260311 to 0.0227109 (surface) and 0.0315781 to
0.0281001 (combined motion), retaining all seven frames. These synthetic gains
are not real-capture qualification. Controls cover both preflight and later green
failures in all four Bayer layouts, exact resume and rejection of the old policy.
Evidence is retained in `out/colour-preflight/`. Eight sampled-frame cases fail
under the old policy; the new policy passes 91 focused checks and the complete
1,331-test default regression with 78 skips, including both old-policy and
reference-support checkpoint rejection.

The [global joint-peak comparison](../results/real-data/global-joint-peak-comparison.json)
is complete and is not adopted. A joint quadratic peak reduces maximum error
from 0.43004 to 0.04162 pixels in 15 independently sampled tilted-Gaussian
translations. However, its paired 512-frame real-capture comparisons improve
RMS by only 0.002085% on Jupiter and 0.006017% on mono Saturn, with fine-detail
correlation increases of 0.00002430 and 0.00000569. Both two-metric gates pass,
but these very small gains do not establish a useful default-application benefit.
Both arms retain all 512 selected frames, identical cached/frozen quality weights,
best selected anchors (1638 and 24683), comparison crops and the original
comparison-registration estimator. Source/sample identities were verified and
application code stayed frozen. The conventional references are not ground truth.
Keep the current default global fit; do not expand to full captures or tune this
candidate on these references. This does not undo the separately qualified joint
peak for masked motion matching. Further application work should target spatially
varying image residuals or a concrete new defect. Private candidate code, controls,
logs and paired output snapshots are retained in `out/global-joint-peak/`.

Local registration now excludes patches whose complete search footprint extends
outside the observed detector after global translation. Previously, zero padding
from the alignment pull could create a strong false local peak and deform an
otherwise correctly aligned cropped scene. Five independently evaluated continuous
translations produced up to 0.78845 pixels of false motion; the support check
reduces the maximum to 0.01678 pixels. Their worst observed-interior image RMS
falls from 1.31902 to 0.02977 (scene brightness about 300). A separate scene with
actual spatial deformation still improves by more than 60% over global alignment
inside the observed area. Unsupported patches retain global motion; the frame is
not rejected. This is a detector-boundary fix, not real-capture qualification or
a change to the default global matcher. Mono/Bayer checkpoints bind the new
support policy and refuse older mixed-policy sums. All 46 focused checks pass,
including CUDA parity, existing colour/local output controls and exact resume.
The full default regression passes: 1,339 passed, 79 skipped in 144.29 seconds.
The failing original controls and numerical probes are retained in
`out/local-observed-support/`.

## Motion execution requirement

Requested motion compensation must be performable. Surface/Combined/Saturn runs
now require a resolved surface rate; blank no longer means zero correction.
Automatic field fitting must contain a valid positive-time sample pair. Explicit
zero rates remain supported as a deliberate choice to disable a component.
Edge-on Saturn rings block the run. Unresolved shared-surface or exposed-ring
tracking stops processing instead of retaining a fixed centre. The already-read
sample frames are checked before reconstruction previews or checkpoint writes;
other frames are checked as processing reaches them. GUI Run lists missing
parameters before launching a worker, while Inspect and Preprocess remain usable.
Errors identify the missing estimate or failing frame and explain how to supply
geometry or select ordinary stacking. These requirements supersede the historical
fallback behavior recorded above; earlier measurements remain historical evidence.
Changed tracking policy is part of checkpoint geometry identity. Resume controls
now use resolved spinning-globe fixtures rather than relying on unobservable
nine-pixel globes silently retaining their centres.
Validation: 1,160 default regression tests passed, 78 skipped; 49 focused motion
refusal and globe/ring image controls passed.
A follow-up closes a Saturn estimation loophole: failed ring translation is now
marked unusable before fitting the automatic field rate. Angular texture alone
cannot certify a rate when its sampled camera drift is unresolved. All 90 focused
motion, angle, timestamp and Saturn controls pass.
The next bounded control exposed the opposite failure: a rejected middle sample
prevented valid endpoints from establishing a field rate. Rate fitting now pairs
consecutive usable samples at their actual measured times, excluding rejected
angles before unwrapping so they cannot change the valid branch. A nine-frame
analytic one-degree control with a 15-pixel middle displacement now retains its
eight within-limit frames instead of refusing an otherwise resolved field run.
Matched unsharpened RMS is 0.000274 versus 0.004612 with explicitly zero field
correction. All 92 focused rate/refusal/timestamp/Saturn controls pass. Failed
ring tracking remains unusable, and unresolved model tracking still stops a run.
Full regression after both rate-validity fixes: 1,163 passed, 78 skipped.

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
  which the original intake timing contract rejected. The timestamp-tie fix now
  retains their recorded equal times and first-to-last duration, excluding zero
  intervals from rate fitting without removing frames or inventing cadence.
  Reversed clocks and multi-frame clocks with no positive span remain invalid. Independent group
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

The user's direction prioritizes improved reconstruction from more frames.
Optional quality-range and frame-count comparisons are implemented; 100% remains
the default. Experimental normalized local patch alignment now improves the full
screened Jupiter image slightly and passes CPU/CUDA, colour, boundary and resume
controls. Follow the image-output priority above: matched all-frame/cutoff
comparisons, then controlled local quality weighting. Keep sharpening separate
and avoid adopting a change merely because a synthetic control improves.

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
