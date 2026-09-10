# Development with local alignment and the upper 50% quality range

Updated 2026-09-10 following the user's review of `interpolatedS.png`.
This is the current direction and supersedes earlier interpolation promotion
and next-step instructions in historical development logs and reports.

## Product scope

PlanetRecon is a stacking application. Further work must improve reconstruction,
registration, frame selection or stacking execution. Do not add image-editing or
post-processing workflows. Sharpening and other finishing belong in external tools.
Any future RGB alignment must be qualified as part of the stacking process, using
capture-derived information; reference-fitted post-stack shifts are not a solution.

## Working application

Use local patch alignment, Motion None, original linear Emil quality weights,
and scores strictly above `(capture best + capture worst) / 2`, followed by cached
screening. This is a quality-range cutoff, not half the frame count. Preserve the
best selected frame as the reference origin, normal template selection, original
CFA measurements, direct colour support and existing local ambiguity/fold guards.
No automatic sharpening. The stronger quality weighting experiment was removed after visual review.

## Completed: coupled local registration peak correction

Local patches previously fitted separate horizontal and vertical parabolas to
their correlation peak. For diagonal features those slices do not locate the
joint centre. The matcher now includes cross-axis curvature, requires a maximum
constrained in both directions, and refuses offsets outside the observed 3x3
neighbourhood. Exact matches, texture/support checks and the fold guard remain.
This changes alignment only; ordinary CFA sampling and frame weights are retained.
The peak method is included in result provenance and checkpoint identity, so old
local-stack sums cannot be resumed under the changed alignment.

Four independently specified tilted quadratic peaks now recover their known
centres. In two continuous diagonal-pattern controls, displacement RMS falls from
0.630/0.406 pixels to 0.045/0.032 pixels, and aligned image RMS also improves.
Weak diagonal ridges retain global motion. All 66 focused CPU and 14 CUDA checks
pass, including old-checkpoint refusal, exact resume, colour and boundary controls.

A fixed 128-frame Jupiter pilot improved both declared reference metrics, followed
by one full application run using all 1,645 upper-half selected frames. It reused
the existing ordinary output for comparison; the reference template is bitwise
identical and anchor 1947 is retained. Full planet-interior relative RMS difference
falls from 0.005293906 to 0.005275487 (0.35%); sigma-three highpass correlation rises
from 0.950682 to 0.950818. All channels are finite and valid, and the raw float32
TIFF matches the scientific snapshot. The full CUDA run with checkpoints/export
took 245 seconds. The paired pilot took about eight seconds per arm for alignment
and projection; these timings are illustrative, not isolated benchmarks.

The user found the difference almost indistinguishable after sharpening. Retain
this as a small alignment correction, not a meaningful quality gain. Full previews show
no gross new colour or limb defect. The retained comparison uses `out/local-joint-peak/full-joint.tif`
against `out/cfa-local-full/ordinary.tif`; no further run is warranted.
Matched display previews are `out/local-joint-peak/full-joint.png` and
`full-prior.png`. The committed evidence is `results/real-data/local-joint-peak.json`;
private scripts, prior source, outputs and logs are under `out/local-joint-peak`.

## Rejected: local colour interpolation

The full Jupiter comparison processed all 1,645 selected frames from 3,749 captured
frames. Interpolation reduced planet-interior relative RMS difference against the
unsharpened conventional reference by 2.22%, and improved the three regional metrics.
However, the user's sharpened `interpolatedS.png` exposes repeating curved texture
artifacts absent from regular stacking. The user judged it a poorer result.
Those numerical checks missed a defect that matters in normal post-processing:
this is a rejected candidate, not an image-quality win. Its 148.8-minute full-run
cost (including checkpoints and exports) is also unjustified by the outcome.

The experimental GUI checkbox, CLI option, configuration field, processing and
checkpoint integration, CPU/CUDA interpolation modules, and candidate-only tools
and tests have been removed. The independent cancellable checkpoint writer remains.
Ordinary jobs saved with interpolation disabled still load with their settings;
jobs that enabled the rejected option are refused instead of silently changed.
Restart an already open GUI to use the updated application.

Removal checks passed: 141 CPU/GUI tests for stacking, RGB completion, exports,
settings and checkpoint/resume behavior, plus all 13 targeted CUDA tests.

Private captures, comparison TIFFs/PNGs, snapshots and checkpoints are retained
under `out/cfa-local-full`; the user's sharpened image remains at the repository
root. The measured report in `results/real-data/cfa-local-full.json` now records
rejection and the review-image digest. Earlier reports remain historical evidence,
not acceptance claims. The removed experiment and its reproduction scripts remain
available in Git at commit `bd2f29c`; do not rerun or revive it from these old plans.

## Further work

No more interpolation tuning, full-capture repeats or default promotion are planned
for this candidate. Future output changes must beat the ordinary local stack in
practical visual assessment, including user-selected sharpening that reveals
structured artifacts, as well as numerical comparisons. Interior reference-error
metrics alone are insufficient; inspect the limb, colour and fine texture too.
Keep sharpening in external, user-selected post-processing tools.

The periodic continuation automation remains paused. This document does not
schedule background work.

## Rejected: reduced local confidence suppression

A sparse-patch continuous-scene control improved when the second confidence
attenuation was removed, but the fixed 128-frame Jupiter comparison worsened
relative RMS from 0.007375950 to 0.007435980 (0.81%). Detail correlation rose
slightly, so this fails the declared material two-metric gate. Production code
remains unchanged; do not expand or tune this candidate. The evidence is
`results/real-data/local-confidence.json`, with private files in
`out/local-confidence`.

## Removed: post-stack RGB alignment

The manual RGB dialog, saved-result editing entry point, CLI command and processing
utility were removed after the user clarified the stacking-only product scope.
The independent Fit-view resize correction remains for the stacking preview.
Private trial images under `out/rgb-alignment` remain available as historical
evidence; their reference-guided offsets are not accepted stacking parameters.
Do not extend this post-processing workflow or infer automatic stacking offsets
from the conventional reference.
Removal validation passed 124 stacking, GUI, preview and export tests, with 16
opt-in tests skipped. Existing captures and private comparison outputs are retained.

## Rejected: smaller fixed local patches

A bounded comparison tested 33-pixel patches on a 16-pixel grid against the
current 65-pixel patches on a 32-pixel grid. Frame selection, the full-selection
template, reference origin, original linear weights and raw CFA projection were
identical. The declared gate required at least 1% lower reference RMS and higher
fine-detail correlation before considering a full run.

An analytic spatially varying deformation control reduced displacement RMS from
0.471 to 0.234 pixels, but a stationary noisy control increased spurious motion
from 0.0065 to 0.0135 pixels. On the fixed 128-frame Jupiter pilot, relative RMS
fell only 0.56% (0.007375950 to 0.007334826), while sigma-three highpass correlation
fell from 0.693732 to 0.692914. This fails the gate. Matched unsharpened previews
show no obvious gross new limb or colour defect, but that does not override the
failed detail comparison or establish acceptance after external sharpening.

Keep the existing patch sizes for now. This rejects a blanket switch to 33 pixels,
not capture-dependent or spatially adaptive sizing: appropriate alignment support
depends on sampling, resolved texture and noise. No production change, full-capture
run or fixed-size sweep is warranted by this candidate. The report is
`results/real-data/local-small-patches.json`; private scripts, controls and paired
TIFF/PNG/snapshot outputs are retained under `out/local-small-patches`.

## Adaptive sizing: prototype evaluated, production selection still open

Following the user's sampling correction, two automatic sizing rules were tested.
A reference-only rule based on the weakest gradient direction selected larger
patches as sampling increased, but worsened the twice-sampled deformation control
from 0.218 to 0.405 pixels in original-sampling units. Feature scale alone does not
balance measurement reliability against averaging over spatially varying motion.

A second prototype tries 33, 65 and 97 pixels at common centres, choosing the
smallest patch with supported two-directional texture and an accepted correlation
peak in each frame. It retains the joint peak, observed-support and fold guards,
and projects original CFA measurements once. It uses a 16-pixel grid, so the trial
compares the complete adaptive strategy, not only the size-selection rule.

The second rule improves the analytic deformation control at half, native and
double sampling (errors 0.548/0.471/0.218 to 0.473/0.234/0.068 pixels in original
sampling units). Exact/brightness/unrelated-noise controls produce no motion;
stationary noisy data show 0.0152-pixel RMS spurious motion. Five CPU/CUDA checks,
including nonzero deformation, agree within 1.4e-15 pixels.

The same 128-frame Jupiter pilot uses all three sizes (17,026 / 14,530 / 10,691
accepted patch matches). Fine-detail correlation rises from 0.693732 to 0.696663,
but relative RMS worsens 2.73%, from 0.007375950 to 0.007577257. It fails the
declared two-metric gate. No production change or full run is justified. The
prototype is also unoptimized: it evaluates every size even after finding a
reliable smaller match. Evidence and limitations are recorded in
`results/real-data/local-adaptive-patches.json`, with private code and outputs in
`out/local-adaptive-patches`.

Automatic sizing remains an open direction. Before another capture trial, improve
the actual criterion for trusting a displacement across sizes; texture extent and
the current binary peak guards alone are insufficient. Qualification must include
sampling changes and noise, not a single globally optimal pixel size inferred from
Jupiter. Keep the existing working default until a candidate improves practical
stack quality; do not expose this failed prototype as an automatic mode.

## Patch reliability: uncertainty alone does not choose the right size

A residual-based uncertainty diagnostic was tested before any further capture
run. It fits brightness/offset and translation-gradient terms, groups nearby
residuals into 8-pixel blocks, and estimates uncertainty from those block scores.
The estimate is approximate and is not a calibrated displacement confidence
interval; model mismatch and interpolation bias remain limitations.

Choosing the smallest patch with estimated error below 0.10 pixels fails 36
known-translation controls spanning three sampling scales and three noise levels:
displacement RMS rises from 0.110 to 0.121 pixels. Passing a precision threshold
does not establish that a smaller patch is needed.

A distinct rule was then frozen and tested on 72 fresh-seed translation and
spatial-deformation controls. It chooses the largest patch consistent with the
smaller estimates within three combined standard errors plus fitted corrections.
Aggregate translation RMS improves from 0.194 to 0.080 pixels and deformation RMS
from 0.275 to 0.235 pixels, but deformation errors at half/native sampling worsen
12.0%/16.4%. This fails the per-scale 5% regression limit despite aggregate gains.
Neither rule is adopted, and no Jupiter pilot or confidence-threshold sweep follows.

The unresolved distinction is random measurement noise versus real variation of
motion within a patch: the latter can inflate the residual-based uncertainty and
make an oversized patch appear compatible. Future adaptive selection must measure
that model bias independently before another capture trial. Keep the working
application unchanged. The summary is `results/real-data/local-patch-reliability.json`;
complete controls, code and individual measurements remain in
`out/local-patch-reliability` with their digests in the summary.

## Withheld-pixel validation: pause automatic selectors

Splitting raw detector pixels before smoothing lets a candidate's fitted shift
be evaluated on detail it did not fit. A 72-case trial improves aggregate
deformation error by 32.8%, but doubles translation error at half sampling
(0.0252 to 0.0513 pixels), failing the per-scale regression gate. A guarded
version on 72 fresh cases requires significant improvement in both splits and
fits photometric terms on training pixels only. It preserves translation results
but changes size in only one deformation case, giving 1.4% aggregate improvement.

A separately declared capture-calibration trial pools 16 frames per condition
and assesses the frozen selection rule on eight separate frames. Across 36
conditions it preserves translation results, but chooses a different size in
only two deformation conditions and improves aggregate deformation error by
1.2%, below the declared 10% gate. No Jupiter runs follow these failed gates.
These are analytic mono controls; they do not qualify Bayer or final-stack output.
Reports and complete private-result digests are in
`results/real-data/local-patch-validation.json`.

Pause this automatic-sizing family rather than adding more confidence thresholds
or larger calibration budgets. The useful application step is explicit manual
patch sizing, preserving the existing 65-pixel default. This supports the user's
capture-dependent workflow without claiming an automatic optimum.

## Available: manual patch size for each capture

The Capture tab now exposes **Alignment patch size (odd pixels)**, with CLI
`--local-patch-size`. Accept odd widths from 15 to 255 pixels; retain 65 as the
default and set grid spacing to half the selected width rounded down. New runs
read the current control value and reuse the existing preprocessing cache.
Saved jobs retain the choice; old jobs default to 65. Custom sizes are bound to
checkpoint identity. Unchanged grids preserve previous checkpoint identities;
the corrected single-row/column grids described below require a fresh stack.
An undersized frame now stops with the selected size, required detector dimensions
and a fitting-size suggestion. Fewer than four selected frames also stops the run.
The user can explicitly choose global alignment in either case.

This is an explicit user choice, not an automatic recommendation or a claim that
a particular size improves every capture. The rejected selectors remain outside
the application. Validation passed 135 focused CPU/GUI/CUDA checks, including 14
hardware checks: changed-size output, cache reuse, exact resume, mismatch refusal,
all-control fresh-run snapshots and custom-size CPU/CUDA parity. Two unrelated
opt-in checks in the additional regression group remained skipped.

## Completed: honour the requested alignment method

The initial manual-size implementation inherited a global-alignment fallback when
the detector could not contain a complete patch search, or fewer than four frames
remained for the template. Those runs now refuse to start local stacking and give
actionable guidance instead of producing a stack with a different method. Patch
failures caused by ambiguous texture still use the existing per-patch global
estimate; the change concerns whether the requested local method can run at all.

Checks confirm no processing frames are read and existing checkpoints remain
untouched for an oversized patch. Selected-frame counts are checked after cache
validation but before selecting the stacking backend. Explicit global alignment,
exactly four frames and exactly fitting detector dimensions remain supported.
Validation passed 133 focused CPU/GUI/CUDA tests, including 14 hardware checks;
two unrelated opt-in GUI checks remained skipped. Valid local stacks and their
checkpoint identities are unchanged.

## Rejected: higher-order registration-proxy resampling

A bounded test changed only the globally shifted registration proxy from bilinear
to cubic sampling; final image/CFA sampling was unchanged. Across three scales
and four fractional global shifts, displacement RMS improved only 0.12%, below
the declared 10% development gate. Exact, brightness, flat and unrelated-noise
controls passed, but the negligible gain does not justify a capture trial or
production change. Evidence is `results/real-data/local-proxy-resampling.json`.

## Completed: centre single-row and single-column patch grids

When only one row or column of patches fits, it previously stayed at the first
valid top/left position instead of the detector centre. Its confidence taper then
suppressed correction around the planet centre asymmetrically. Such axes now use
the nearest lower integer centre; axes with multiple patches are unchanged. CUDA
tiles use the same offset origin as CPU patches and retain complete observed
search footprints.

In a 97-by-97 continuous-scene control, the sole patch moves from (35,35) to
(48,48). At the image centre the recovered displacement changes from (0.093,0.062)
to (0.751,0.499) pixels for known motion (0.75,0.50). Central image RMS falls from
5.860 to 1.981 with the same ordinary bilinear image projection. This establishes
a small-detector placement correction, not a real-Jupiter quality gain. The
424-by-656 Jupiter-shaped control retains identical grids and CPU displacements;
no full capture rerun is needed.

All 104 focused checks pass, including 17 CUDA checks, reflection symmetry,
minimum support, output/resume behavior and rejection of pre-correction checkpoint
grids. Only captures whose singleton grid changes receive a new checkpoint-policy
tag; their old accumulators must not be mixed with the corrected alignment. The
evidence is `results/real-data/local-singleton-grid.json`.

## Rejected: automatic retry of local search-boundary peaks

A candidate retried correlated peaks at the three-pixel search boundary with a
five-pixel search, using the same patch centres and existing texture, peak and
fold guards. Expanded candidates required complete support inside both the aligned
proxy and the original detector. Known signed 3.8-pixel shifts improve from
3.821-pixel displacement error to 0.037 pixels; ordinary small shifts, exact,
brightness, blur and unrelated-noise controls are unchanged. Shifts beyond the
expanded boundary remain rejected. Four signed detector-edge checks match between
CPU and CUDA within 3.7e-15 pixels.

The fixed 128-frame Jupiter trial nevertheless worsens relative RMS from
0.007375950 to 0.007446203 (0.95%), while highpass correlation rises slightly from
0.693732 to 0.696181. This fails the declared requirement for at least 1% lower RMS
and higher detail correlation. Keep the existing search radius; no full run,
automatic expansion option or threshold sweep is warranted. Original colour
sampling was unchanged throughout. Evidence is in
`results/real-data/local-search-retry.json`, with private code and outputs under
`out/local-search-retry`.

## Completed: preserve observed brightness at local-template borders

Re-centring the averaged registration template previously attenuated partially
covered border pixels and filled uncovered pixels with black. It also shifted
the best-frame fallback from its original coordinates. The template now divides
by its resampled observed support and fills unobserved locations from the best
frame in its original coordinates. This prevents detector padding becoming a
registration feature. Final image sampling, quality selection and weights are
unchanged. Local checkpoint identities include this template policy; restart
older local checkpoints rather than mixing their accumulated sums with new runs.

All 109 focused CPU/CUDA checks pass, including signed fractional/integer border
coverage, best-frame fallback, exact resume and old-policy refusal. Controlled
four-pixel template origins previously produced false local displacement peaks
of 0.0010 to 0.1719 pixels on identical observed texture; the corrected template
produces zero. These controls isolate re-anchoring from correlation estimation.
The real Jupiter template retains anchor 1947 and the same 1,645 selected frames.
Its 102,530 planet pixels (above 15% of reference peak) are bitwise unchanged;
only 1,078 detector-border pixels change, by at most 0.000199 ADU. This is a border
correctness fix, not a demonstrated Jupiter quality gain; no full stack rerun.

## Rejected: skip ineligible CUDA patch correlations

A candidate applied the existing texture and detector-support checks before
CUDA correlation, gathering only usable tiles. The Jupiter reference contains
228 patches, of which 103 pass the texture checks. No matching thresholds,
frame choices, template values or final image sampling changed.

The candidate passed 115 focused CPU/CUDA tests. In three alternating-order
paired trials on the same 32 selected Jupiter frames, median local-alignment plus
raw-CFA projection time fell from 1.7444 to 1.5952 seconds (8.5%). Displacements
agreed within 1.3e-13 pixels and normalized RGB stacks within 3.6e-13 ADU. This
missed the declared 10% execution-time reduction gate, and those timings exclude
capture loading, global registration and export. Remove the candidate from the
application; do not repeat or expand the benchmark. Evidence is in
`results/real-data/local-sparse-patches.json`; private code and measurements remain
under `out/local-sparse-patches`.

## Completed: flat-field correction independent of illumination units

Calibration previously clamped flat values below `1e-6` before dividing the
observations, then multiplied by the original flat median. Saving the same flat
on a smaller numeric scale could therefore darken the stack and leave spatial
sensitivity variations uncorrected. Form the flat's relative sensitivity first
and multiply observations by its median-to-pixel ratio. This removes the absolute
floor and avoids large intermediate values caused solely by the flat's units.
Explicit calibration objects now reject nonfinite, nonpositive or mismatched
flats just as file-loaded calibration tables do.

For a known 100-ADU field with sensitivity varying from 0.5 to 1.5, the previous
correction produced 0.5 to 1.5 ADU when the flat was scaled by `1e-8`; the corrected
field stays at 100 ADU within 3e-14. Checks cover scale factors from `1e-200` to
`1e200`, bias/dark/gain order, negative calibrated signals and input preservation.
Local mono/Bayer/RGB stacks using the upper 50% quality range give identical
screening measurements, frame choices, images and coverage for flats differing
by a power-of-two scale, on both CPU and CUDA.

All 115 focused calibration, GUI, local-stack and resume checks pass; one opt-in
test remains skipped. The flat-normalisation policy is included in calibration
provenance, preprocessing identities and checkpoint identities. Existing jobs
using flats require fresh preprocessing and a new stack; current checkpoints
resume exactly. Jobs without flats retain their existing calibration path and
identities. No Jupiter quality claim or full capture rerun: its normal workflow
does not use a flat.
