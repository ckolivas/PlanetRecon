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
No automatic sharpening. Stronger quality weighting remains optional.

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
