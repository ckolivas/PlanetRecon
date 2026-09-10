# Development with local alignment and the upper 50% quality range

Updated 2026-09-10 following the user's review of `interpolatedS.png`.
This is the current direction and supersedes earlier interpolation promotion
and next-step instructions in historical development logs and reports.

## Working application

Use local patch alignment, Motion None, original linear Emil quality weights,
and scores strictly above `(capture best + capture worst) / 2`, followed by cached
screening. This is a quality-range cutoff, not half the frame count. Preserve the
best selected frame as the reference origin, normal template selection, original
CFA measurements, direct colour support and existing local ambiguity/fold guards.
No automatic sharpening. Stronger quality weighting remains optional.

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
Keep sharpening an explicitly user-selected post-processing step.

The periodic continuation automation remains paused. This document does not
schedule background work.
