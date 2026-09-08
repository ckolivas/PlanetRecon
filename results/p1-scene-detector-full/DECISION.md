# Extended scene-to-detector consistency decision

All 3,000 frames (three development seeds × two seeing regimes × 500 frames),
evaluated on both feature and bland crops, pass the prospectively declared
numerical consistency thresholds. The 10-frame-per-sequence pilot passed first.
Inputs, operator, runner and supporting tool identities were unchanged during
both runs. Total wall time was 205.19 s for six workers with two CPU threads each;
small independent solver tests ran concurrently, so this is not a controlled
scaling benchmark. No GPU or Q3 work was performed.

| Mean-square residual / detector-noise variance | Extended model range | Legacy crop-circular range |
|---|---:|---:|
| Per-frame mean | 2.75e-13–7.49e-13 | 41.85–2711.66 |
| Whole-stack mean | 2.99e-13–8.79e-13 | 10313.98–936634.50 |
| Whole-stack worst pixel | 4.65e-12–1.71e-11 | 1.66e6–5.67e7 |

Acceptance required per-frame mean ≤1e-8 and whole-stack worst pixel ≤1e-6.
These are numerical consistency checks including float32 serialization, not
physical reconstruction quality thresholds. The residual of the summed stack is
normalized by summed independent-noise variance; persistent bias therefore grows
with frame count instead of disappearing in a per-frame average.

Full optical exposure PSFs were regenerated from unchanged simulation
implementations and random-screen identities. The shared operator used the
stored extended optical scene, detector integration and true crop. Stored scene
truth was used only in this forward-consistency test. A reconstruction must solve
for all margins without access to that truth. No cropped PSF normalization or
interior trimming was used to suppress mismatch.

The result resolves the measured simulator-versus-crop-model inconsistency for
these known physical frames. It does not qualify the finite phase basis, motion
approximation, real-data optics, selected boundary prior, constrained solver on
the full family, blind inference, or Gate-1/Q2 opportunity. Those remain separate
P1/P2/P3/P4 controls. No production baseline or GUI solver was changed.

`protocol.json` records identities and thresholds; `report.json` and per-case
files preserve all 500 frame indices, cases, residual statistics and failures.
Ignored local `out/p1-scene-full/` retains atomic per-frame checkpoint payloads.
