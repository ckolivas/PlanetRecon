# Current implementation and qualification status

Updated 2026-09-09. This matrix supersedes current-status wording in the historical
R10 roadmap; archived experiments and their original decisions remain unchanged.
Latest complete regression: 913 passed, 64 skipped. The new projection solver
and cropped-FFT controls also pass with CUDA enabled (19 and 40 tests respectively).

| Area | Implemented | Qualification / remaining work |
|---|---|---|
| Capture baseline | SER/native AVI, mono/RGB/raw CFA output, translation, nearest-neighbour colour previews | Independent nights/cameras and matched conventional-stack comparisons remain |
| Preprocessing | Separate optional cache, quality/shape exclusions, best retained reference, SER duration, apparent flattening, geometry hints | Spin may be unresolved; apparent flattening alone is not intrinsic shape |
| Geometry | Field and surface motion, Saturn globe/rings, coverage, CPU resume | Physical inference and combined atmospheric/geometry accuracy remain experimental |
| GUI/export | Updated per-run controls, tooltips, cancel/resume, PNG16/TIFF16/float32 and provenance | Full independent capture workflow and refreshed bundle acceptance remain |
| Compute/resources | Local RTX 5070 venv support, CPU/CUDA translation, scoped Linux RAM/Torch allocator limits, recovery | Shared CPU/CUDA operators pass all 500-frame parity; bounded parallel independent CPU verification is qualified; full reconstruction performance and production integration remain |
| Scientific inputs | 30 certified full-resolution files under recorded generation identity | Input certification does not qualify a changed reconstruction method |
| Scientific reconstruction | Legacy estimators, constrained Newton-CG, coupled-inverse and projection-phase experiments | Extended model matches all 3,000 development frames; the 12-case, 11-frame pilot passes; reference, L-BFGS-B and diagonal Newton-CG pass 25 frames; projection-phase/window solver also passes 25 frames but fails 500; one 500-frame reference endpoint now passes independent accuracy and cap stability; the full 60-selection family remains incomplete |
| Gate-1/Q2/Q3 | Historical reports preserved | Requalification required; Q3 is not authorized |
| Experiment execution | Atomic per-estimator/subset/crop/budget checkpoints in full Gate-1 audit; exact identity resume, failure records and stage-boundary wall budget | Reference sensitivity/family/endpoint runners retain exact iterate/momentum resume and incomplete outcomes; alternative solver has completed-stage resume only |
| Releases | Five native GitHub tag-build targets, Linux local CPU/CUDA packaging | Refresh artifacts; owner license/signing decisions for publication; Windows/macOS runtime tests excluded |
| Capture interpretation | IR642 Mars and L3 Mars are both OSC RGGB | True mono Mars and further independent captures still need sourcing and permission records |

New operator evidence: [full-scene consistency decision](../results/p1-scene-detector-full/DECISION.md).
Reference solver evidence: [qualified numerical/noise pilot](../results/p2-scene-noise-v2-qualified/DECISION.md).
New numerical family: [12-case pilot decision](../results/p2-development-numerical-pilot/DECISION.md).
Backend measurements: [shared CPU/CUDA transforms](../results/p5-shared-fft/DECISION.md),
[full-frame profile](../results/p5-full-count-profile/DECISION.md) and
[bounded independent CPU verification](../results/p5-parallel-reference/DECISION.md).
Full selections: [observed manifest](../results/p2-full-selection-manifest/DECISION.md) and
[reference endpoints](../results/p2-selection-endpoints-reference/DECISION.md) and
[optimizer comparison](../results/p2-selection-endpoints-comparison/DECISION.md).
Conditioning: [fixed-iterate bounds](../results/p2-frozen-conditioning/DECISION.md)
and [three-scaling comparison](../results/p2-frozen-jacobi/DECISION.md). All three
32-product probes pass independent CPU/CUDA checks but leave the distance bound
at 0.149922. Better inner correction alone cannot certify this saved iterate with
the tested bound family. Constrained solver progress remains necessary; none of
these diagnostics updates a scene or qualifies the incomplete 500-frame fits.
The subsequent [Newton-CG endpoint study](../results/p2-selection-endpoints-newton/DECISION.md)
does update feasible scenes and passes 25 frames, but both 500-frame fits stop at
their wall limits with bound 0.167701. Its successful descent and identical cap
outputs do not establish convergence. A [retained-spectrum cache](../results/p5-retained-cache/DECISION.md)
passes full-count parity and reduces local product time about 11%; production
defaults remain unchanged. The subsequent [periodic-inverse probes](../results/p2-periodic-probe/DECISION.md)
improve fixed-state linear residuals but the [constrained fit regresses](../results/p2-selection-endpoints-periodic-newton/DECISION.md)
at both 25 and 500 frames. The [window/state diagnostic](../results/p2-window-state-probe/DECISION.md) now passes
all twelve independent probes. Window weighting improves sampled Fourier energies
and late-state reduced residuals, but unit-step projection remains material and
no fitted scene is qualified. A separately implemented gradient-projection/CG
candidate passes 19 CPU/CUDA controls and [both 25-frame caps](../results/p2-projection-ablation/DECISION.md)
at 556 products. Its [500-frame extension fails](../results/p2-projection-full-count/DECISION.md)
at both wall limits with bound 2.793714; identical outputs do not establish
convergence. A new exact detector-crop FFT embedding passes 40 CPU/CUDA controls,
including independent fitted-scene certificates; [full-count operator/resource qualification](../results/p5-cropped-fft/DECISION.md)
now passes. Its 4 GiB retained spectra reduce local normal-product time from
1.052 to 0.391 seconds (about 2.69 times faster) while preserving independent
operator parity. The [cropped-backend solver comparison](../results/p2-cropped-comparison/DECISION.md)
now retains both completed studies: diagonal Newton reaches its full product caps
but fails 500-frame accuracy and stability. The reference first meets the full-count
independent bound at iteration 1410 under the 1500 cap, while its original 750-cap
fit remains incomplete. A separately declared [1500/3000 stability check](../results/p2-reference-stability/DECISION.md)
now passes: the frozen lower scene and fresh zero-start upper solve both have
independent bound 9.859417e-6 and identical latent/detector images. This qualifies
one endpoint, not the complete numerical family or scientific quality. The
[gated full-family protocol](scene-selection-family-protocol.md) is ready for
all 12 cases and 60 observed selections, one case at a time in manifest order.
Case 0 (seed 1001, Dr0=4, feature crop) is now running in
`out/p2-selection-family/case-00`; check its process and final report before
starting another case. Its first two selections pass, but the case has not yet
completed. The [family report checker](scene-selection-family-summary.md) now
revalidates all 60 selections and shared identities, with 33 additional tests
passing. It retains missing/failed cases and refuses to overwrite earlier summaries.
Production defaults and numerical criteria remain unchanged.

Independent-data preparation: [seven-capture intake](../results/real-data/README.md)
records whole-file hashes and unknown permissions without publishing pixels or
private paths. Three mono Saturn files are available. OSC Saturn and both OSC
Mars files contain duplicate timestamps; valid motion timing remains unresolved
for those captures. Camera/night independence and true mono Mars are still missing.
All 64 sampled equal-timestamp pairs per affected capture had different raw
pixels; those bounded checks neither recover cadence nor justify frame deletion.
Scientific setting limits: [unbracketed prior](../results/p2-prior-extension/DECISION.md),
[domain](../results/p2-domain-sensitivity/DECISION.md),
[sampling](../results/p2-sampling-sensitivity/DECISION.md),
[photon normalization](../results/p2-photons-sensitivity/DECISION.md).
Legacy solver evidence: [full-grid decision](../results/r10-full-grid-summary/DECISION.md).
Remaining sequence: [development plan](development-plan.md).

## Resuming a full Gate-1 audit

Use `tools/audit_full_gate1.py --inputs ... --out ... --wall-budget-s 3600`
and record `--background-workload`. Repeat the identical command with `--resume`
to reuse completed numerical stages. Input hashes, package/tool source, solver,
regularisation, budgets, protocol and crop identities must match exactly.
Use one invocation per output directory. A source change requires a new study.
The pre-existing strict simulation certificate checks remain in effect.

Each estimator solve commits an NPZ and a checksum manifest before the next
solve. Failed convergence is retained as completed computation, never converted
to a pass. Failed stages and completed attempt reports remain as separate records.
An interrupted solve restarts from its beginning; prior stages are reused.
The time budget is checked between stages, so an active solve may exceed it.
Peak memory is process high-water RSS, not an isolated stage allocation.

The explicitly selected `--input-compatibility archived-full-grid` mode can reuse
only the 30 byte-identical archived files. Its reviewed bridge records their
original Git/package identity and hashes the conservative simulation/validation
import closure. It checks the original certificate, current physics/sampling and
input manifest; it never rewrites HDF5 or refreshes pass flags. This lets new
reconstruction modules change without declaring their results qualified. Changes
to a generation/validation dependency still require a new reviewed comparison.

## Resuming the new extended-scene studies

`tools/audit_scene_sensitivity.py` and `tools/audit_scene_family.py` accept
`--resume` with the identical command and output directory. Their atomic solver
states include both iterate and momentum, plus checksummed identity metadata.
A changed cap, tolerance, solver, source, input, protocol or runtime rejects reuse.
Each doubled-budget fit starts independently. Deadline failures preserve separate
incomplete records; resuming continues their iteration state. An exhausted
iteration cap remains an incomplete scientific outcome, not an automatic retry.
Deadlines are checked between iterations/stages; a running transform or I/O is
not forcibly interrupted. Cache limits cover retained PSF spectra, not total RAM
or VRAM. No GUI physical-solver integration follows from these audit pilots.
