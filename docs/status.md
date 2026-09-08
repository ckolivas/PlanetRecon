# Current implementation and qualification status

Updated 2026-09-09. This matrix supersedes current-status wording in the historical
R10 roadmap; archived experiments and their original decisions remain unchanged.

| Area | Implemented | Qualification / remaining work |
|---|---|---|
| Capture baseline | SER/native AVI, mono/RGB/raw CFA output, translation, nearest-neighbour colour previews | Independent nights/cameras and matched conventional-stack comparisons remain |
| Preprocessing | Separate optional cache, quality/shape exclusions, best retained reference, SER duration, apparent flattening, geometry hints | Spin may be unresolved; apparent flattening alone is not intrinsic shape |
| Geometry | Field and surface motion, Saturn globe/rings, coverage, CPU resume | Physical inference and combined atmospheric/geometry accuracy remain experimental |
| GUI/export | Updated per-run controls, tooltips, cancel/resume, PNG16/TIFF16/float 32 and provenance | Full independent capture workflow and refreshed bundle acceptance remain |
| Compute/resources | Local RTX 5070 venv support, CPU/CUDA translation, scoped Linux RAM/Torch allocator limits, recovery | Shared CPU/CUDA operators pass all 500-frame parity; bounded parallel independent CPU verification is qualified; full reconstruction performance and production integration remain |
| Scientific inputs | 30 certified full-resolution files under recorded generation identity | Input certification does not qualify a changed reconstruction method |
| Scientific reconstruction | Legacy estimators, constrained Newton-CG and coupled-inverse experiments | Extended model matches all 3,000 development frames; the 12-case, 11-frame pilot passes; reference, L-BFGS-B and diagonal Newton-CG pass 25 frames; periodic-inverse Newton-CG regresses at 25/500 frames and is not adopted; full-count convergence remains incomplete |
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
at both 25 and 500 frames. A window-average inverse is dense-tested only; early/late
state, active-mask and boundary sensitivity are next before another fit protocol.

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

Use `tools/audit_full_gate 1.py --inputs ... --out ... --wall-budget-s 3600`
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
