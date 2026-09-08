# Current implementation and qualification status

Updated 2026-09-08. This matrix supersedes current-status wording in the historical
R10 roadmap; archived experiments and their original decisions remain unchanged.

| Area | Implemented | Qualification / remaining work |
|---|---|---|
| Capture baseline | SER/native AVI, mono/RGB/raw CFA output, translation, nearest-neighbour colour previews | Independent nights/cameras and matched conventional-stack comparisons remain |
| Preprocessing | Separate optional cache, quality/shape exclusions, best retained reference, SER duration, apparent flattening, geometry hints | Spin may be unresolved; apparent flattening alone is not intrinsic shape |
| Geometry | Field and surface motion, Saturn globe/rings, coverage, CPU resume | Physical inference and combined atmospheric/geometry accuracy remain experimental |
| GUI/export | Updated per-run controls, tooltips, cancel/resume, PNG16/TIFF16/float32 and provenance | Full independent capture workflow and refreshed bundle acceptance remain |
| Compute/resources | Local RTX 5070 venv support, CPU/CUDA translation, scoped Linux RAM/Torch allocator limits, recovery | Physical solver is CPU; corrected-operator backend parity and whole-job benchmarks remain |
| Scientific inputs | 30 certified full-resolution files under recorded generation identity | Input certification does not qualify a changed reconstruction method |
| Scientific reconstruction | Legacy Fourier estimators and blind prototypes, bounded oracle/gradient checks | Full-grid legacy convergence incomplete; extended forward model now matches all 3,000 development frames; new solver qualification remains |
| Gate-1/Q2/Q3 | Historical reports preserved | Requalification required; Q3 is not authorized |
| Experiment execution | Atomic per-estimator/subset/crop/budget checkpoints in full Gate-1 audit; exact identity resume, failure records and stage-boundary wall budget | No within-solve restart or hard wall deadline; extend to other runners as migrated |
| Releases | Five native GitHub tag-build targets, Linux local CPU/CUDA packaging | Refresh artifacts; owner license/signing decisions for publication; Windows/macOS runtime tests excluded |
| Capture interpretation | IR642 Mars and L3 Mars are both OSC RGGB | True mono Mars and further independent captures still need sourcing and permission records |

New operator evidence: [full-scene consistency decision](../results/p1-scene-detector-full/DECISION.md).
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
