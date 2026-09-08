# Summarizing complete and partial family evidence

Inspect an unfinished case without modifying its files:

```sh
.venv/bin/python tools/scene_family_progress.py out/p2-selection-family/case-01
```

This prints the five selection records, stored stages, latest optimizer traces
and final-report availability. It rechecks recorded selection certificates and
stability against the frozen contract. An optimizer trace or a stored-stage
convergence flag never substitutes for independent CPU evidence. Every record
read has its own checksum; files may advance during inspection, so this is not
an atomic snapshot of the entire case or a source/input verification.

Process liveness remains explicitly unknown. A missing final report, an old
trace or an empty restricted process listing does not prove the run has stopped.
Check the owning execution session before starting or resuming any study. Never
start a second scientific run solely because a progress report appears stale.

After each completed case, archive its JSON evidence and decision without
rewriting prior attempts. Produce a new summary path with:

```sh
.venv/bin/python tools/summarize_scene_selection_family.py \
  --cases results/p2-selection-family/case-00 \
  --out results/p2-selection-family/summary-through-00.json
```

Add subsequent case directories to `--cases` as they become available. Never
aggregate an actively written report. Missing cases remain explicitly missing;
a partial report retains its execution failures and can expose individually
qualified selections without qualifying its case. All 12 cases and 60 selections
are required for a family pass.

The checker revalidates the frozen observed ranking, selected indices, count,
prior, numerical and execution contract, shared source/runtime identity,
input/manifest hashes and prerequisite evidence. It examines both independent
CPU certificate records, solver identity and iteration caps, feasibility, and
both latent/detector changes. Stale pass flags, missing values, nonfinite or
negative error bounds, duplicate selections and mixed protocols cannot qualify
the family. Duplicate case reports are rejected rather than silently choosing
one attempt. Select one identified attempt per case and retain earlier attempts.

Each summary hashes the exact JSON bytes it reads and rechecks them before
writing. Existing summary paths are rejected. The summary verifies recorded
certificates; it does not rerun the CPU operator or load image checkpoints.
Original per-fit independent verification and source/input checks remain the
source of the numerical evidence. A summary does not qualify scientific prior
choice, blind reconstruction, production integration or Q3.
