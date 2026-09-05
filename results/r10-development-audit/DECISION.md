The advanced atmospheric solver remains experimental. Q3 is not authorized.

This bounded development audit covers seeds 1001–1003, D/r0=4 and 8, and both
feature and bland crops: 12 cases, two starts, M=15/35/60, one outer iteration per
stage, two phase iterations, eight frames, 32×32 crops and a 16-sample pupil
diameter. It uses the physical frozen-flow simulator, not a phase-basis generator.
The protocol was written before execution. Wall time was 106.33 s on CPU.

E1/E2a0 relative disagreement was at most 4.48e-11. All six bounded low-frequency
ranking comparisons preserved S10 membership (Jaccard=1). These establish local
controls; they do not qualify the full development family or convergence grid.

Every Q2 case is incomplete because the prescribed bounded optimization did not
establish convergence. High closure values on these small crops are diagnostic
only. Known-shift pupil mapping had maximum centroid errors of 0.0124–0.2642 px
across captures. The 60-mode projected midpoint snapshot differed from the
exposure-averaged OTF by 3.90–18.28% in relative norm. This combines phase-basis
truncation and exposure mismatch; it does not isolate pure exposure error.
Per-frame basis residual RMS and known/estimated/no-shift fixed-truth-object
residual controls are retained in each record. Those controls use truth only for
diagnostics and do not choose the blind model or prior.

The held-out frames select initialization; they are not an independent assessment
partition. Eight-frame reduced crops do not establish the original Gate-1 or Q2
outcome, production full-disc atmospheric recovery, or real-data quality. No
historical evaluation seed or table was regenerated. Remaining scientific work:
full W01 convergence families, isolated exposure/basis/operator ablations with
predeclared budgets and independent assessment, followed by W03 affected-family
requalification. W11 advanced claims remain gated; the geometry baseline remains
available independently.

Reproduce in a new directory with:
`PLANETRECON_THREADS=2 python3 -m planetrecon.science_audit --out /tmp/new-audit`

The manifest hashes all per-case reports. Synthetic HDF5 inputs are regenerable;
their hashes are recorded, while the 36 MiB files remain local outside Git.
