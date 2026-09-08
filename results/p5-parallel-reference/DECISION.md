# Bounded parallel independent CPU verification

2026-09-08. **The eight-worker normal and adjoint arrays are bit-for-bit equal
to the one-worker ordered CPU sums on all500 frames.** The resulting constant
probe distance bound also matches the earlier serial profile (zero measured
relative difference). This qualifies the bounded frame-parallel implementation
for these independent verification operations, not a scientific reconstruction.

| Frame workers | Normal operation (s) | Weighted adjoint (s) | Peak process RSS (bytes) |
|---|---|---|---|
|1|110.002|58.383|2,638,057,472|
|8|35.581|17.111|3,514,109,952|

The original spatial CPU operators are unchanged. Each FFT uses one thread;
at most eight futures are pending and results are summed in input order.
The pending-result bound excludes stored input/operators, output arrays,
accumulator and worker workspaces. This is not a total process memory ceiling.
The caller caps BLAS at two threads. Eight bounded tests cover mono/RGB/CFA
parity, result order, solver/certificate agreement and worker exception handling.

Data/PSF preparation took48.33s and the whole profile270.53s. The profile overlapped
endpoint studies; its observed timings do not establish an isolated or general
speedup. The final result does support using this verifier in a new prospective
study to reduce the measured serial-check bottleneck. Existing running studies
and their identities were not changed.

[Report](report.json), [protocol](protocol.json), per-worker records and
[checksums](checksums.json) retain byte-identical evidence and source/input
identities. No source/input changed during the profile. The constant scene is
an operator probe, not a certified solution; Q3 remains unauthorized.
