These reports describe private local Bayer SER captures supplied by the user.
Distribution permission is unset; capture files and reconstructed images remain
outside Git. Reports omit filenames and observer metadata and bind each input
by SHA256. Independent additional nights/cameras and missing mono/rotation
categories must still be sourced by the user.

The predeclared baseline uses CPU float64, 2 threads, 32-frame batches, no supplied
calibration, no sharpening and no field/surface/Saturn geometry. The whole capture
is processed. Assessment uses 128 uniformly spaced frames, split into alternating
halves, plus an in-sample raw detector residual with detector-fixed CFA sampling.
These are consistency diagnostics, not independent recovery or noise calibration.
The first-preview clock starts immediately before stacking; opening and hashing
are excluded. RSS includes the assessment and result serialization. Concurrent
benchmark/build work was running on this host; rates are not isolated maxima.

Jupiter: 3,749 frames, 424×656, RGGB8; 3,743 used and 6 rejected; 467.20 s;
first preview 3.84 s; peak RSS 201.8 MiB. Split-half relative RMS 0.02324;
in-sample raw relative RMS 0.06010. These figures establish a local baseline only.
Saturn full-capture processing is recorded separately when complete.

Reproduce locally in a new output directory:
`python3 -m planetrecon.benchmark --path CAPTURE.ser --out NEW_DIRECTORY --id PRIVATE_ID`
The directory contains the protocol, report and full-resolution result NPZ.
Use `--max-frames` only for explicitly labelled bounded checks.

W16 acceptance remains incomplete: the local corpus does not establish geometry
accuracy or universal performance limits. Local baseline GPU parity is reported
below. Real-data reports cannot qualify the advanced solver or Q3.

The completed CUDA OSC Saturn run used 17,928/17,934 frames (6 rejected),
320×512 RGGB8, in 332.69 s (53.91 frames/s), with first preview at 1.83 s and
1,383.6 MiB process peak RSS. This includes the Torch runtime; full-capture
results are retained locally in `out/real-data/osc-saturn-gpu-full/result.npz`.
The earlier unfinished CPU Saturn attempt was not treated as completed evidence.

Five 512-frame samples uniformly spanning the full local captures now cover mono
and OSC Mars, mono and OSC Saturn, and OSC Jupiter. The IR642 Mars data is treated
as mono per user confirmation despite its RGGB header; L3 Mars remains RGGB.
All paired CPU/CUDA outputs have exactly equal image arrays, validity and used /
rejected counts. Stack speedups were 2.10–4.32×; these were concurrent desktop
measurements, not isolated speed records. The three new captures contain 27,689,
116,691 and 116,629 source frames; their whole-file identities were hashed,
while reconstruction used the explicitly reported uniform samples. Full new
capture reconstruction is not claimed. See the paired JSON files and
`results/gpu/rtx5070-local.json` for counts, timings and residuals.
