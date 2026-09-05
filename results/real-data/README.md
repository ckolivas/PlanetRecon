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

W16 acceptance remains incomplete: these two OSC captures do not establish a
multi-camera corpus, geometry accuracy, supported-GPU parity, or universal
performance limits. Real-data reports cannot qualify the advanced solver or Q3.
