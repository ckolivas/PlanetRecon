# Performance audit: 12–13 September 2026

Authorized unattended work ends 2026-09-13 04:01:42 UTC. Continue measured,
output-preserving improvements without questions; commit completed steps. An
hourly heartbeat in this task is scheduled only through that deadline.
Stop earlier and pause the heartbeat if no reasonable candidates remain; do not
fill the time with speculative changes or redundant qualification.

## Measurement and behavior constraints

- Use `.venv/bin/python`, real CUDA where appropriate and at most 32 CPU threads.
- Existing background processes remain untouched. Record this process's CPU time
  separately from contended elapsed time. Trace this process's CUDA activities;
  do not infer work from machine-wide utilization. Run comparisons serially,
  freeze their source revisions and use alternating before/after order.
- Preserve float64 interpolation, raw CFA coverage, frame selection and confidence
  gates, checkpoint/cancellation behavior, and best-frame anchor/midpoint epoch.
  Keep local patch alignment and the upper 50% quality range as defaults. Do not
  restore rejected weighting/interpolation variants or add sharpening.
- Startup/opening stay idle. Run validates/reuses cached preprocessing or computes
  it when necessary. The separate Preprocess and Inspect actions remain explicit.
- Preserve user edits in AGENTS.md and docs/development-plan.md, captures and
  untracked prototypes. No package installation, release publication or external
  messages are needed for this audit.

## Starting point

HEAD `781c6a3`. Previous improvements and scoped evidence are in
`results/real-data/runtime-speedups.json` and `owned-runtime-speedups.json`.
The latest changes avoid BLAS worker spin in motion correlations (`aa3ff76`),
share CUDA pull coordinates (`d32c88e`), and keep source selection idle (`781c6a3`).

A new 34-frame CPU local-stack profile (4 threads, same RGGB Saturn pilot) records
5.83 CPU-seconds, with 2.41 in pull resampling, 1.90 in local registration, 0.89
in demosaicing and 0.48 in phase correlation (overlapping cumulative categories).
The frozen baseline is `/tmp/planetrecon-speed3-before` (archive of 781c6a3).

## Work queue

1. Constant/zero-translation fast paths: complete in ad397ab.
   107 tests pass, including real CUDA; all ABBA capture outputs are identical.
   Evidence: results/real-data/translation-runtime-speedups.json. Ordinary CPU
   median 5.90 -> 5.175 CPU-seconds; Saturn CPU was effectively unchanged, while
   Saturn CUDA median was 4.485 -> 4.21 CPU-seconds. Wall times are contended.
2. Immutable reference FFT reuse: complete in 1f4fecd.
   Rebuilt for mean template, resume and CPU fallback; unprepared mutable inputs
   are not cached implicitly. Independent replacement/fallback/resume tests pass,
   and real capture outputs remain exact. Evidence: reference-runtime-speedups.json.
   Ordinary CPU median improves about 2%; whole-run CUDA timing shows no reliable
   gain, although repeated reference FFTs are removed.
3. Prepared local motion reference: 101 tests pass with real CUDA. Exact
   Saturn/surface ABBA outputs; timing changes are within noise. Stacking and
   uploading the fixed reference once saves 33 uploads / 86.5 MB in the 34-frame
   pilot, retaining 2.5 MiB on CUDA during the run. See motion-reference-speedups.json.
   Remaining CPU/CUDA costs were profiled and led to demosaic geometry reuse below.
4. Removed one redundant frame copy per 16-bit SER read. 69 reader/cache tests
   pass; identical decoded digests in serial ABBA. Synthetic decode-plus-hash
   own CPU improves about 2% little-endian and 4% big-endian. Supplied real
   captures are all 8-bit, so this does not accelerate them. Evidence:
   ser-decode-speedups.json. Full content verification remains unchanged.
5. Unused GUI previews removed in e06c10b; other consumers
   retain the default stream. 18 worker/GUI tests pass, including cancellation,
   snapshots, legacy checkpoints and batch runs. Exact replay outputs; unused
   serialized payload falls by 37 MB for 34 frames at batch size 32. No reliable
   default-batch timing gain. See worker-preview-speedups.json. Export/checkpoint
   review found validation, compression and atomic publication work that must be
   retained; no measured reason to rewrite those paths.
6. Integration qualification complete for the nine improvements below; see
   the exact result and next bounded candidate at the end.

## Deferred experiments

A scratch NumPy dense CFA backprojection sharing corner geometry gave mixed
microbenchmark gains (10 calls: original 0.52/0.60 CPU-seconds vs 0.52/0.45), with
up to 8.6e-14 contribution differences. It is not in production. Prefer the
simpler exact translation/reference reuse first; this candidate needs clearer
benefit and bounded allocation evidence before reconsideration. Scratch file:
`/tmp/try-shared-cpu.py`.

An exact-zero residual inversion shortcut was not implemented: the real Saturn
pilot had zero all-zero residual fields among 34 frames, so it would not help
that capture. Existing nonfinite/divergent residual safeguards remain intact.

## Preprocessing metadata repetition

Saturation screening fetched metadata once per frame. SER metadata computes
statistics over the full timestamp trailer, making this part quadratic in frame
count. Bit depth is now read once per screening pass (lazily, so disabled
saturation checks/cancelled passes do not fetch it). 39 tests pass; exact
measurements, masks and summaries in serial ABBA partial real-capture screenings.
For 256 frames, Saturn L own CPU median is 1.47 -> 0.975 s; Mars L3 is
4.105 -> 0.695 s. This does not include geometry discovery. Evidence:
results/real-data/preprocess-metadata-speedups.json. Full verification remains.

Timestamp epoch conversion also replaced its Python integer loop with exact
unsigned integer differences before float64 scaling. Monotonic spans fit uint64
even across signed int64 boundaries. 82 timing/exposure/viewing tests pass.
64 metadata reads: Saturn CPU median 0.14 -> 0.015 s; Mars 0.82 -> 0.07 s.
Metadata remains freshly read and validated. See timestamp-conversion-speedups.json.

The latest Saturn CUDA profile (34 frames, 4 threads) attributes 4.11 CPU-seconds
to this process: 1.56 cumulative in local residual coordinates, 1.48 in building
the motion template, 0.94 in ring drift registration, and 0.91 in 69 CPU bilinear
demosaics. Categories overlap; this is not exclusive device timing. This led to the
reusable demosaic geometry described below.

## Reusable Bayer geometry

Demosaic masks and float64 neighbour counts are now prepared once per run.
The arithmetic and neighbour order are unchanged; there is no global cache.
Private retained storage is 22 bytes per pixel (3.44 MiB for the Saturn pilot).
All four serial ABBA replays have exact image/coverage/validity/count parity.
Own CPU medians: ordinary CPU 4.995 -> 4.485 s, ordinary CUDA 2.385 -> 2.000 s,
Saturn CUDA 4.025 -> 3.545 s. CPU Saturn is variable and does not establish a
gain (24.18 -> 25.155 s); do not advertise a CPU Saturn improvement.
Independent pixel-oracle and CPU/CUDA/memory/resume tests pass. Two outdated
test assumptions were corrected: small-patch refusal and allocator pool reuse.
See results/real-data/prepared-demosaic-speedups.json.

## Bounded parallel screening

Independent frame measurements now use at most four threads, bounded further by
the requested threads, batch count and a conservative 128 MiB scratch estimate.
BLAS stays single-threaded inside this pool and is restored afterward. Without
optional threadpoolctl, screening remains serial. Source I/O and result ordering
remain on the calling thread; cancellation closes queued work and joins workers.
Exact real-capture parity. Saturn L 256-frame median elapsed 0.922 -> 0.507 s;
Mars L3 0.637 -> 0.420 s. Own CPU increases about 22% (0.970 -> 1.185 and
0.705 -> 0.860 s): this is an explicit bounded concurrency tradeoff, not a CPU
work reduction. See parallel-screening-speedups.json. Parallel calibration,
ordering, cancellation, exception cleanup and missing-runtime fallback are tested.

## Integration qualification and continuation

Completed production commits, in order: ad397ab, 1f4fecd, e06c10b, 378abd3,
08b62c3, b2a50c6, abe8b52, d3c0475 and fb83392. Each candidate has focused
regressions and scoped measurements above; none changes reconstruction samples.

The tracked integration suite collected 2053 tests with real CUDA explicitly
opted in. 2017 passed in the broad invocation, and 31 passed on corrected
reruns; 5 slow/scientific tests remained opt-in/skipped. No unresolved failures.
The first broad launcher invoked pytest.main from stdin, which cannot be reopened
by multiprocessing spawn; do not repeat that launcher. Invoke `.venv/bin/python
-m pytest` normally, passing tracked test filenames as arguments through a
subprocess or shell array. There is no reason to repeat the whole suite unchanged.

GUI expectations now cover the actual preview subscription, refusal after
automatic preparation, and retained local alignment when switching from None
to Saturn. The corrected worker recheck log is out/run-speed/integration-recheck.log
(30 passes), followed by the corrected Saturn mode-switch test (1 pass). Earlier
pytest lastfailed entries include removed/renamed historical tests, so do not
blindly replay that cache. Production defaults, startup-idle behavior, saved
settings semantics and full content checks remain unchanged. User edits in
AGENTS.md and docs/development-plan.md are untouched.

Next bounded candidate: flat-field correction currently recomputes the same
normalization/division array per frame in apply_calibration. Supplied real
captures do not include calibration flats; use clearly labelled synthetic
calibration only to measure this path, and retain a change only if it improves a
calibrated replay with identical outputs. Any prepared calibration must own its
arrays, refresh between runs, preserve dataclass-based calibration provenance and
cache/resume identities, and retain the existing flat-scale/invalid-flat rules.
Do not silently cache caller-mutable arrays or merely benchmark a scalar helper.

Other larger possibilities (persistent GPU pipelines, combining job stages,
fusing interpolation or altering reduction order) lack evidence sufficient to
justify their correctness/memory tradeoffs now. SER reads already have bounded
batching and source checks; weakening full cache verification is excluded. More
parallelism is not automatically better than the measured four-worker cap.
After investigating the remaining bounded candidate, reassess whether any
reasonable output-preserving option remains. If not, stop early and pause the
heartbeat rather than filling the authorized 24 hours with speculative work.
