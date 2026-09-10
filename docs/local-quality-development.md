# Development with local alignment and the upper 50% quality range

Updated 2026-09-10. This is the current direction following the user's selected
best settings. It supersedes earlier default and next-step instructions in the
historical development log. New GUI and CLI jobs already use these defaults
(commit `512be78`). Existing saved jobs retain their settings.

Use local patch alignment, Motion None, original linear Emil quality weights,
and scores strictly above `(capture best + capture worst) / 2`, followed by cached
screening. This is a quality-range cutoff, not half the frame count. Preserve the
best selected frame as the reference origin, normal template selection, original
CFA measurements, direct colour support and the existing local ambiguity/fold
guards. No automatic sharpening. Stronger quality weighting remains optional.

## Completed: reference-coordinate interpolation for local maps

The previous quadratic prototype supported global translations only. Building
its global-only resumable engine is deferred: it would not support the chosen
working mode. The new separate [local transport probe](../tools/cfa_local_transport_probe.py)
locates observed raw detector centres in the reference plane by inverting the
actual local pull map. It refuses maps outside a conservative contraction bound
or whose inverses do not converge. It uses the frozen radius-four triweight and
quadratic fit, then affine under an ordinary-stack iid-noise variance cap, then
ordinary colour completion when sample geometry cannot support either fit.
Only ordinary observed output channels are eligible. Direct CFA support remains
separate from reconstructed-image variance. Frame contributions are staged before
publication; cancellation or an invalid map leaves all preceding sums intact.

Twenty-one controls pass: four Bayer layouts, known row-shear maps, polynomial
and curved scenes, midpoint selection, an independent weighted least-squares
oracle for translation, raw-impulse variance and colour isolation, inverse refusal,
frame cancellation, chunk independence and exact cropped/full geometry agreement.
A preliminary check also matched the earlier frozen global prototype to 1e-10
on its complete interior; the committed control uses a self-contained independent
least-squares oracle. These are image-sampling controls, not physical detector
PSF or motion-estimation qualification. Local deformation changes pixel footprints;
that physical effect remains outside this interpolator's model.

The [fixed Jupiter pilot](../results/real-data/cfa-local-transport.json) uses 64
uniformly spaced entries among the 1,645 frames passing the upper-half range and
screening. It includes anchor 1947, uses the normal 64-frame selected template,
and shares the exact local maps and original linear weights between both arms.
The fixed central crop is y=164:260, x=280:376; ordinary support and completion
are calculated on the full detector. A shared registration from the full ordinary
stack positions the conventional unsharpened reference. Both comparisons fit
per-colour gain/offset on the same 6,400-pixel interior, without sharpening.

RMS difference falls from 0.00765598 to 0.00516725 (32.51% relative reduction),
and sigma-three highpass correlation rises from 0.633049 to 0.850667. All 27,648
crop channel samples support quadratic fits under the variance cap. The run took
79 seconds using GPU registration and CPU fitting. This passes the predeclared
requirement that both metrics improve. It is a small development crop, not proof
of improvement for the full selected stack; the conventional reference is not
independent truth. No application reconstruction code changed.

Private evidence is under `out/cfa-local-transport`: the prior protocol, exact
run script/source, logs, indices/template, hashes, float32 TIFFs and matched PNG
previews. `ordinary.tif` and `quadratic.tif` contain unsharpened crop intensities;
PNG previews share display scaling after the declared gain/offset comparison.
The [reproduction script](../tools/cfa_local_transport_pilot.py) requires the private
capture and its existing cache and refuses to overwrite a completed pilot.

## Completed: bounded batched local execution

The [execution-only follow-up](../results/real-data/cfa-local-execution.json)
batches outputs with equal neighbour-list lengths while retaining each output's
own sample positions, colours and weights. It does not share local geometry.
Neighbour counts are checked before list construction and dense allocations;
the default entry budget is 65,536. A conservative array-plan check defaults to
2 GiB and refuses larger requests before allocating accumulator arrays. This is
not an RSS cap: caller inputs, interpreter/library and KD-tree overhead are extra.
Cancellation is now checked during inverse-map iterations, neighbour work and
bounded final-fit blocks, preserving previously completed frame sums.

All 38 controls pass. A sequential four-frame synthetic profile at 424x656 pixels,
with the same 96x96 output crop, reduces mean frame time from 0.7834 to 0.5176
seconds (1.51x faster). Final fitting is 0.092 versus 0.101 seconds. Peak RSS is
218.2 versus 222.7 MiB. All checked moments, image, variance, fit choices and
ordinary coverage/validity match the original implementation bitwise. The profile
ran with two CPU threads while a GUI worker was active; these illustrative timings
are not a total application speed claim. No real capture comparison was repeated.

## Completed: larger selected-frame confirmation

The [512-frame confirmation](../results/real-data/cfa-local-confirmation.json)
passes the frozen gate below. All 512 selected frames contribute with the same
local maps, original quality weights, template and anchor for both arms. Each
region contains 6,400 common comparison pixels. RMS reductions are 9.63% on the
left, 10.67% in the centre and 10.42% on the right. Detail correlations improve
from 0.887706 to 0.957857, 0.907276 to 0.964155, and 0.893505 to 0.960559,
respectively. Aggregate RMS falls from 0.00466010 to 0.00418363 (10.22%), and
aggregate detail correlation rises from 0.897687 to 0.961324.

There were no unexpected frame/map rejections. Both raw float TIFF strips and
matched PNG previews are saved in `out/cfa-local-confirmation`; the exact script,
source, declaration, selection/weight manifest and hashes are archived there.
The reproduction script is `tools/cfa_local_confirmation.py`; it refuses to
repeat a started study. Two gate controls ensure all regions must pass and
that an aggregate improvement cannot hide a regional regression. No application
processing changed. This is a selected-frame strip confirmation, not a full
1,645-frame or whole-globe result; the reference remains non-independent.

## Completed: exact fixed-input CPU resume

The [checkpoint wrapper](../tools/cfa_local_resume.py) now binds the original
source identity, ordered selected indices/qualities, every calibrated raw-frame
and dense-map digest, output region, interpolation/package source and execution
policy. It also records NumPy/SciPy versions, byte order and the resume source
version. Inputs are verified before a frame enters the existing staged accumulator.
Final output is refused until every selected frame is complete.

Checkpoints preserve all six accumulator arrays and a completed-frame count, with
array and metadata checksums. Atomic replacement occurs only after the temporary
file is complete and cancellation is checked. Failed or cancelled saves preserve
the previous checkpoint; loads verify headers before loading arrays, then validate
identity, shape, dtype, finiteness, checksums and execution history before returning
a resumed run. Capture paths/symlinks and unrelated NPZ files cannot be replaced.

All [37 checkpoint controls](../results/real-data/cfa-local-resume.json) pass,
including exact results for all four CFA layouts at empty/intermediate/completed
checkpoints, cancellation without double counting, changed inputs/maps/policies,
failed writes/replacements and malformed state. The combined local controls total
77 passes. This is CPU accumulation of externally frozen raw frames/maps, not
GPU fitting or automatic registration recovery. The implementation remains a
separate prototype, and no application defaults or processing path changed.

## Completed: CUDA moments and complete-frame CPU retry

The [GPU execution qualification](../results/real-data/cfa-local-gpu.json) adds
bounded float64 CUDA moment products. A conservative detector-space window encloses
the same radius-four reference-space neighbourhood; inverse-map coordinates and
kernel weights determine the contributing samples. Inverse geometry, ordinary CFA
projection and final fitting remain on CPU. All six frame contributions are
validated before publication. GPU failure discards partial work and retries the
whole frame on CPU, then retains CPU execution for the rest of the run.

The checkpoint wrapper binds CUDA source and execution settings, records each
completed frame's backend and the fallback boundary, and preserves the device and
runtime identity. Same-device continuation and continuation after a CPU transition
are exact. A different available GPU/runtime refuses active GPU continuation;
unavailable CUDA triggers complete-frame CPU retry. Cancelled retry can be saved
without counting the incomplete frame. Caller-owned raw frames and maps remain
frozen and digest-checked; registration is not repeated by this wrapper.

All 124 distinct controls pass: 86 CPU and 38 CUDA checks. Full-output synthetic
424x656 profiles preserve the original CPU arrays and image bitwise after staging.
The CUDA image differs by at most 1.42e-13; variance, fit choices, ordinary image,
coverage and validity are bitwise identical. A first CUDA implementation retained
tree lookup and gave no useful gain. The detector window reduces warm frame time
to 4.94 seconds versus 7.50 seconds for the original CPU implementation (1.52x).
Final fitting takes 1.84 seconds versus 1.80 seconds. Staged CPU alone was slower
in this profile. This is three-frame synthetic accumulation, not total application
throughput or a new image-quality comparison.

The RTX 5070 profile used Torch 2.13.0+cu132, two CPU threads and at most 51,200
neighbour entries per batch. Peak PyTorch CUDA allocation was 57.84 MiB, including
32 MiB retained allocation. Process peak RSS was 2,095 MiB versus 1,190 MiB for
original CPU, including runtime overhead. Array budgets are not RSS caps, and
PyTorch metrics exclude allocations outside its allocator. Chunk sizes remain
explicitly bounded rather than tuned to GPU core count. Private scripts, source
snapshots, paired arrays and logs are under `out/cfa-local-gpu`.

## Completed: optional application and GUI integration

The qualified CPU/CUDA implementation now lives inside the installable
`planetrecon.pipeline` package. Historical probe imports forward to that same
implementation; installed applications do not import the development tools.
The new **Local colour interpolation (experimental)** checkbox in the Capture
tab is off by default. CLI users can add `--local-cfa-interpolation`. It requires
a Bayer capture, cached preprocessing, local alignment, Motion None and original
linear weights. Local alignment and the upper 50% quality range remain the defaults.
No sharpening is applied.

Each calibrated frame and its actual application local map enter the interpolator
once. A moment failure retries that same frame/map on CPU without repeating
registration. Each completed frame records its source index, original weight,
raw digest and map digest. The existing global fallback for small captures or
insufficient local template candidates remains explicit in result warnings.
An unqualified inverse map stops the run instead of silently changing algorithms.

Batch previews and intermediate snapshots show the ordinary local CFA stack;
per-frame progress reports moment accumulation and the final fit. The final event,
scientific snapshot and image export contain the fitted RGB result. Effective
sample layers (`iid_effective_samples_R/G/B`) describe reciprocal relative variance
under independent equal-variance detector noise. They are separate from direct
CFA weights and are not actual frame counts or calibrated photon/read noise.
Exported layers carry their distinct units.

Optional application checkpoints save the reference template, all accumulated
moments and ordinary sums, completed-frame manifest, registration backend and
moment execution transitions atomically. Identity binds capture content,
preprocessing, calibration, settings, package code and numerical runtime; frozen
builds also bind the executable content. Loads check array headers, checksums and
state before continuing. Already completed maps are not recomputed; future maps
are estimated once from the saved template on the same registration backend/runtime.
Registration CPU fallback is retained across continuation. Cancellation keeps the
last published checkpoint; an incomplete batch may be replayed from that point.
The earlier fixed-input wrapper still supports externally pre-frozen future maps.

End-to-end synthetic controls cover all four Bayer layouts, actual local alignment,
parity with the qualified operator, unchanged selection/template/coverage,
midpoint selection, cancellation during moments and final fitting, exact CPU and
CUDA continuation, injected GPU failure, corrupt state, GUI control changes,
spawned worker delivery and full-resolution snapshot/TIFF export. An installed-wheel
check confirms fitting and GUI controls work without development-tool imports.
Private logs and the rendered Capture controls are under `out/cfa-application`.

## Next steps

1. User test of the full Jupiter capture: restart the venv-launched GUI, keep local
   alignment and the upper 50% quality range, enable **Local colour interpolation
   (experimental)**, then run using the existing preprocessing cache. Use a new
   checkpoint/output filename for this candidate. Compare the final unsharpened
   output against the ordinary local stack using matching display levels.
2. Use that full-result assessment to decide whether the candidate merits further
   work or default promotion. It remains opt-in: the previous three-region,
   512-frame confirmation is not a full 1,645-frame or whole-globe validation.
   No capture study was repeated for this application integration.

### Declared larger confirmation

Use 512 uniformly spaced entries from the same full capture midpoint selection,
including the best selected frame by replacing the nearest sampled entry if
necessary, then sorting. Preserve the application's top-64 selected template,
anchor 1947, cached linear qualities and local patch rules. Both arms share each
raw frame and exact local map. Stop and report an unexpected rejection or unsupported
map rather than changing selection to make the comparison run.

Fit the fixed strip y=164:260, x=184:472, containing three adjacent 96x96 regions:
left x=184:280, original centre x=280:376, right x=376:472. Compute ordinary support
on the full detector. Use one registration from the full ordinary image for both
arms. Exclude eight pixels around each region for comparisons and use identical
validity masks. Fit per-colour gain/offset separately for each arm and region,
using the same unsharpened conventional reference and sigma-three highpass metric.
Require lower RMS and higher detail correlation in each region; also report the
aggregate. This gate is frozen before inspecting the new outputs. If it fails,
stop this candidate without window/weight tuning or a full-capture rerun. Record
indices, weights, template and map hashes, comparison masks and paired float outputs
under a new `out/cfa-local-confirmation` directory. This remains a development
comparison, not independent truth or proof of a full selected-stack improvement.

Physical motion modes and transformed detector footprints require separate
qualification. The earlier rejected global stronger-weighting/first-moment
integration stays reverted. Do not repeat completed global or rejected weighting
studies. The periodic continuation automation remains paused; this document does
not schedule background work.
