# Capture preprocessing

Use the **Preprocess** button or the separate `preprocess` command whenever
fresh measurements are needed. It measures quality, shape and geometry without
reconstructing or replacing the displayed result. The measurements are cached
beside the capture as `<capture filename>.planetrecon-preprocess.npz`.

The **Use cached preprocessing (quality and shape)** checkbox independently
controls whether later runs use those decisions. Processing never launches a
new analysis automatically. Without a cache, runs apply only their existing
validity, saturation and registration checks. A stale or corrupt selected cache
requires another Preprocess action or disabling cache use. Input files are not
modified.

The GUI shows quality exclusions, shape exclusions, overlap, other invalid or
saturated frames, the unique total excluded, and the retained frame count before
a run. Missing targets and clipped silhouettes count as shape failures. Toggling
cache use updates the display; changing input/calibration settings marks the
counts unverified until inspection or preprocessing validates them.

```sh
.venv/bin/python -m planetrecon preprocess --path capture.ser
.venv/bin/python -m planetrecon stack --path capture.ser --device gpu
.venv/bin/python -m planetrecon stack --path capture.ser --no-frame-preselection
```

`preprocess --config settings.json` accepts calibration and geometry settings;
`--bayer` and `--cadence` provide direct overrides. To choose another cache
location, use `preprocess --cache measurements.npz` and
`stack --preprocessing-cache measurements.npz`.

Python callers can call `preprocess_source(source, config)` independently, then
`stack_source(source, config, preprocessing=selection)` with its returned
selection. File-backed sources also save/load the sidecar automatically. Set
`ReconstructionConfig(frame_preselection=False)` to ignore preprocessing.

## Cache validity

Input inspection and preprocessing report capture duration from the first and
last per-frame timestamp. For SER, integer 100 ns trailer ticks are subtracted
before conversion to seconds, preserving small intervals at large absolute
epochs. Irregular cadence and dropped-frame gaps remain part of the duration;
the last exposure's unknown length is not added. Measured timestamps take
precedence over a supplied cadence. Missing timestamps require an explicit
cadence to estimate seconds; duplicate or reversed times show an invalid timing
report instead of a guessed duration. A single timestamp has zero span.
The GUI capture line, SER metadata and preprocessing report expose this timing.

Cache identity hashes **all observed pixels**, frame layout/colour, bit depth,
timestamps and actual calibration tables/settings. It distinguishes indexed
subsets of the same source. Thread count, device, batch size and reconstruction
geometry can change without invalidating quality/shape decisions. Checking a
cache reads the observations but does not repeat quality or shape estimation.
Geometry suggestions with incompatible new timing/viewing settings are labelled
outdated independently of the still-valid frame selection.

Writes are atomic. Cancellation and failed analysis retain the previous cache.
The input is checked before and after analysis to catch changes during the pass.
Accumulator resume verifies the cached accepted indices and weights as well as
the original input/configuration; refreshing only geometry suggestions does not
invalidate otherwise identical sums.

## Quality estimator and attribution

Emil Kraaikamp describes AutoStakkert's [noise-robust gradient estimator](https://astrokraai.nl/autostakkert.php)
and subsequent [Laplacian estimator](https://www.astrokraai.nl/software/secret_2018_JLD.php).
PlanetRecon implements an explicitly specified adaptation of that estimator
family, not a verified reproduction of AutoStakkert's proprietary formula or
its Noise Robust control. Its public description does not establish the exact
kernel, normalization and noise-filter parameters used here.

For each calibrated, finite, unsaturated frame:

1. Form a luminance measurement plane using 2×2 area means. Each Bayer cell
   averages its R, G, G and B samples, suppressing the colour lattice without
   demosaicing the data used for reconstruction. RGB uses weights ¼, ½, ¼ and
   then the same area bin; mono uses the area bin directly. Odd trailing rows
   and columns are omitted from measurements only. Mono/RGB below 10 pixels
   on either side use their original grid.
2. Apply a Gaussian with σ = 1 measurement pixel to suppress detector noise.
3. Estimate sky from the median border intensity and sky noise from 1.4826
   times its median absolute deviation. The silhouette threshold is the
   larger of 15% of peak contrast and six times sky noise above sky.
4. Keep the largest connected silhouette (at least nine measurement pixels),
   excluding detached moons and isolated hot pixels. Reject absent targets
   and silhouettes touching a measurement-grid edge.
5. The score is the mean squared four-neighbour discrete Laplacian of the
   smoothed plane over the silhouette dilated by two measurement pixels,
   excluding the outermost image border. Larger scores mean sharper frames.
   There is no brightness normalization, so changing transparency or gain
   can affect scores. This filter is only for measurement, not sharpening.

## Planet dimensions and rejection

The silhouette's binary second moments determine its principal axes. Project
its pixels along both axes, including the projected extent of each pixel, to
obtain apparent width and height in original detector pixels. The major axis
is treated as horizontal separately for each frame, so camera roll does not
turn a broad ring system into an apparent height increase. Width and height
have separate statistics; neither is forced to equal a circular diameter.

These are *apparent illuminated silhouette* dimensions including attached
rings and phase, not a recovered globe diameter or a uniquely determined
physical equator. A crescent's longest apparent axis can be its polar direction;
a near-circular disc does not constrain orientation. Ring/phase modelling in
the reconstruction remains controlled by the existing geometry settings.

Over frames with valid quality and complete silhouettes, calculate arithmetic
means and population standard deviations (`ddof=0`). Make one simultaneous,
non-iterative selection:

- Reject quality **below mean − 2σ**. High-quality frames are retained.
- Reject either apparent dimension **outside mean ± 2σ**.
- Keep equality at the boundary and constant/single-frame populations.

The finite/saturation, missing-target and clipped-target rejections happen
before statistics. All three statistical tests use the same initial valid
population. With few frames or many bad frames, two-sigma filtering has limited
discriminating power; it does not guarantee a particular retained percentage.

## Reconstruction, progress and continuation

The next pass processes accepted frames in their original order, retaining
original indices, timestamps and calibration. Translation and ordinary
geometry stacking reuse the measured quality as their accumulation weight.
Saturn retains its existing region-masked globe-texture weighting to avoid
mixing illumination boundaries. Raw CFA backprojection and final RGB completion
remain the image formation steps; no sharpening is added.

Automatic reference selection uses the **highest-quality retained frame** from
the preprocessing cache, in both ordinary and geometry processing. Ties choose
the earliest original frame index. The GUI and CLI cache report show this index;
existing caches already contain the required scores and need no new pass.
The reference is loaded before processing the first batch, even if it occurs
later in the capture, and remains the same across checkpoint continuation.
With no cache or cache use disabled, automatic selection uses the first usable
frame. The Capture reference value `0` selects this automatic behaviour. An explicitly
selected nonzero reference that fails screening produces a clear error. Geometry
estimation samples accepted frames across the capture and retains original times.

The [Jupiter reference check](../results/preprocessing/best-reference-validation.json)
selects original frame **1947** (counting from zero). CUDA stacking and CPU surface
processing use it before their first batch. Re-estimating geometry with this
reference still leaves Jupiter's rotation unresolved; reference selection alone
does not supply a reliable spin rate.

Preprocessing runs on CPU using bounded source batches and four scalar
measurements per frame; reconstruction still uses the requested backend. The
GUI shows preprocessing progress without replacing an existing image with an
empty preview. Cancellation raises `InterruptedError` in the Python API and
does not replace a cache or accumulator checkpoint. Resuming reuses validated
cached decisions before continuing the saved sums. Operator versions reject
incompatible older checkpoints.

Result metadata includes `preprocessing`: estimator parameters, mean/standard
deviation/cutoffs, accepted counts and original zero-based rejected indices
grouped by reason. A frame can fail more than one test; reason counts must not
be summed to obtain the unique rejection count. Existing reconstruction rejection
checks (such as maximum registration displacement) can reject additional frames.

The locked R9 scientific ranking in `planetrecon/rank.py` is unchanged.
Unit fixtures that exercise flat fields, raw lattice algebra or cropped textures
explicitly disable whole-planet screening; the dedicated preprocessing tests
exercise independent preprocessing followed by optional cached reconstruction.

## Development validation

The [standalone cache validation](../results/preprocessing/cache-validation.json)
reused the saved Jupiter measurements on the RTX 5070: 3,341 frames retained,
408 excluded (81 quality, 348 shape, 21 overlapping), with identical RGB pixels
and validity to the previous screened reconstruction. The cache implementation
passes the full suite (434 passed, 41 opt-in skips), including separate GUI/CLI
actions, optional reuse, pixel/calibration invalidation, corruption and cancellation.

[Recorded validation](../results/preprocessing/capture-validation.json) includes
410 passing tests and 41 opt-in skips, exact screened checkpoint continuation,
and a 32-frame CPU/CUDA comparison with identical selection and validity
(maximum RGB difference 5.13×10⁻¹¹ ADU).

The full 3,749-frame Jupiter capture retained 3,341 frames on the RTX 5070.
There were 81 low-quality, 178 width and 190 height failures, with overlap;
408 unique frames were rejected. Apparent dimensions averaged 378.69×352.53 px.
Preprocessing plus reconstruction took about 109 seconds. Against the supplied
unsharpened conventional stack, the existing position/gain/offset comparison
gave relative RMSE 0.005319 (previously 0.005627). This is a development
comparison of both filtering and weighting changes, not independent qualification.
Uniform 128-frame samples of each remaining capture also produced usable
silhouettes; neither Mars recording was interpreted as mono. Those sampled
checks do not substitute for full-capture runs.

## Geometry suggestions

Preprocessing also prefills **Globe flattening** from the median retained
silhouette minor/major ratio `q`. For a supplied observer latitude `B`, it uses
`f = 1 - sqrt((q² - sin²(B))/cos²(B))`; this is the oblate projection relation
in [Braga-Ribas et al. (2013), equation 2](https://audreythirouin.wordpress.com/wp-content/uploads/2013/12/bragaribas2013.pdf),
with latitude complementary to their polar aspect angle. Unknown latitude uses
an explicitly labelled equator-on approximation, `f = 1 - q`.

At least 12 retained silhouettes are required. Ring/Saturn models, ratios below
0.75, robust ratio scatter above 0.03, incompatible latitudes and near-pole-on
views remain unresolved. The reported scatter describes frame variation, not a
complete physical uncertainty. Mild phase, limb darkening and seeing can still
bias the illuminated outline. User-entered flattening is preserved, and stale
automatic values clear when a later preprocessing pass cannot estimate it.

Accepted frames also supply three averages aligned to the highest-quality
retained frame (up to 32 frames each, plus the reference if outside those groups,
reduced to at most 256 pixels per side) for a small-angle projected-sphere
motion fit. Translation, depth-dependent surface drift and image roll are fit
jointly with SciPy's robust least-squares solver. Consistent, significant motion
in both time intervals can supply signed surface and field rates and a pole
orientation. Timestamps or a supplied cadence are required for rates per second.
Unknown observer latitude uses an explicitly labelled equator-on approximation.
The pole direction is an image-coordinate convention, not physical north.

A stable mildly oval silhouette supplies an initial pole-axis hypothesis when
surface motion is unresolved; this assumes its major axis is equatorial and
can be biased by phase. Ring/strong-phase captures require a supplied globe
radius for the spherical motion fit. Unconstrained rotation is left unset.

The GUI prefills Geometry for the next run and preserves user edits, including
explicit zeros. New captures clear unchanged automatic values; unresolved new
estimates clear stale automatic rates. Runs with an accumulator checkpoint
retain their settings to preserve resume compatibility. Estimates remain in
result metadata. The active reconstruction configuration is never changed.

A blank surface rate applies **no surface rotation correction**: an unresolved
estimate is not a measured zero or an inferred planetary rotation period. Field,
surface and combined modes retain subpixel translation tracking. They predict
the registration reference at each frame's time with the selected motion model
before estimating the remaining translation, then backproject raw observations
once with the combined transform. The configured centre anchors the reference
frame; `max_shift_px` rejects excessive tracking displacement. Saturn retains
its fixed-centre contract because its moon tracks use detector coordinates.

Zero surface rotation bypasses the spherical round trip; bilinear coordinates
within 10⁻¹⁰ pixels of integer sites are snapped consistently in sampling and its
adjoint to prevent round-off from creating spurious CFA support. Coverage and
validity display levels start at zero so uniform positive support stays visible.

The [Jupiter surface regression](../results/preprocessing/surface-tracking-validation.json)
uses 96 frames distributed through the capture, with 89 retained by its own
preprocessing pass. With the surface rate blank, the corrected surface result
matches the no-motion-model aligned stack within 3×10⁻¹³ ADU, with identical
validity. Relative RMSE against the supplied conventional unsharpened stack
improved from 0.017708 to 0.007212. This bounded development comparison does not
establish Jupiter's rotation rate or replace full-capture qualification.

The synthetic validation covers both spin signs, tilted poles, tracking-only
motion, both image-roll directions and signed viewing latitudes. The local
[Jupiter diagnostic](../results/geometry-discovery/jupiter-estimate.json) gives
an initial silhouette pole angle of 1.91°, but surface motion is unresolved
and no rotation rate is prefilled for that capture.
