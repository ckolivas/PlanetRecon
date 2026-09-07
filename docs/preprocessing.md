# Capture preprocessing

The GUI and `stack` command now scan the entire capture before accumulating
any frames. The default Capture checkbox **Preprocess: quality and size (2σ)**
controls this step. For surface-detail crops, flat fields or other data without
a complete visible planet, disable it (CLI: `--no-frame-preselection`; Python:
`ReconstructionConfig(frame_preselection=False)`). No input files are modified.

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

Automatic reference selection starts from an accepted frame. An explicitly
selected nonzero reference that fails screening produces a clear error. Geometry
estimation samples accepted frames across the capture and retains original times.

Preprocessing runs on CPU using bounded source batches and four scalar
measurements per frame; reconstruction still uses the requested backend. The
GUI shows preprocessing progress without replacing an existing image with an
empty preview. Cancellation during screening returns an incomplete result with
zero accumulated frames and does not replace an accumulator checkpoint.
Resuming reruns screening on the original capture before continuing the saved
sums. Operator versions reject incompatible older checkpoints.

Result metadata includes `preprocessing`: estimator parameters, mean/standard
deviation/cutoffs, accepted counts and original zero-based rejected indices
grouped by reason. A frame can fail more than one test; reason counts must not
be summed to obtain the unique rejection count. Existing reconstruction rejection
checks (such as maximum registration displacement) can reject additional frames.

The locked R9 scientific ranking in `planetrecon/rank.py` is unchanged.
Unit fixtures that exercise flat fields, raw lattice algebra or cropped textures
explicitly disable whole-planet screening; the dedicated preprocessing tests
exercise the default two-pass path.

## Development validation

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
