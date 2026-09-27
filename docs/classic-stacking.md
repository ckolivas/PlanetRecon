# Classic circular multiscale stacking

Set **Sampling multiplier** on the Capture tab to the actual focal ratio divided
by the sensor pixel pitch in micrometres. The default is **5**: a 3.75 µm camera
at f/18.75. At f/26.25, enter 7. For binned or resized input, account for the
change in effective pixel pitch. Settings persist without opening a capture at
startup.

**Alignment wavelength** defaults to 550 nm, appropriate as a green-weighted
approximation for colour alignment. Set the effective filter wavelength for
monochrome captures, including infrared. The displayed minimum circle diameter
updates as either value changes.

Click **Stack** for ordinary stacking with circular multiscale alignment. This
selects Motion model None, cached preprocessing and local alignment. It preserves
the current quality-range/count selection and calibration settings. It validates
or builds preprocessing automatically, then stacks the selected frames. At least
four selected frames are required for the averaged alignment reference. **Run**
and **Batch** use the selected controls; fixed square alignment remains available
for the existing geometry-compensated processing paths.

The starting diameter is eight diffraction PSF FWHMs:

```
FWHM_pixels = 1.028 * wavelength_nm / 1000 * sampling_multiplier
minimum_diameter = odd ceiling of max(15, 8 * FWHM_pixels)
```

The diffraction factor follows the [ESO telescope FWHM model](https://www.eso.org/observing/etc/doc/helpkmos.html).
Eight FWHMs is an engineering starting heuristic, not a measured universal
optimum. At 5× and 550 nm the minimum is 23 pixels; at 7× it is 33 pixels.
The 15-pixel floor protects the correlation fit at unusually low sampling.

The matcher uses up to four diameters, approximately d, sqrt(2)d, 2d and
2sqrt(2)d, rounded up to odd pixels. Centres overlap at half-diameter spacing.
Scales too large for the detector plus the search margin are omitted. If even
the minimum cannot fit, the result explicitly warns that global alignment was
used.

Circular radial-cosine weights suppress the sample edge. A structure tensor
rejects patches that constrain only one motion direction. Coarse scales establish
motion; smaller scales refine it only when correlation is strong, its peak is
unique and curved in both directions, and a reverse match returns to the original
position. Unreliable fine regions retain their coarse estimate; unsupported areas
retain global motion. This decision is made locally for each frame, rather than
assuming a size that worked on one frame remains reliable throughout the capture.

A separate smooth circular contribution footprint combines overlapping samples
into a normalised displacement field. Samples are not independent repeated votes
for an output pixel. Coordinate maps are composed between scales, and a Jacobian
guard rejects a scale that would collapse or fold the field. Only alignment
proxies are warped during fitting; original calibrated detector measurements and
their support are resampled once for accumulation. Existing linear frame-quality
weights and colour support normalisation remain in use. No sharpening is added.

Frames retain their recorded brightness during stacking: there is no per-frame
gain, mean, histogram or background matching before accumulation. Normalisation
inside the correlation calculation is used only to estimate motion. The output
is the quality-weighted average of the recorded intensities, with coverage
normalisation at detector edges. Only explicitly configured detector calibration
changes those values before stacking. Display black/white levels affect the preview separately.

The Capture tab's optional **Normalise brightness** applies one gain to the
output stack. The percentage defaults to **70%** of the capture's full brightness
range: 178.5 ADU for 8-bit data, or 45874.5 ADU for 16-bit data. The brightest
finite supported output sample sets the gain, shared across every pixel and RGB
channel. This retains intrinsic frame brightness, colour ratios, frame weights
and coverage. It applies no gamma, clipping, background subtraction or filtering.
Normalisation is off by default. Automatic result display and default export
levels use zero to full scale so they preserve the chosen percentage.

The CLI equivalents are `--normalise-brightness --normalise-percent 70`.
Gain-calibrated captures use the corresponding full scale in electrons; floating
captures without a known detector range cannot use this option. The applied
gain and original peak are recorded in result metadata. Accumulator checkpoints
remain unscaled, so the percentage can be changed when resuming a stack.

CPU matching is available on every supported machine. Ordinary CPU stacks process
independent frames concurrently, up to **CPU threads** in the Capture tab. Fresh
GUI settings detect available logical CPUs at startup (respecting CPU affinity,
up to the application limit of 32); saved manual choices are retained. The run
status shows the actual frame-worker count. Frames per batch, remaining frames
and a conservative scratch-memory allowance can reduce concurrency. A batch size
below the requested thread count cannot keep all workers occupied.
Quality-rejected frames do not occupy work slots: selected frames are collected
across consecutive input batches, so strict quality cuts can still fill a worker
batch without loading the whole capture.

Workers share immutable reference data, while source reads and accumulation stay
in capture order on the owner thread. This preserves recorded brightness and
bit-for-bit image/coverage results across worker counts. Outstanding work is
bounded by the worker count, numerical libraries use one native thread per worker,
and cancellation joins the workers before returning. Geometry-compensated motion
paths retain their existing execution strategy. More workers need not mean a
linear speedup: patch fitting, memory traffic and serial preparation still limit
scaling.

Circular CUDA alignment retains templates, forward/reverse matching, peak tests,
field blending and coordinate composition on the GPU. The resulting device field
goes directly into raw-data backprojection. CPU preparation still supplies the
filtered proxy, and the CPU owns the ordered final accumulator and checkpoints.
All calculations retain float64; frame brightness is unchanged.

When Triton is available with PyTorch, a fused correlation kernel reads each
candidate directly and reduces it without constructing the large array of all
shifted patches. It supports windows up to 127 pixels; larger windows and builds
without Triton use bounded PyTorch batches on the GPU. Reference data is uploaded
once per run, and no patch-by-patch score downloads are needed. The first fused
run can include kernel compilation; subsequent runs reuse its disk cache. CUDA
allocation or execution failures retain the existing CPU fallback.

On the RTX 5070, a 32-frame 696×404 monochrome Saturn subset (29 retained,
5× sampling, 650 nm) took approximately 6.0 seconds with the earlier CUDA path,
4.4 seconds on the first fused run including compilation, and 1.2 seconds on the
next run. The maximum image difference from the earlier CUDA result was below
6e-13 ADU, with identical coverage. These are small-capture measurements, not a
full-capture throughput guarantee. GPU tests separately compare monochrome, RGB
and Bayer output against CPU and exercise failures during matching/backprojection.

The result records CUDA execution details, actual diameters,
spacing, sampling, wavelength and matching rules in its provenance. Sampling or
method changes do not invalidate the preprocessing cache, but cannot resume an
accumulator made with different settings.

CLI equivalent:

```sh
planetrecon preprocess --path capture.ser
planetrecon stack --path capture.ser --alignment-method circular_multiscale \
  --sampling-multiplier 5 --alignment-wavelength-nm 550 --out result
```

Synthetic tests cover known local distortions, brightness and blur changes,
ambiguous structure, unrelated noise, border support, colour handling and exact
checkpoint resume. These checks establish implementation behaviour; they do not
establish superiority to other planetary stackers or optimal sizing for every
capture.
