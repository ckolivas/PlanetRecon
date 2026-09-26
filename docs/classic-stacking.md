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

Frames retain their recorded brightness by default: there is no per-frame
gain, mean, histogram or background matching before accumulation. Normalisation
inside the correlation calculation is used only to estimate motion. The output
is the quality-weighted average of the recorded intensities, with coverage
normalisation at detector edges. Only explicitly configured detector calibration
changes those values. Display black/white levels affect the preview separately.

CPU matching is available on every supported machine. CUDA forward-correlation
costs and final backprojection use the existing supported backend and CPU fallback;
reverse checks and field fitting run on CPU. The result records actual diameters,
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
