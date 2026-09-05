# PlanetRecon

by Con Kolivas

Planetary multi-frame atmospheric reconstruction. The prototype processes synthetic observations and
monochrome, RGB and raw Bayer planetary SER captures on CPU.

This is MOMFBD / short-exposure inverse imaging in an amateur-planetary
regime, tested against lucky-imaging architectures.

The [R10 evidence review and implementation roadmap](planetary_multi_frame_reconstruction_proposal.md)
retains the frozen R9 Gate-1 experiment and plans the remaining application work
in sections 21–26: CPU/optional GPU execution, Qt6 live preview and progress,
raw-CFA SER reconstruction, field/surface rotation, Saturn's globe and rings,
16-bit PNG and 16/32-bit TIFF export, and standalone Windows/Linux/macOS builds.
Implementation status and remaining qualification work are listed below.
Prompt 1 is the synthetic simulator and its validation tests (Python, NumPy/SciPy, HDF5).
Prompt 2 is the known-transfer estimators, Laplacian ranking, and G1/G2/G3
tables. Q2 is E2b plus all-frame blind D / D-tail and the 40% closure gate.

## Status

Prompt 1 simulator is implemented. Prompt 2 estimators, Laplacian ranking,
and G1/G2/G3 tables are implemented. E2b and Q2 all-frame MFBD (D / D-tail)
are implemented.

On the 12 evaluation seeds, feature-rich crop, fixed \(p=10\):

| Regime | G1 | G2 | G3 |
|---|---|---|---|
| \(D/r_0=4\) | strong (median \(g=10.3\%\)) | negative (median \(g=0.95\%\)) | strong |
| \(D/r_0=8\) | negative | negative | negative |

G1 at \(D/r_0=4\) is **oracle-sensitive**: E2a0 classifies that gap as
inconclusive. Positivity/support in E2a slightly improves the all-frame
reconstruction relative to unconstrained E1/E2a0. Bland-crop \(E_H\) is
high-band ill-conditioned (\(R_H<10^{-5}\)) and does not override the
feature-rich decision. Full tables: `results/prompt2/` and `out/prompt2/`.

Q2 is scoped to moderate seeing (\(D/r_0=4\)), all-frame D / D-tail, because
G1 is strong only there. E2b's production TV weight froze at \(\mu=0\)
(within 2% of the scan minimum), so \(\mathrm{E2}^*=\mathrm{E2b}=\mathrm{E2a}\)
and the result is not prior-limited. Closure tables: `results/q2/`.

| Family | Crop | median \(C\) | vs 0.40 |
|---|---|---:|---|
| development (n=3) | feature-rich | 0.47 | pass |
| evaluation (n=12) | feature-rich | 0.36 | fail |
| evaluation (n=12) | bland | 0.88 | diagnostic only |

The evaluation feature-rich median is short of the 40% gate, so Q3
(\(N=10^3,5\times10^3,2\times10^4\)) does not start. D recovers most of its
gain from known translations placed in pupil tip/tilt; D-tail modes 15→60
change \(E_H\) only weakly. Bland \(C\) can exceed 1 because that crop is
high-band ill-conditioned. A two-initialization held-out-frame diagnostic on
development seed 1001 (feature, 90/10 split) selects the `subset` start and has
holdout/train residual 1.0103; the all-frame closure remains 0.5244.

W00/W01 bounded operator audit is implemented; full scientific qualification
and development-family ranking/gap checks remain outstanding.
The W04–W08 prototype opens a SER or observed HDF5
crop, runs an owned cancellable baseline stack (mono / RGB / raw CFA RGB),
and shows it in a Qt6 raster shell. W09 adds field rotation and a rigid
oblate-globe surface model as composable operators, with geometry-aware
accumulation, coverage masks and degeneracy warnings. W10 adds Saturn as
separate globe and ring layers with near/far occlusion, illumination masks
and moving-moon masking. GPU `auto`/`gpu` probe live CUDA operators
and fall back to CPU when the PyTorch build cannot run the device (Debian
torch 2.6 does not support RTX 5070 / sm_120). Default CPU thread cap is 32.

Review corrections preserve shifted-edge brightness and per-channel CFA
coverage, reject non-finite frames and unsupported colour modes, validate SER
headers/trailers, and retain calibrated units. Progress counts processed frames,
including rejections. Window close cancels and joins its worker; preview traffic
cannot stall the stack. Checkpoints store arrays and metadata atomically in one
NPZ file (schema 1.1); they are snapshots, with no resume implementation yet.
Older split NPZ/JSON checkpoints are rejected. CLI NPZ output includes validity,
units and provenance.

This is a translation-only baseline using integer phase correlation and
normalised CFA backprojection, with unsupported colour samples marked invalid,
plus a W09 geometry-aware baseline for declared field rotation and/or a rigid
oblate globe, and a W10 Saturn layered scene. Field attitude and surface
longitude are separate operators; unseen longitudes are left at zero coverage
rather than filled. Saturn rings are a static equatorial annulus with near/far
occlusion and do not inherit globe spin. CFA parity stays in detector
coordinates under rotation. Freeze-mid-exposure is the default and warns when
limb motion during \(T_{\rm exp}\) is large. The iterative raw-CFA inverse solve, explicit RAM/VRAM
budget enforcement, parent-crash recovery and full W04–W08 acceptance remain
outstanding. Explicit memory-budget settings are rejected until enforcement
exists. The Linux PyInstaller spec is a local packaging spike; clean-system and
cross-platform release acceptance remain planned.

Historical R9 tables are unchanged. Q3 does not start. W02/W03 remain the
next scientific work; W11 adds production MFBD. W13 scientific export is implemented;
W14 connects the Qt capture/geometry/calibration workflow and scientific save controls.
Large-input hardening and later release qualification remain planned.
Advanced atmospheric claims stay gated by W03.

Geometry currently runs on CPU float64, including when Auto/GPU is selected.
Rates in seconds require measured timestamps or an explicit `--cadence`; duplicate
or reversed timestamps are rejected. Without timing, inferred field motion uses
frame indices and is labelled accordingly. `--reference-epoch` is the exact output
time in seconds from the first frame start; exposure midpoints affect observed
poses only. Exposure integration beyond the midpoint approximation is unsupported.
`reference_index` selects the disc-fit anchor, while `reference_epoch_s` selects the
output pose. The current prototype assumes a fixed centre and rigid rates;
tracking drift and surface spin estimation remain unimplemented. Sparse field-angle
estimates can alias large rotations between sampled frames; use a declared rate
for those captures. Moment-based radius estimates are approximate for textured or
limb-darkened discs, so use a measured radius for surface reconstruction.

Saturn reconstruction requires explicit globe and ring radii and a signed
`--sub-obs-lat-deg` opening. The automatic Saturn fit is diagnostic and can supply
an approximate centre; `--center-x` and `--center-y` override it. Ring inclination
sign cannot be recovered from an ellipse alone. Near-edge-on projected ring bands
are masked, including their overlap with the globe. Transparent foreground-ring
and globe mixtures are excluded because the baseline does not solve their separate
radiances. Samples stay within matching globe, ring, background and shadow regions
at the output epoch. The renderer's shadow attenuation is illustrative, not a
calibrated photometric model.

Saturn results retain `layer_coverage` in worker previews/results and checkpoints.
CLI NPZ files contain `layer_coverage__globe` and `layer_coverage__ring` arrays;
these are spatial sample coverage, while `validity` remains per colour for CFA.
Moon positions and velocities use the reference detector axes (+x right, +y down)
and then follow field rotation. A moving moon requires known cadence or timestamps.

Local one-shot-colour RGGB `.ser` files may sit in the repository root for
later real-data tests. They are gitignored. Prompts 1 and 2 do not read them.

## Run

From the repository root (Python 3 with numpy, scipy, h5py, tifffile, pytest). Default
scientific execution is CPU-only, while capture commands expose CPU/auto/GPU
selection. The default per-process thread cap is 32, reduced by
`PLANETRECON_THREADS` or `--threads`; some FFT stages use one thread. Worker counts
are separate limits, not an overall 32-thread budget:

```bash
export PLANETRECON_THREADS=32
export OMP_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32 MKL_NUM_THREADS=32
python3 -m planetrecon evidence --results results
python3 -m planetrecon probe-device --device auto
python3 -m planetrecon stack --path capture.ser --device auto --out out/stack
python3 -m planetrecon stack --path capture.ser --geometry combined \
  --field-rate-deg-s 0.2 --surface-rate-deg-s 0.03 --radius 80 --flattening 0.065 \
  --out out/stack-geo
python3 -m planetrecon stack --path capture.ser --geometry saturn \
  --radius 80 --flattening 0.098 --sub-obs-lat-deg 20 \
  --ring-inner 95 --ring-outer 180 --surface-rate-deg-s 0.03 \
  --out out/stack-saturn
python3 -m planetrecon gui --path capture.ser
python3 -m pytest tests -q
python3 -m pytest tests -q --run-slow
python3 -m planetrecon generate --seed 1001 --dr0 8 --out out
python3 -m planetrecon validate-dev --out out
python3 -m planetrecon freeze-reg --out out
python3 -m planetrecon evaluate --family dev --out out --no-generate
python3 -m planetrecon evaluate --family eval --out out --workers 4 --eval-workers 2
python3 -m planetrecon freeze-prior --out out
python3 -m planetrecon q2 --family dev --out out --no-generate
python3 -m planetrecon q2 --family eval --out out --no-generate --eval-workers 2
```

`pytest tests -q` is the default CPU suite: unit tests and bounded integration.
It does not run `@pytest.mark.slow` convergence jobs, scientific seed families,
or hardware/GPU tests. Pass `--run-slow` / `--run-scientific` / `--run-hardware`
to opt in. Do not put expensive simulations in the default suite.

`python3 -m planetrecon evidence` verifies the archived bundle's checksums and
writes `results/manifests/` without modifying or relabelling archived R9 tables.
Their estimator version is `1.0`; new results use `1.2`. Legacy HDF5 files remain
readable, but absent/stale generation identities or certificates cannot authorize
new Gate tables. They require explicit revalidation/regeneration, not an automatic
stamp from current code. No historical capture or result was regenerated here.

`validate-dev` runs the mandatory physics checks and, unless `--no-generate`
is given, writes development-seed HDF5 files for `D/r0 = 8` and `4` at
seeds 1001–1003. Structure-function and numerical-method convergence outcomes
are frozen from those six files; evaluation files retain their own convergence
numbers as diagnostics but inherit only the development pass/fail decisions.

`evaluate --family eval` generates the 12 evaluation seeds in both seeing
regimes if they are missing, then writes classification tables under
`out/prompt2/`. Extension seeds `2013–2024` are only for an inconclusive
12-seed result.

## Qt capture workflow (W14)

Start `python3 -m planetrecon gui --path capture.ser`, or use **Open capture**.
**Inspect input** reads metadata and a bounded raw preview in an owned process.
CFA inputs show a labelled green proxy; the reconstruction still uses raw samples.
The four settings tabs expose:

- **Capture:** CPU/Auto/GPU, 1–32 CPU threads, batch size, Bayer and byte-order
  overrides, HDF5 crop, rejection, reference frame, cadence and exposure.
- **Calibration:** bias/dark/flat `.npy` tables matching the entire raw frame shape,
  optional gain and saturation threshold, and read-noise metadata. Dark tables
  must already be exposure-scaled. Flat tables must be finite and positive;
  flat correction retains the existing median normalization. No broadcasting is
  allowed. Gain converts ADU to electrons; missing gain retains approximate noise.
  Read noise is recorded, but does not introduce a detector-noise likelihood into
  the baseline. CLI stack exposes `--bias`, `--dark`, `--flat`, `--gain`,
  `--read-noise` and `--saturate` with the same engine behavior.
- **Geometry:** motion model, field/surface rates, fixed centre, globe radius and
  flattening, orientation and reference epoch. Angles/rates are shown in degrees;
  radii and positions stay in pixels. Blank optional values retain engine defaults.
- **Saturn:** ring radii/transmission, illumination and moon masks. Physical globe
  and ring radii plus signed observer latitude remain mandatory for Saturn.

Run uses an owned process. Controls lock while processing; progress distinguishes
unknown pose-estimation work from accumulation and reaches 100% only on completion.
The UI receives at most one unacknowledged full-resolution snapshot at a time,
with a bounded event queue. Result, input, coverage, validity and named Saturn
layer views support channel selection, preview zoom and scrollbar panning. The
preview histogram reports display clipping and sample support; magenta identifies
missing colour support. Coverage is accumulation weight, not calibrated uncertainty.
The display stretch is fitted once at the first result and stays fixed until
**Fit levels** or manual changes. Display operations never modify result arrays.

**Save result** exports the last received full-resolution snapshot through W13,
including while processing continues. Float TIFF is the default. Integer output
requires fixed black/white levels; optional gamma explicitly selects a display
rendering. Existing images require confirmation. Saving runs on a separate encoder
thread, retains its chosen result as newer snapshots arrive, and reports actual
clipped/invalid pixel counts. An error or cancellation preserves the last usable
image, its source identity and save controls. Stale job identities/sequences are
ignored. Cancel processing is nonblocking and terminates only the owned worker
if cooperative cancellation does not finish within the grace period.

Window close joins the processing worker and cancels an active save before its
publication. A save already inside an encoder must return before the window closes;
large or slow disk writes can delay that step. This is bounded-prototype workflow
validation, not a promise for arbitrary capture sizes: RAM/VRAM budgets, checkpoint
resume, parent-crash recovery and all-platform lifecycle qualification remain open.
A bounded local check with 32 busy CPU processes recorded an 82 ms maximum Qt
heartbeat gap; latency depends on hardware, frame size and processing stage.

Run the repeatable GUI/worker/save smoke from source or a frozen executable:

```bash
QT_QPA_PLATFORM=offscreen python3 -m planetrecon gui-smoke --out /tmp/planetrecon-gui-check
QT_QPA_PLATFORM=offscreen dist/planetrecon/planetrecon gui-smoke --out /tmp/planetrecon-gui-check
```

Each invocation uses a fresh directory containing a tiny Bayer SER, float TIFF,
metadata, a window screenshot and `smoke.json`. The test verifies raw-CFA validity
and preserved float values. This local Linux smoke does not certify clean target
systems or Windows/macOS builds. W15 large-input/AVI work is the next independent
application step; W02/W03 scientific qualification remains outstanding.

## Scientific export (W13)

Export uses the floating linear result at full resolution. Choose `png16`,
`tiff16` or `tiff32`; all support mono and RGB. TIFF32 is IEEE float32 and
preserves negative values and values above one without normalizing. Finite
values outside float32 range are rejected. No resampling or sharpening is applied.

```bash
python3 -m planetrecon stack --path capture.ser --device cpu --out out/stack \
  --export png16 --black 0 --white 65535 --checkpoint out/live.npz
python3 -m planetrecon export --path out/stack/stack.npz --out out/linear.tif
python3 -m planetrecon export --path out/live.npz --out out/intermediate.tif \
  --encoding tiff16 --black 0 --white 4095
```

Integer export requires explicit black/white levels in result units, shared by
all channels: clip to that interval, map to 0–65535, then round to nearest with
half values upward. Metadata and CLI output report clipped sample/pixel counts.
There is no automatic percentile or per-channel scaling. Optional
`--display-gamma 2.2` labels an integer export **display-rendered** and applies
power `1/2.2` after mapping. This does not convert camera RGB to sRGB or assign
colour primaries. PNG records the transfer in its gAMA chunk; TIFF records it
in ImageDescription. Linear output is the default.

Invalid, non-finite or zero-coverage samples become NaN in float TIFF and zero
in integer output. The effective validity mask distinguishes invalid zeros from
real black pixels. TIFF pages contain the image, uint8 validity, float64 coverage,
then named spatial layer coverage. PNG has a companion coverage TIFF with the
same mask/coverage pages. Coverage means accumulation weight, not calibrated
uncertainty. RGB validity remains per channel.

The image embeds JSON metadata and names an immutable generation-specific JSON
sidecar (`image.tif.<id>.json`); PNG also names `image.png.<id>.coverage.tif`.
Keep these companions with the image. The sidecar includes an image SHA256.
Metadata retains units, reference epoch, completion state, frame counts,
CFA/source interpretation, calibration identities, geometry, settings and device
precision. Large input identities use size, mtime and hashes of the first/last
64 KiB, explicitly **not** full-file checksums.

Files are staged beside the destination and flushed before publication. The
image is published atomically only after its companions exist. Existing image
paths require `--overwrite`; a failed save leaves the existing image and the
in-memory result intact. Process interruption may leave unreferenced staging or
companion files, and overwrites retain old companions for readers of the previous
generation. These unused files may be removed once no image references them.
Power-loss durability depends on the filesystem; this is not a multi-file disk
transaction.

`--checkpoint` atomically updates a full-resolution NPZ after each batch. Export
can read that snapshot while stacking continues, with `incomplete` and frame
counts recorded. New worker checkpoints are also exportable. Older NPZs without
result metadata are rejected for export rather than assigned guessed units or
completion state. CLI stack snapshots retain the earlier analysis array keys.
W14 GUI save controls retain full-resolution results; Qt's downsampled display is not an export source.

PNG uses the bundled dedicated 16-bit writer and zlib. TIFF uses `tifffile`
without compression, avoiding extra codec runtimes. Run the local Linux package
smoke after building:

```bash
python3 -m PyInstaller packaging/planetrecon-linux.spec
python3 packaging/smoke_export.py dist/planetrecon/planetrecon
```

The smoke checks all six mono/RGB encodings; it does not qualify clean-system,
Windows or macOS releases.

## Numerical conventions (Prompt 1)

- **FFT (optics / convolution).** NumPy default: forward unnormalised,
  inverse `1/N`. A centered optical PSF is formed as
  `fftshift(fft2(ifftshift(P exp(iφ))))`, then sum-normalised to 1.
  Object convolution uses `scipy.signal.fftconvolve(..., mode="same")`.
- **FFT (metric).** Two-dimensional orthonormal FFT (`norm="ortho"`) of
  the Tukey-windowed 128×128 crop. Frequency axes are `fftfreq(n, d=Δθ)`
  with `Δθ = 0.5 λ/D`. Detector Nyquist equals `f_c = D/λ`.
- **OTF storage.** `fft2(ifftshift(psf))` so DC is at `[0, 0]`.
- **Phase screen.** Kolmogorov PSD `0.023 r0^{-5/3} f^{-11/3}` (DC zeroed),
  Fourier synthesis plus four Johansson–Gavel / Schmidt subharmonic
  levels. No finite outer scale. Bilinear extraction, no wrap.
- **Frozen flow.** Wind `v = 5 m/s` along +x. Finite-exposure PSF is the
  average of `J=8` instantaneous **intensity** PSFs at bin midpoints of
  `[t_k, t_k+T_exp]`. Never an averaged phase screen. Doubling to `J=16`
  changes the high-band PSF metric by ≪ 0.5%, so `J=8` is frozen.
- **Paired seeds.** The unit random field depends only on the integer
  seed. Amplitude scales as `(r0_ref/r0)^{5/6}` with `r0_ref = 1 m`.
  Along-wind length is sized for `D/r0 = 4` so both regimes share the
  field. Noise is independent per `(seed, regime, frame)`.
- **Detector.** Optical sampling is `0.125 λ/D` (`L_pupil = 8D`). Detector
  pixels sum 4×4 optical samples (area integration).
- **Flux.** Source rate is locked so the diffraction-limited feature-rich
  crop has mean 800 e⁻/pixel over the reference disk mask at
  `T0 = 0.58875 ms`. Frames scale as `T_exp/T0`. No post-seeing
  renormalisation.
- **Padding.** Default 64 detector pixels around the 80 px-radius disk.
  Crops are taken only after convolution and detector integration.
- **Coordinates.** Array index `[y, x]`; `x` is column / along-wind;
  `y` is row. Crop origins and `shift_xy` are `(x, y)` in detector pixels.

## Numerical conventions (Prompt 2)

- **E1.** Noise-weighted multi-frame Wiener in the unnormalised NumPy FFT
  layout: \(\hat O(f)=\sum_k H_k^* I_k/\sigma_k^2\big/\big(\sum_k|H_k|^2/\sigma_k^2+\lambda(f)\big)\).
  \(\sigma_k^2\) is the spatially averaged Poisson+read variance of frame \(k\).
  Detector MTF is already in the stored finite-exposure OTF.
- **Stabilisation.** \(\lambda(f)=\lambda_{\rm rel}\) (constant), frozen on
  development seeds 1001–1003, both regimes, feature-rich `E1(S_100)`, as the
  smallest value whose median \(E_H\) is within 2% of the scan minimum.
  Frozen value: `E1_LAMBDA_REL = E2A_LAMBDA_REL = 0.03`.
- **E2a0.** Independent conjugate-gradient solve of the same quadratic as E1,
  started without the analytical E1 solution. No positivity. Relative \(E_H\)
  must match E1 to \(<10^{-4}\) and the image norm to \(<10^{-6}\).
- **E2a / A1o.** Same quadratic as E1 with positivity and \(|f|\le f_c\)
  spectral support, solved by FISTA. The feasible set is the intersection of
  those two convex constraints; W01 projects onto it with Dykstra rather than
  clip-then-support-then-clip. A1o registers with the known Fourier shift,
  forms the uniform mean stack, uses \(H_{\rm eff}=\mathrm{mean}(H_k)\) of the
  registered OTFs, and the exact stacked white-noise variance
  \(\mathrm{mean}(\sigma_k^2)/|S|\). Estimator operator version: `1.2`.
  Convergence requires feasibility and a projected-gradient residual, with
  correctly rebased warm-start duals. Iteration exhaustion is not convergence.
  Simulator convolution remains the padded linear operator (version `1.0`);
  crop-FFT circular convolution is the frozen estimator model, audited as an
  a proposed interior approximation; full scientific qualification is pending.
- **Ranking.** Exact Fourier registration, sky-median subtraction if a sky
  mask is present, 4-neighbour Laplacian energy, 2-pixel border ignored.
  Decision subset \(p=10\). Diagnostic grid \(\{5,10,25,50,100\}\).
- **Metric.** Tukey \(\alpha=0.125\), orthonormal FFT, MTF-weighted high band
  \(\mathcal H\). Planted contrast uses the stored 2-σ aperture / 3–5-σ
  annulus on the untapered image. Bland \(R_H<10^{-5}\) is high-band
  ill-conditioned and cannot overturn the feature-rich stop decision.

## Numerical conventions (Q2)

- **E2b.** Same quadratic, positivity, and \(|f|\le f_c\) support as E2a, plus
  a Charbonnier isotropic total-variation prior. The TV weight \(\mu\) is
  frozen on development seeds 1001–1003, both regimes, feature-rich
  `E2b(S_100)`, as the smallest value whose median \(E_H\) is within 2% of
  the scan minimum. Frozen value: `E2B_TV = 0` (TV does not improve \(E_H\)
  enough to pay for an extra prior). A1o does not receive this prior.
- **E2\*.** \(\mathrm{E2b}\) unless the matching oracle gap
  \(E_H(\mathrm{A1o}(\mathcal S_{10}))-E_H(\mathrm{E2}(\mathcal S_{100}))\)
  for E2b is less than half the E2a gap, in which case \(\mathrm{E2}^*=\mathrm{E2a}\)
  and the result is labelled prior-limited. Gate-1 stop logic still uses E2a.
- **D / D-tail.** Snapshot MFBD on the first \(M\in\{15,35,60\}\) QR-Noll
  modes of the obstructed pupil, continuing from 15 to 35 to 60. Object step
  uses E2\* assumptions. Observations stay unregistered: known Gate-1
  translations initialise (and freeze) tip/tilt in the pupil basis so the
  model PSF carries the shift. Higher-order modes are fitted by L-BFGS-B.
  Finite-exposure averaging is not modelled.
- **Initialisations.** Both put known translations into tip/tilt. `zero`
  starts the object from all frames; `subset` starts it from \(\mathcal S_{10}\).
  The reported D is the better of the two by held-out residual if present,
  otherwise by training residual. Truth \(E_H\) is not used to pick an init.
- **Held-out prediction.** Development family fits a disjoint 10% of frames
  phase-only against an object estimated on the complementary 90%. Evaluation
  closure uses all-frame D.
- **Closure.** \(C=(E_H(\mathrm{A1o})-E_H(D))/(E_H(\mathrm{A1o})-E_H(\mathrm{E2}^*))\).
  Target: median \(C\ge 0.40\) on the feature-rich crop at \(D/r_0=4\)
  (strong G1, moderate seeing). Crops are never shrunk to fit more modes.

W15 local hardening: `stack` and the GUI accept AVI 1.0 with a single uncompressed
BI_RGB RGB24 or identity grayscale palette8 video stream. The native decoder is
included in the application; no installed FFmpeg is required. Compressed/YUV,
OpenDML, audio, dropped-frame and Bayer AVI are explicitly unsupported. Row
padding, orientation and BGR storage are decoded exactly. AVI code values have
unknown transfer curves; nominal header cadence is reported but is not used as
measured timing. Set cadence explicitly for geometry only when justified.

AVI indexing uses a temporary disk file (8 bytes/frame). Raw batches are capped
at 64 MiB, with one frame as the minimum working set; reconstruction arrays,
calibration, geometry poses and exports still require additional RAM. SER uses
bounded file reads and detects input disappearance/replacement/truncation.
GUI worker payloads use a private bounded disk spool so killing a worker cannot
leave a partial large pickle in the GUI pipe. Normal close removes the spool;
a parent crash can leave `planetrecon-events-*` in the system temporary directory.
Checkpoint write failures preserve the prior checkpoint and clean temporary
files. Checkpoints are inspectable/exportable results, **not resumable solver
state**. Hard RAM limits and checkpoint resume remain unqualified.
