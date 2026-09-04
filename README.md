# PlanetRecon

by Con Kolivas

Planetary multi-frame atmospheric reconstruction for high-frame-rate
monochrome planetary SER/AVI sequences.

This is MOMFBD / short-exposure inverse imaging in an amateur-planetary
regime, tested against lucky-imaging architectures.

The Gate-1 implementation specification is frozen at revision R9 in
`planetary_multi_frame_reconstruction_proposal.md`. Prompt 1 is the
synthetic simulator and its validation tests (Python, NumPy/SciPy, HDF5).
Prompt 2 may begin only after Prompt 1 passes.

## Status

Prompt 1 simulator is implemented. Prompt 2 (estimators / G1–G3) is not.

Local one-shot-colour RGGB `.ser` files may sit in the repository root for
later real-data tests. They are gitignored. Prompt 1 does not read them.

## Run

From the repository root (Python 3 with numpy, scipy, h5py, pytest):

```bash
python3 -m planetrecon generate --seed 1001 --dr0 8 --out out
python3 -m planetrecon validate-dev --out out
python3 -m pytest tests -q
```

`validate-dev` runs the mandatory physics checks and, unless `--no-generate`
is given, writes development-seed HDF5 files for `D/r0 = 8` and `4` at
seeds 1001–1003.

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
