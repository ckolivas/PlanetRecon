# PlanetRecon

by Con Kolivas

Planetary multi-frame atmospheric reconstruction for high-frame-rate
monochrome planetary SER/AVI sequences.

This is MOMFBD / short-exposure inverse imaging in an amateur-planetary
regime, tested against lucky-imaging architectures.

The Gate-1 implementation specification is frozen at revision R9 in
`planetary_multi_frame_reconstruction_proposal.md`. Prompt 1 is the
synthetic simulator and its validation tests (Python, NumPy/SciPy, HDF5).
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

Local one-shot-colour RGGB `.ser` files may sit in the repository root for
later real-data tests. They are gitignored. Prompts 1 and 2 do not read them.

## Run

From the repository root (Python 3 with numpy, scipy, h5py, pytest):

```bash
python3 -m planetrecon generate --seed 1001 --dr0 8 --out out
python3 -m planetrecon validate-dev --out out
python3 -m planetrecon freeze-reg --out out
python3 -m planetrecon evaluate --family dev --out out --no-generate
python3 -m planetrecon evaluate --family eval --out out --workers 4 --eval-workers 2
python3 -m planetrecon freeze-prior --out out
python3 -m planetrecon q2 --family dev --out out --no-generate
python3 -m planetrecon q2 --family eval --out out --no-generate --eval-workers 2
python3 -m pytest tests -q
```

`validate-dev` runs the mandatory physics checks and, unless `--no-generate`
is given, writes development-seed HDF5 files for `D/r0 = 8` and `4` at
seeds 1001–1003. Structure-function and numerical-method convergence outcomes
are frozen from those six files; evaluation files retain their own convergence
numbers as diagnostics but inherit only the development pass/fail decisions.

`evaluate --family eval` generates the 12 evaluation seeds in both seeing
regimes if they are missing, then writes classification tables under
`out/prompt2/`. Extension seeds `2013–2024` are only for an inconclusive
12-seed result.

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
  spectral support, solved by FISTA. A1o registers with the known Fourier
  shift, forms the uniform mean stack, uses \(H_{\rm eff}=\mathrm{mean}(H_k)\)
  of the registered OTFs, and the exact stacked white-noise variance
  \(\mathrm{mean}(\sigma_k^2)/|S|\).
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
  uses E2\* assumptions. Known Gate-1 translations register the frames;
  residual tip/tilt stay in the fitted basis. Finite-exposure averaging is
  not modelled.
- **Initialisations.** `zero`: diffraction-limited deconvolution of the
  all-frame mean. `subset`: the same using \(\mathcal S_{10}\). The reported
  D is the better of the two by held-out residual if present, otherwise by
  training residual. Truth \(E_H\) is not used to pick an init.
- **Held-out prediction.** Development family fits a disjoint 10% of frames
  phase-only against an object estimated on the complementary 90%. Evaluation
  closure uses all-frame D.
- **Closure.** \(C=(E_H(\mathrm{A1o})-E_H(D))/(E_H(\mathrm{A1o})-E_H(\mathrm{E2}^*))\).
  Target: median \(C\ge 0.40\) on the feature-rich crop at \(D/r_0=4\)
  (strong G1, moderate seeing). Crops are never shrunk to fit more modes.
