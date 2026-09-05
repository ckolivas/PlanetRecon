# Planetary Multi-Frame Atmospheric Reconstruction
## Evidence review and implementation roadmap — revision R10

**Status:** Synthetic prototype implemented; scientific qualification incomplete; desktop application planned

**Revision:** R10 — 2026-09-05

**Code evidence baseline:** `d1b600a` — `Fix Q2 held-out initialization selection`

**Intended outcome:** A sequenced, testable plan for all remaining scientific and application work

**Final application:** CPU- or GPU-operated Qt6 desktop reconstruction of monochrome, RGB and raw Bayer planetary captures, including field rotation, surface rotation and Saturn's rings; standalone Windows, Linux and macOS releases

This is MOMFBD / short-exposure inverse imaging in an amateur-planetary regime, tested against lucky-imaging architectures. It is not a new inverse problem.

R10 retains the R9 synthetic experiment as a reproducible historical baseline.
Sections 0–20 specify that experiment, not a claim that the current code passes
every requirement. Sections 21–26 are the current evidence review, product
contracts, implementation sequence and release criteria; start there for the
next code changes. The user's expanded application requirements explicitly
supersede R9's restriction against planning further product work. They do not
relax the scientific stop rules or turn a failed Q2 evaluation into a pass.

The R9 changes retained from the earlier specification are:

1. pooled 24-seed thresholds are expressed as fractions, not stale 12-seed counts;
2. the no-wrap screen length includes the final exposure interval;
3. low-frequency Kolmogorov content is explicitly tested so an FFT screen cannot accidentally suppress tilt / lucky excursions;
4. source-flux normalisation, convolution padding, and crop extraction are frozen;
5. Fourier metric apodisation and frequency coordinates are frozen;
6. the relationship between E1, E2a0, E2a, and “oracle sensitivity” is explicit;
7. the practical Laplacian ranking is now fully specified;
8. the synthetic Jupiter test object is specified sufficiently for Prompt 1;
9. the HDF5 truth schema is frozen;
10. Prompt 1 is included verbatim;
11. Oval 2 and Oval 3 are repositioned so the locked bland-crop requirements are simultaneously satisfiable.

Documentation revision R10 does **not** change the existing HDF5 `revision="R9"`,
schema version, frozen physics, seed lists or committed result values. Corrected
operators and new colour/geometry experiments require separately versioned
configurations and results, with explicit comparisons to the historical baseline.
Do not relabel old files as newly validated data.

---

# 0. Primary claim and programme logic

A physically constrained estimator that keeps every valid frame and uses per-frame transfer can recover practically more object fidelity **inside telescope+detector support** than a selected, registered stack reconstructed with the **true effective stack PSF**.

Lucky imaging makes two approximations:

1. **selection** — discard frames outside a quality subset;
2. **collapse** — replace per-frame transfers by one effective transfer of the selected stack.

The synthetic gate separates them.

- **G1:** value of frames outside the fixed 10% practical subset when every per-frame transfer is known.
- **G2:** value of retaining per-frame transfer on the same selected 10%.
- **G3:** total known-PSF opportunity relative to the selected collapsed stack.

The synthetic gate asks whether a recoverable opportunity exists. It does not claim victory over a practitioner pipeline on real nights.

## 0.1 Non-claims

- No optical super-resolution beyond pupil+detector support.
- No learned Jupiter prior.
- No claim that one low-mode MFBD run answers the information question.
- No claim that monotone Fisher information is success.
- No claim that the 32″ Gate-1 scene is physically isoplanatic.

## 0.2 Secondary projects are not rescue hatches

- **Q1-B:** spatially varying PSFs across several \(\theta_0\), versus an AP/local baseline.
- **Q1-C:** fixed atmospheric transfer with undersampled detector phases, versus drizzle.

A clean two-regime negative on G1 and G2 terminates the primary atmospheric-transfer programme. B and C remain logically distinct, but are not started merely to avoid a negative primary result.

R10 product scope is separate: usable capture ingestion, registration/stacking,
Qt6 interaction, rotation handling, export and distribution remain required even
if blind atmospheric reconstruction fails its gate. A validated stacking path is
the product fallback, not evidence for the primary atmospheric-transfer claim.
Raw-CFA sampling and full-disc geometry need their own matched controls; do not
pool their outcomes into the original monochrome gate.

---

# 1. Stop rules

```text
Phase 0 simulator
    -> Gate 1 at D/r0 = 8
    -> Gate 1 at D/r0 = 4   [mandatory before termination]
         strong G1               -> Q2 all-frame
         weak G1, strong G2      -> Q2 on selected subset only
         G1 and G2 weak/negative in both regimes
                                  -> STOP primary programme
                                     optional named pivot:
                                     cheap quality/frequency weighting
```

Terminate the primary programme only when all of the following are true on the **feature-rich crop**, practical ranking, fixed \(p=10\):

1. G1 and G2 are weak or negative at both \(D/r_0=8\) and \(D/r_0=4\);
2. the bland crop does not show a contradictory large gap that survives its metric-validity and pathology checks;
3. E1/E2a0 verification passed and E2a is not **oracle-sensitive** under §8.4;
4. simulator, padding, screen, and A1o tests passed;
5. every inconclusive regime/gap received the predeclared extra 12 seeds and remained weak/negative on the pooled 24.

Scoped outcomes:

- strong only at \(D/r_0=8\): scope claim to poor seeing;
- strong only at \(D/r_0=4\): scope claim to moderate seeing;
- strong G2, weak G1: product is “retain per-frame transfer on the lucky subset”;
- feature-rich strong, bland weak: scope Q2 to structured regions.

---

# 2. Forward model

For frame \(k\),

\[
I_k = D\left[\bar h_k * R_k(O)\right] + N_k ,
\]

with finite exposure

\[
\bar h_k =
\frac{1}{T_{\rm exp}}
\int_{t_k}^{t_k+T_{\rm exp}}
h(t)\,dt .
\]

Gate 1 uses:

- one spatially invariant PSF per frame;
- known subpixel translation only;
- no planetary rotation;
- no free warp;
- monochromatic light;
- one frozen-flow Kolmogorov layer.

The PSF derives from the obstructed pupil and moving phase screen.

\[
\tau_0 = 0.314\frac{r_0}{v}.
\]

The independent physical inputs are \(D\), \(D/r_0\), \(v\), \(T_{\rm exp}/\tau_0\), and \(\Delta t/\tau_0\).

Then derive:

\[
r_0 = D/(D/r_0),
\]

\[
\tau_0=0.314r_0/v,
\]

\[
T_{\rm exp}=0.30\tau_0,
\qquad
\Delta t=\tau_0.
\]

Do not independently assign \(r_0\), \(v\), \(\tau_0\), exposure, and cadence.

## 2.1 Frozen-flow screen trajectory

Let a two-dimensional phase screen move at velocity \(\mathbf v\).

For sub-exposure sample \(j\) of frame \(k\),

\[
\phi_{k,j}(x,y)
=
\phi_{\rm screen}
\left(x-v_x t_{k,j},\,y-v_y t_{k,j}\right).
\]

The finite-exposure PSF is the average of instantaneous **intensity PSFs**:

\[
\bar h_k =
\frac{1}{J}
\sum_{j=1}^{J} h(\phi_{k,j}),
\]

not the PSF of an averaged phase screen.

The exposure integration sample count \(J\) is adaptive:

- start with \(J=8\);
- double to 16 on development seeds;
- require the resulting \(E_H\) change \(<0.5\%\);
- if not, continue doubling until the tolerance passes.

## 2.2 No-wrap condition

For \(N\) frames, the along-wind phase-screen extent must cover at least

\[
L_x
\ge
v\left[(N-1)\Delta t + T_{\rm exp}\right]
+
D
+
2m ,
\]

where \(m\) is the extraction/interpolation margin.

The final exposure term \(vT_{\rm exp}\) is mandatory.

For the locked \(N=500\) core before margin:

- \(D/r_0=8:\quad L_x \gtrsim 5.1494\ {\rm m}\);
- \(D/r_0=4:\quad L_x \gtrsim 10.0488\ {\rm m}\).

The cross-wind extent must be at least \(D+2m\).

No periodic screen coordinate may be revisited during a sequence.

---

# 3. Kolmogorov screen generation and convergence

The core remains **Kolmogorov**, not finite-\(L_0\) von Kármán.

A finite FFT screen can suppress low-frequency power if implemented naively. That would bias:

- tip/tilt statistics;
- Strehl distribution;
- lucky-frame tails;
- temporal correlation.

The simulator shall therefore use either:

1. Fourier synthesis with low-frequency subharmonic augmentation; or
2. another generator shown by tests to reproduce the same pupil-scale statistics.

The implementation is judged by tests, not by the label on the algorithm.

## 3.1 Required screen tests

On development seeds:

### Structure function

For pupil-relevant separations,

\[
D_\phi(\rho)
\propto
\left(\rho/r_0\right)^{5/3}.
\]

Compare the measured structure function with the intended Kolmogorov law over a declared range excluding:

- sub-pixel separations dominated by discretisation;
- separations approaching the finite screen dimension.

Because \(r_0\) defines an ensemble statistic, the mandatory pass/fail comparison
uses the mean structure-function ratio across the three independent development
seed screens. Paired \(D/r_0\) regimes are not double-counted. Each HDF5 file
retains its individual curve and local diagnostic, then receives the pooled
suite result before it may be marked Gate-eligible.

### Aperture tilt / centroid statistics

The implementation must not systematically lose low-frequency tip/tilt.

Record the ensemble centroid variance of instantaneous PSFs and verify convergence when:

- base screen sampling is doubled (relative centroid change below 5%);
- the number of subharmonic levels is increased (relative centroid-variance change below 10%).

### Gate-metric convergence

Doubling optical/screen resolution changes development-seed \(E_H\) by \(<2\%\).

Increasing low-frequency augmentation by one level changes development-seed:

- Strehl distribution summary by \(<2\%\);
- practical-ranking top-10% membership overlap by less than a predeclared tolerance or, if membership is unstable, changes G1/G2 by \(<2\%\).

### Inference mismatch

Projection onto the first 60 inference KL modes must leave non-zero residual phase with documented RMS.

The simulator and future solver must not share an artificially closed low-order subspace.

---

# 4. Locked physical parameters

## 4.1 Shared geometry and detector

| Parameter | Locked value |
|---|---:|
| Telescope diameter \(D\) | 250 mm |
| Central obstruction | 0.30 of diameter |
| Spiders | off |
| Wavelength | 610 nm monochromatic |
| Frozen-flow speed \(v\) | 5 m/s |
| Detector sampling | \(0.5\,\lambda/D\) rad/pixel |
| Angular pixel scale | \(\approx0.252''/{\rm px}\) |
| Evaluation crop | 128×128 detector pixels |
| Frames \(N\) | 500 |
| \(T_{\rm exp}/\tau_0\) | 0.30 |
| \(\Delta t/\tau_0\) | 1.0 |
| Read noise | 2 e⁻ rms |
| Evaluation seeds | 12 |
| Extension seeds if inconclusive | 12 additional |
| Development seeds | 3, disjoint |

## 4.2 Mandatory seeing regimes

| Quantity | Poor seeing | Moderate seeing |
|---|---:|---:|
| \(D/r_0\) | 8 | 4 |
| \(r_0\) | 31.25 mm | 62.5 mm |
| \(\tau_0\) | 1.9625 ms | 3.9250 ms |
| \(T_{\rm exp}\) | 0.58875 ms | 1.17750 ms |
| \(\Delta t\) | 1.9625 ms | 3.9250 ms |
| frame cadence | 509.55 fps | 254.78 fps |
| mean reference signal | 800 e⁻/px/frame | 1600 e⁻/px/frame |
| nominal seeing FWHM \(0.98\lambda/r_0\) | \(\approx3.95''\) | \(\approx1.97''\) |

The second regime receives twice the electrons because its exposure is twice as long at the same source photon rate.

No cross-regime claim treats the two runs as equal-photon experiments.

## 4.3 Optional arms after both mandatory tables

- \(D/r_0=16\);
- \(T_{\rm exp}/\tau_0=1\), with photon count scaled by exposure;
- \(\Delta t/\tau_0=4\);
- von Kármán \(L_0=25\) m;
- finite bandwidth;
- spiders.

No full factorial.

---

# 5. Synthetic object and crop definition

Prompt 1 needs a deterministic object. It does not need a photorealistic Jupiter.

The object is designed to contain:

- a hard planetary limb;
- broad belt/zone structure;
- moderate gradients;
- compact planted features with measurable in-band high-frequency content;
- a separate bland region.

## 5.1 Object grid

Generate the latent object on a grid at **4× detector linear sampling**.

Reference detector-pixel units are used below.

Use a full synthetic planet scene large enough to contain a disk of:

\[
R_{\rm planet}=80\ {\rm detector\ pixels}.
\]

This corresponds to a diameter of \(\approx40.3''\).

The object is generated on a padded high-resolution canvas and only later blurred, detector-integrated, and cropped.

## 5.2 Limb darkening

Inside the disk,

\[
\mu=\sqrt{1-(r/R_{\rm planet})^2}.
\]

Use the locked linear law

\[
L(\mu)=1-u(1-\mu),
\qquad
u=0.45.
\]

Outside the disk the object is zero.

## 5.3 Broad belt/zone component

Let \(y_n=y/R_{\rm planet}\).

Use a smooth multiplicative latitude modulation

\[
B(y_n)
=
1
+
0.07\cos(3\pi y_n)
-
0.04\cos(7\pi y_n)
+
0.025\sin(11\pi y_n).
\]

The unplanted object is

\[
O_{\rm base}(x,y)=L(\mu)\,B(y_n)
\]

inside the disk, clipped only if necessary to remain positive.

The purpose is reproducible spatial structure, not a physical atmospheric model of Jupiter.

## 5.4 Planted ovals

Add three multiplicative elliptical Gaussian features:

\[
O \leftarrow O
\left[
1+c_i
\exp\left(
-\frac12
\left(
x_i'^2/\sigma_{x,i}^2+
y_i'^2/\sigma_{y,i}^2
\right)
\right)
\right].
\]

Locked detector-pixel-scale parameters:

| Oval | Centre \((x,y)\) relative to disk centre | \(\sigma_x\) | \(\sigma_y\) | angle | peak contrast \(c_i\) |
|---|---|---:|---:|---:|---:|
| 1 | \((+45,-12)\) px | 2.0 px | 3.5 px | +20° | -0.16 |
| 2 | \((+40,+18)\) px | 2.5 px | 4.0 px | -15° | +0.12 |
| 3 | \((-25,-40)\) px | 1.8 px | 3.0 px | +35° | -0.12 |

All features must remain inside the disk.

Before evaluation, automatically verify that Oval 1 and at least one other oval contribute non-zero truth power to the conditioned high band \(\mathcal H\).

If they do not, Prompt 1 fails rather than silently changing the ovals.

R9 changes only the centres of Oval 2 and Oval 3 from R8. The R8 centres made
the bland-crop constraints mathematically incompatible with an 80-pixel-radius
disk and a central 96×96 interior region; no axis-aligned crop origin existed.

## 5.5 Evaluation crops

Derive two 128×128 detector-pixel crops from the same full scene.

### Feature-rich crop

Choose a fixed crop that includes:

- the positive-\(x\) planetary limb;
- Oval 1;
- at least one belt boundary.

The limb must lie at least 16 detector pixels from the crop boundary.

### Bland crop

Choose a fixed interior crop:

- no planted oval centre inside the central 96×96 evaluation region;
- no planetary limb inside that central 96×96 region;
- broad belt/zone structure only.

Record the exact crop origins in the HDF5 metadata.

The locked R9 bland-crop origin relative to the disk-centred detector canvas is
\((-72,-54)\) detector pixels. Its central 96×96 region contains no planted
oval centre and no planetary limb.

## 5.6 Padding

Atmospheric convolution is never performed directly on an isolated 128×128 crop.

Generate and convolve a larger padded scene, then detector-integrate and extract the evaluation crops.

Padding is considered sufficient when doubling padding changes development-seed \(E_H\) by \(<0.5\%\).

This prevents circular convolution / cropped-PSF leakage from becoming a fake high-frequency result.

---

# 6. Photon normalisation and noise

Define a **source electron rate**, not an arbitrary per-frame post-seeing normalisation.

Let the poor-seeing core exposure be

\[
T_0 = 0.58875\ {\rm ms}.
\]

Scale the latent object so that, after diffraction-limited telescope+detector integration but before atmospheric blur/noise, the mean expected signal over a fixed **reference disk mask** in the feature-rich crop is:

\[
800\ {\rm e^- / pixel}
\]

for exposure \(T_0\).

The reference mask:

- includes only disk pixels;
- excludes a 4-pixel detector-space band around the planetary limb;
- is stored in the truth file.

The resulting electron **rate** is then fixed for every seeing/exposure arm.

Therefore:

\[
\mu_{\rm e^-}\propto T_{\rm exp}.
\]

Do not renormalise each atmospheric frame after convolution.

Noise generation:

1. Poisson draw from expected electrons per detector pixel;
2. additive Gaussian read noise, \(\sigma_{\rm read}=2\ {\rm e^-}\);
3. store floating-point electron values for Gate 1.

No 8-bit quantisation in the mandatory core.

---

# 7. Registration and practical ranking

Gate 1 knows the injected subpixel translations exactly.

All estimators and rankings use the same registered coordinate system.

Do not allow the registration problem to contaminate Gate 1.

## 7.1 Locked practical quality proxy

The decision ranking is **Laplacian energy**.

After exact registration:

1. use the observed noisy 128×128 crop;
2. subtract the median value of pixels in the known sky mask, if a sky region is present;
3. convolve with

\[
K=
\begin{bmatrix}
0&1&0\\
1&-4&1\\
0&1&0
\end{bmatrix};
\]

4. ignore a 2-pixel image border;
5. score

\[
Q_k=\frac{1}{N_{\rm valid}}\sum (\nabla^2 I_k)^2.
\]

No denoising and no pre-sharpening.

Higher \(Q_k\) ranks better.

The practical top 10% is locked before any reconstruction metric is inspected.

## 7.2 Diagnostic rankings

Report separately:

- true instantaneous/exposure-averaged Strehl proxy;
- RMS pupil phase;
- mean \(|H_k|\) over conditioned high-band frequencies.

These do not vote on the stop decision.

## 7.3 Percentiles

Decision subset:

\[
p=10.
\]

Diagnostic grid:

\[
p\in\{5,10,25,50,100\}.
\]

Also report:

\[
p_E^\star
=
\arg\min_{p<100}
E_H(\mathrm{E2a}(\mathcal S_p)),
\]

\[
p_A^\star
=
\arg\min_{p<100}
E_H(\mathrm{A1o}(\mathcal S_p)).
\]

Neither changes the stop percentile.

---

# 8. Estimators and oracle controls

## 8.1 E1

Frequency-domain multi-frame quadratic/Wiener reference using true per-frame transfer.

Detector MTF is part of the forward transfer.

E1 is implemented only for assumptions under which the analytical expression is valid.

## 8.2 E2a0

Numerical implementation of **exactly the same objective as E1**.

Purpose: software verification.

On a matched deterministic test:

\[
\frac{|E_H({\rm E2a0})-E_H({\rm E1})|}
{\max(E_H({\rm E1}),\epsilon)}
<10^{-4}.
\]

A separate numerical image-norm check is also stored.

Do not compare E1 and E2a0 after adding positivity, Poisson likelihood, or other changed assumptions.

## 8.3 E2a

Known per-frame transfer, detector-aware numerical reconstruction with:

- positivity;
- known physical spectral support;
- minimal fixed quadratic stabilisation selected on development seeds only.

This is the primary Gate-1 oracle.

## 8.4 Oracle-sensitivity rule

E2a is **oracle-sensitive** if E2a0 and E2a would lead to different qualitative G1/G2 classifications, or if

\[
|g_{\rm E2a}-g_{\rm E2a0}| > 0.5\,|g_{\rm E2a0}|
\]

for a gap whose E2a0 magnitude is at least 5%.

In that case:

- do not issue a clean negative;
- inspect only development seeds;
- identify whether positivity/support/stabilisation causes the difference;
- freeze any correction before reopening evaluation seeds.

This is not a new scientific branch. It is an oracle-validity check.

## 8.5 E2b

E2a plus the production object prior.

The prior is selected and locked on the three development seeds before evaluation tables are generated.

If an E2b oracle gap is less than half the corresponding E2a gap, label that result **prior-limited**.

E2b does not replace E2a for Gate-1 stop logic.

## 8.6 A1o

For subset \(\mathcal S\):

1. exactly register every selected observation;
2. compute the uniform mean stack;
3. propagate the exact stacked noise model;
4. use the true effective transfer

\[
H_{\rm eff}(f)
=
\frac{1}{|\mathcal S|}
\sum_{k\in\mathcal S} H_k(f);
\]

5. reconstruct using the **same object assumptions as E2a**.

A1o is deliberately stronger than a practical stack: it knows its exact effective transfer.

Do not add per-frame sufficient statistics to A1o; that would recreate E1/E2a and erase the quantity G2 is intended to measure.

## 8.7 Optional practical estimators

- **A1p:** same selected stack but estimated effective PSF and documented RL/Wiener.
- **A2:** practitioner pipeline on real SER; synthetic only if export is demonstrably linear and reproducible.

No undocumented wavelets on quantitative outputs.

---

# 9. Fourier metric definition

## 9.1 Frequency coordinates

Use the 128×128 detector-sampled evaluation image.

Use a two-dimensional **orthonormal FFT** (`norm="ortho"` or mathematically equivalent).

Define radial spatial frequency relative to the telescope cutoff:

\[
\rho = |f|/f_c,
\qquad
f_c=D/\lambda.
\]

At the locked detector sampling, detector Nyquist coincides with \(f_c\).

## 9.2 Boundary apodisation

Raw cropped FFTs can create artificial high-frequency power from crop boundaries.

Before computing \(E_H\):

- multiply both \(O\) and \(\hat O\) by the **same fixed 2D Tukey window**;
- Tukey taper fraction: \(\alpha=0.125\);
- the window is generated once and stored;
- planted-feature contrast is measured on the untapered image.

The feature-rich crop must place the limb and planted Oval 1 outside the taper region.

## 9.3 Ideal telescope+detector transfer

Compute the ideal monochromatic telescope transfer from the actual obscured pupil used by the simulator.

Multiply by the detector pixel-aperture MTF.

Do not substitute an unobscured analytic MTF if the simulator uses an obstruction.

Define:

\[
\mathcal H
=
\left\{
f:
0.5 f_c\le |f|\le f_c,\quad
\mathrm{MTF}_{\rm tel+det}(f)\ge0.05
\right\}.
\]

## 9.4 Primary metric

\[
E_H(\hat O)
=
\frac{
\sum_{f\in\mathcal H}
\mathrm{MTF}_{\rm tel+det}^2(f)
\,
|\widehat{w\hat O}(f)-\widehat{wO}(f)|^2
}{
\sum_{f\in\mathcal H}
\mathrm{MTF}_{\rm tel+det}^2(f)
\,
|\widehat{wO}(f)|^2
}.
\]

Lower is better.

Use the same photometric scaling and geometric frame for every estimator.

## 9.5 Bland-crop validity

A bland crop can have very little truth power in \(\mathcal H\).

Define

\[
R_H
=
\frac{
\sum_{f\in\mathcal H}
\mathrm{MTF}^2 |\widehat{wO}|^2
}{
\sum_{f}
|\widehat{wO}|^2
}.
\]

If \(R_H < 10^{-5}\):

- mark bland-crop \(E_H\) as **high-band ill-conditioned**;
- do not let it contradict the feature-rich stop decision;
- still report mid-band and image-domain errors.

This prevents division by an almost empty high-band truth signal from manufacturing a dramatic gap.

## 9.6 Planted-feature consistency

For each planted oval wholly contained in the feature-rich metric region:

1. use a locked feature aperture;
2. estimate local background from a surrounding annulus excluding belt edges;
3. compute recovered signed contrast relative to truth.

The mandatory consistency statistic is Oval 1 relative contrast error.

A nominally strong G1/G2 that worsens Oval-1 contrast relative error by \(>5\%\) is **pathological**, not strong.

---

# 10. Gap definitions

All \(E_H\) values are lower-is-better.

Decision subset:

\[
\mathcal S_{10}.
\]

## 10.1 G1 — tail-frame opportunity

\[
G1
=
E_H(\mathrm{E2a}(\mathcal S_{10}))
-
E_H(\mathrm{E2a}(\mathcal S_{100})),
\]

\[
g1
=
\frac{G1}
{E_H(\mathrm{E2a}(\mathcal S_{10}))}.
\]

## 10.2 G2 — per-frame-transfer opportunity on selected data

\[
G2
=
E_H(\mathrm{A1o}(\mathcal S_{10}))
-
E_H(\mathrm{E2a}(\mathcal S_{10})),
\]

\[
g2
=
\frac{G2}
{E_H(\mathrm{A1o}(\mathcal S_{10}))}.
\]

## 10.3 G3 — total mechanism opportunity

\[
G3_m
=
E_H(\mathrm{A1o}(\mathcal S_{10}))
-
E_H(\mathrm{E2a}(\mathcal S_{100})).
\]

Require algebraically and numerically:

\[
G3_m=G1+G2.
\]

## 10.4 Diagnostic companions

Report the same quantities at \(p_E^\star\) and \(p_A^\star\), plus

\[
G3_{\rm best}
=
E_H(\mathrm{A1o}(\mathcal S_{p_A^\star}))
-
E_H(\mathrm{E2a}(\mathcal S_{100})).
\]

Practical companion, when available:

\[
G3_{\rm practical}
=
E_H(\text{practical lucky})
-
E_H(\mathrm{E2a}(\mathcal S_{100})).
\]

---

# 11. Decision thresholds and seeds

Classification is per gap, per mandatory seeing regime, using the feature-rich crop.

## 11.1 Strong

A gap is **strong** if:

1. median \(g\ge10\%\);
2. at least \(2/3\) of seeds have \(g>0\);
3. Oval-1 consistency passes.

For 12 seeds:

- at least 8/12 positive.

For pooled 24 seeds:

- at least 16/24 positive.

## 11.2 Negative

A gap is **negative** if:

1. median \(g<5\%\); and
2. no more than \(1/4\) of seeds reach \(g\ge10\%\).

For 12 seeds:

- at most 3/12 reach 10%.

For pooled 24 seeds:

- at most 6/24 reach 10%.

## 11.3 Inconclusive

Anything else.

For an inconclusive 12-seed result:

- run the predeclared 12 extension seeds;
- pool to 24;
- apply the same **fractional** rules above;
- do not edit thresholds.

## 11.4 Required reporting

For every gap/regime/crop report:

- median;
- 16th–84th percentile;
- fraction \(g>0\);
- fraction \(g\ge10\%\);
- worst seed;
- Oval-1 contrast consistency.

Resource curves are companions only.

---

# 12. Resource curves

For each estimator plot the primary fidelity metric against:

1. frames **captured**;
2. frames **used**;
3. expected detected photons used.

Lucky methods may capture \(N\) but use only \(pN\).

This distinguishes:

- “use the SER already recorded more completely”;
- “reach a target fidelity with less observing resource.”

These curves do not change Gate-1 stop logic.

---

# 13. HDF5 truth schema — version 1

Prompt 1 writes one HDF5 file per seed/regime.

Recommended filename:

```text
gate1_Dr0-8_seed-02001.h5
```

Root attributes:

```text
schema_name = "planetary-mfbd-gate1"
schema_version = "1.0"
revision = "R9"
seed = int
```

## 13.1 `/config`

Store scalar/string configuration as attributes or datasets:

```text
D_m
obstruction_ratio
wavelength_m
wind_m_s
Dr0
r0_m
tau0_s
texp_s
dt_s
N_frames
detector_pixel_scale_rad
detector_pixel_scale_arcsec
read_noise_e
source_rate_scale
pupil_grid_size
screen_dx_m
screen_shape
subharmonic_levels
exposure_samples_J
padding_detector_px
```

Also store the complete configuration as canonical JSON:

```text
/config/json_utf8
```

## 13.2 `/object`

```text
/object/full_latent_4x          float32
/object/full_latent_detector    float32
/object/reference_disk_mask     uint8
/object/feature_crop_origin     int32[2]
/object/bland_crop_origin       int32[2]
/object/feature_truth           float32[128,128]
/object/bland_truth             float32[128,128]
/object/eval_window             float32[128,128]
/object/ideal_mtf               float32[128,128]
/object/high_band_mask          uint8[128,128]
```

Store planted-feature table under:

```text
/object/features/*
```

including centres, sigmas, angle, contrast, and measurement apertures.

The locked measurement aperture is the elliptical two-sigma region of each
oval. The local-background annulus spans elliptical radii from three to five
sigma. Store the clipped feature-crop aperture and annulus masks for every oval.

```text
/object/features/<oval>/feature_crop_aperture_mask  uint8[128,128]
/object/features/<oval>/feature_crop_annulus_mask   uint8[128,128]
```

## 13.3 `/pupil`

```text
/pupil/amplitude                float32
/pupil/frequency_x
/pupil/frequency_y
```

## 13.4 `/atmosphere`

Do not require storage of the entire multi-metre high-resolution screen if it makes files unreasonable.

Required:

```text
/atmosphere/frame_time_s        float64[N]
/atmosphere/phase_rms_rad       float32[N]
/atmosphere/strehl_proxy        float32[N]
/atmosphere/kl60_coeff          float32[N,60]
/atmosphere/kl60_residual_rms   float32[N]
```

Optional/debug:

```text
/atmosphere/screen_or_reference
```

The generator configuration and seed must be sufficient to regenerate the screen exactly.

## 13.5 `/truth_transfer`

For each crop:

```text
/truth_transfer/feature/psf_bar   float32[N,H,W]
/truth_transfer/feature/otf_bar   complex64[N,H,W]
/truth_transfer/bland/psf_bar
/truth_transfer/bland/otf_bar
```

If both crops share exactly the same spatially invariant transfer, store one copy and hard-link the second path.

## 13.6 `/frames`

```text
/frames/feature_expected_e      float32[N,128,128]
/frames/feature_observed_e      float32[N,128,128]
/frames/bland_expected_e
/frames/bland_observed_e
/frames/shift_xy_detector_px    float32[N,2]
```

Chunk by frame.

Compression is allowed only if lossless.

## 13.7 `/validation`

Prompt 1 writes simulator checks:

```text
/validation/structure_function_rho
/validation/structure_function_measured
/validation/structure_function_target
/validation/psf_energy_error
/validation/grid_convergence
/validation/exposure_convergence
/validation/lowfreq_convergence
/validation/no_wrap_pass
/validation/kl60_residual_summary
/validation/padding_convergence
/validation/*_pass
/validation/gate_eligible
```

A newly generated file is provisional. The development-suite command applies
the pooled structure-function result, and a file is **Gate-eligible** only if
every mandatory validation value is finite and every mandatory flag passes.

---

# 14. Deterministic seeds

Lock seed families to make evaluation reproducible.

Suggested seed IDs:

```text
development: 1001, 1002, 1003

evaluation:
2001..2012

extension if inconclusive:
2013..2024
```

The same seed ID in \(D/r_0=8\) and \(D/r_0=4\) may use the same underlying unit random stream rescaled to the requested \(r_0\), but the implementation must document whether regimes are paired or independent.

**R9 default:** use paired base random streams across the two mandatory seeing regimes. This reduces irrelevant Monte-Carlo differences when comparing regime dependence.

Noise draws remain deterministic functions of `(seed, regime, frame_index)` and are independent between regimes after signal scaling.

---

# 15. Must-pass tests before Gate-1 tables

## Optics

- PSF integral = 1 within tolerance.
- Obscured diffraction-limited PSF agrees with direct pupil FFT reference.
- Tip/tilt shifts PSF in the expected direction.
- No pupil wrap / screen reuse.

## Atmosphere

- Kolmogorov structure function passes.
- Low-frequency / tilt statistics converge.
- Screen-grid doubling changes development \(E_H<2\%\).
- Exposure sample doubling changes development \(E_H<0.5\%\).
- first 60 KL modes leave residual phase.
- no-wrap length includes final exposure.

## Timing and flux

- \(\tau_0=0.314r_0/v\).
- \(\Delta x=v\Delta t\).
- \(T_{\rm exp}\to0\) approaches instantaneous PSF.
- expected photons scale linearly with \(T_{\rm exp}\).
- total flux is conserved apart from documented crop/padding leakage.

## Object / padding

- object generator deterministic.
- all planted features lie inside disk.
- feature-rich crop contains required limb/feature geometry.
- padding doubling changes \(E_H<0.5\%\).
- Oval 1 has non-zero conditioned-high-band truth power.

## Metrics

- FFT frequency grid reaches \(f_c\) at detector Nyquist as expected.
- high-band mask uses actual obstructed-pupil+detector MTF.
- window is identical for truth and reconstructions.
- bland high-band validity flag works.

## Estimators

- E2a0 matches E1 under matched assumptions.
- A1o \(H_{\rm eff}=\mathrm{mean}(H_k)\) numerically.
- A1o uses exact stacked noise variance/covariance as represented by the chosen reconstruction formulation.
- \(G3_m=G1+G2\) to numerical tolerance.

## Reproducibility

- development and evaluation seeds disjoint.
- ranking code/config hash stored before evaluation metrics.
- E2a regularisation frozen from development seeds before evaluation.
- all file-local checks pass on every development seed and regime;
- pooled ensemble checks pass across the three independent development seeds;
- all Gate-eligible HDF5 files contain the same schema version.

---

# 16. Prompt 1 — frozen Codex handoff

The following is the R9 implementation prompt.

> **Project:** Planetary Multi-Frame Atmospheric Reconstruction — Gate-1 Simulator, R9.
>
> Implement only the synthetic simulator and its validation tests. Do not implement blind deconvolution, MFBD, ranking, G1/G2/G3, tiling, planetary rotation, Bayer reconstruction, or a user interface.
>
> Use Python with NumPy/SciPy and HDF5 (`h5py`) for the reference implementation. GPU support is not required in Prompt 1. Structure the optics code so a later PyTorch/CUDA implementation can reproduce the same arrays and conventions.
>
> Implement the R9 locked physics:
>
> 1. 250 mm circular pupil, 0.30 central obstruction, no spiders, 610 nm monochromatic.
> 2. Single-layer frozen-flow Kolmogorov turbulence with \(v=5\) m/s.
> 3. Mandatory regimes \(D/r_0=8\) and \(4\), with \(r_0=D/(D/r_0)\) and \(\tau_0=0.314r_0/v\).
> 4. \(T_{\rm exp}=0.30\tau_0\), \(\Delta t=\tau_0\), \(N=500\).
> 5. A finite exposure is the average of instantaneous **intensity PSFs** sampled along the same moving phase screen; never average independent phase screens.
> 6. The phase screen must not wrap during the 500-frame sequence. Required along-wind extent includes \((N-1)\Delta t+T_{\rm exp}\).
> 7. Use Kolmogorov screen generation with low-frequency fidelity sufficient to pass the R9 structure-function and tilt/centroid convergence tests. Fourier synthesis plus subharmonics is acceptable.
> 8. Generate the deterministic R9 synthetic Jupiter-like latent object, including limb darkening, belts, and the three planted ovals.
> 9. Simulate on a padded scene and crop only after optical convolution and detector integration. Determine padding by the R9 convergence rule.
> 10. Detector sampling is \(0.5\lambda/D\) radians per pixel. Detector pixels integrate irradiance over their area.
> 11. Set the source electron rate using the R9 reference-mask rule so the poor-seeing core exposure has 800 e⁻/pixel mean before atmospheric blur/noise. The moderate-seeing run must therefore receive 1600 e⁻/pixel at twice the exposure.
> 12. Add Poisson shot noise and 2 e⁻ rms Gaussian read noise.
> 13. Generate feature-rich and bland 128×128 crops from the same deterministic latent planet.
> 14. Store true finite-exposure PSFs/OTFs, timestamps, expected electron images, observed electron images, exact shifts, relevant atmospheric summaries, object truth, pupil, MTF/high-band mask, and validation results in the R9 HDF5 schema.
> 15. Use development seeds 1001–1003 initially. Add evaluation seed generation capability but do not generate Gate tables.
>
> Implement automated tests for:
>
> - PSF energy conservation;
> - obscured diffraction-limited pupil/PSF;
> - Kolmogorov structure function;
> - low-frequency/tilt convergence;
> - grid-doubling convergence;
> - finite-exposure integration convergence;
> - \(T_{\rm exp}\to0\) instantaneous-PSF limit;
> - exact timing relations;
> - no phase-screen wrap including the final exposure;
> - photon scaling with exposure;
> - padding convergence;
> - deterministic object/crop geometry;
> - first-60-KL residual phase;
> - HDF5 schema validation.
>
> The simulator is complete only when all mandatory tests pass on all three development seeds for both \(D/r_0=8\) and \(D/r_0=4\).
>
> Produce:
>
> - source code;
> - unit tests;
> - a command-line entry point to generate one seed/regime;
> - a command-line entry point to run the development validation suite;
> - one human-readable validation summary per generated HDF5 file;
> - a short README documenting numerical conventions, FFT normalisation, phase-screen method, padding choice, and reproducibility.
>
> Do not optimise performance beyond what is needed for the development seeds. Correctness and reproducibility are the objective of Prompt 1.

---

# 17. Prompt 2 — historical implementation scope

Prompt 2's estimators and committed tables now exist. The following preserves
its original scope and prerequisites; the remaining qualification work is in
§21 and W01–W03, not a request to restart Prompt 2 from scratch.

It implements:

- E1;
- E2a0;
- E2a;
- locked Laplacian ranking;
- diagnostic rankings;
- subsets \(p=\{5,10,25,50,100\}\);
- A1o;
- Tukey-windowed \(E_H\);
- planted-feature contrast;
- G1/G2/G3;
- 12 evaluation seeds;
- two crops;
- both mandatory seeing regimes;
- classification tables.

Prompt 2 does **not** implement MFBD.

Before Prompt 2 starts, only the following numerical implementation details may be chosen on development seeds and then frozen:

- E1/E2a0 stabilisation \(\lambda(f)\);
- E2a minimal stabilisation;
- exact planted-contrast aperture/annulus pixel masks if the analytic object geometry does not define them unambiguously.

No Gate-1 threshold may change during Prompt 2.

---

# 18. Q2 / Q3 gate after Prompt 2

Strong G1:

- all-frame blind D / D-tail target.

Strong G2 only:

- blind D / D-tail on \(\mathcal S_{10}\).

Define closure:

\[
C
=
\frac{
E_H(\mathrm{A1o})-E_H(D)
}{
E_H(\mathrm{A1o})-E_H(\mathrm{E2}^{\star})
}.
\]

Target:

\[
\mathrm{median}(C)\ge0.40.
\]

Use:

\[
\mathrm{E2}^{\star}=\mathrm{E2b}
\]

unless E2b's matching oracle gap is less than half the E2a gap, in which case:

\[
\mathrm{E2}^{\star}=\mathrm{E2a}
\]

and label the result prior-limited.

Required Q2 diagnostics:

- two initialisations;
- held-out-frame prediction;
- feature-rich and bland crops separate;
- \(M_{\rm fit}\in\{15,35,60\}\);
- no crop shrinking merely to permit more modes.

Q3 starts only after Q2 reaches the 40% closure target:

\[
N=10^3,\ 5\times10^3,\ 2\times10^4
\]

before scaling the primary MFBD research programme to tiles or spherical
rotation. This dependency does not block independently validated product
geometry and baseline stacking work in §24.

Engineering reporting must include:

> practical reconstruction gain per CPU-hour and, where supported, GPU-hour,
> together with wall time, peak RAM/VRAM, device, precision and thread count.

A GPU is optional; no scientific gate or core application function may require
one. Q3 remains blocked by the current evaluation result described in §21.

---

# 19. Active failure modes

1. inconsistent \(r_0,v,\tau_0\);
2. calling the 32″ Gate-1 field isoplanatic;
3. phase-screen low-frequency loss suppressing lucky excursions;
4. screen wrap or periodic content reuse;
5. no-wrap calculation omitting the final exposure;
6. independent-screen “finite exposure”;
7. simulator and solver sharing a closed low-order basis;
8. circular object convolution or insufficient padding;
9. flux renormalised after seeing instead of fixed as a source rate;
10. photons held fixed when exposure changes;
11. crop-boundary Fourier power mistaken for recovered high-frequency object structure;
12. unweighted high-band metric dominated by MTF≈0 frequencies;
13. bland-crop near-zero denominator treated as a real huge effect;
14. defining G2 with an estimated rather than true stack PSF;
15. E2a and A1o using different object constraints;
16. E2a0 compared with E1 after changing the mathematical objective;
17. tuning E2a/E2b on evaluation seeds;
18. using \(p_E^\star\) rather than fixed 10% for stop logic;
19. applying 8/12 or 3/12 counts unchanged to a pooled 24-seed result;
20. averaging bland and feature-rich crops;
21. stopping after only \(D/r_0=8\);
22. opening Q1-B/Q1-C as rescue branches after a clean two-regime negative;
23. one sharp JPEG.

---

# 20. Success language

## Synthetic opportunity

Strong G1 or strong G2 in at least one mandatory regime, with the other regime reported.

## Q2

Median ≥40% closure of the matching oracle gap, supported and stable.

## Practical result

Later, beat a documented practitioner pipeline on real SER with split-half repeatability and no extra undocumented sharpening.

## Honest pivot

Weak G1/G2, but a cheap quality/frequency-weighted stack approaches E2a:

> publish the cheap estimator and stop full MFBD development.

## Clean negative

G1 and G2 weak/negative at both \(D/r_0=8\) and \(4\) after the inconclusive-seed protocol.

Permitted conclusion:

> In these two spatially invariant, high-cadence, amateur-scale synthetic regimes, a selected stack reconstructed with its true effective PSF is sufficiently close to the known-per-frame-PSF ceiling that all-frame atmospheric-transfer modelling is not justified.

Do not generalise that negative to anisoplanatic fields, detector-sampling diversity, or every planetary observing regime.

---

# 21. Current code evidence and implications

This review uses the source tree and committed results at `d1b600a`. It does not
claim to have rerun the expensive seed families. Ignored local captures and
`out/` artifacts are useful development inputs, not reproducible release evidence.
Implementation presence, unit-test coverage and scientific qualification are
different statuses.

## 21.1 Implemented, but not a finished application

| Area | Evidence in the tree | Consequence |
|---|---|---|
| Synthetic mono simulator | `simulate.py`, `atmosphere.py`, `optics.py`, `object.py`, `hdf5io.py`, `validate.py` under `planetrecon/` | Reuse physics and deterministic fixtures; audit operator consistency before extending them. |
| Known-transfer estimators and Gate 1 | `planetrecon/estimators.py`, `evaluate.py`, `gate.py`, `metric.py`, `rank.py`; `results/prompt2/` | Preserve E1/E2a0 controls and both seeing regimes. Tables are not proof of real-data performance. |
| E2b and blind D / D-tail | `planetrecon/mfbd.py`, `kl.py`, `q2.py`; `results/q2/` | Prototype exists; it still uses synthetic known translations, a snapshot phase model and small fixed iteration budgets. |
| Held-out diagnostic correction | `q2.py`, `tests/test_q2.py`, `results/q2/holdout_dev_seed-01001_feature.json` | Both object starts use training frames only; held-out residual chooses the reported start. One development crop/seed is not a completed family-wide diagnostic. |
| Runtime | `pyproject.toml`, `planetrecon/cli.py` | Python with NumPy/SciPy/h5py, CPU CLI. No device abstraction, Qt6 interface, SER reader, image exporter or standalone release pipeline exists. |
| Tests | `tests/test_prompt1.py`, `test_prompt2.py`, `test_q2.py` | Small synthetic tests are valuable but do not cover real input, geometry, colour, GPU parity, GUI lifecycle or installed binaries. |

All unqualified source filenames in this section refer to `planetrecon/`.

## 21.2 What the committed results actually establish

At fixed practical top-10% selection, evaluation seeds 2001–2012:

| Experiment | Recorded result | Interpretation |
|---|---|---|
| Gate 1, feature crop, D/r0=4 | G1 strong, median about 10.3%; G2 negative, about 0.95%; G3 strong | Opportunity is conditional: E2a0 makes G1 inconclusive, so the result is oracle-sensitive. |
| Gate 1, feature crop, D/r0=8 | G1/G2/G3 negative | Report alongside the moderate-seeing result; not a two-regime clean negative while the other regime is unresolved. |
| Bland crop | High-band truth energy ratio below the validity threshold | Diagnostic only; cannot override the feature decision. |
| E2b prior freeze | TV weight μ=0; E2b=E2a | No demonstrated production-prior benefit or prior limitation for these runs. |
| Q2 development, feature, n=3 | Median C about 0.466 | Numerical target met on development data only. |
| Q2 evaluation, feature, n=12 | Median C=0.3638845774; 3/12 seeds at or above 0.40 | Evaluation gate fails. Q3 must not start on this evidence. |
| Q2 evaluation, bland | Median C about 0.88 | Ill-conditioned diagnostic, not a pass. |
| Corrected held-out development diagnostic | Seed 1001 feature, 450 train / 50 held out; subset start; residual ratio 1.0103; all-frame C=0.5244 | Limited consistency evidence. The held-out object's phase is fitted on held-out images, so this is conditional phase-fit prediction, not a fully untouched predictive test. |

Sources: `results/prompt2/classification_eval_feature.json`,
`classification_eval_bland.json`, `frozen_regularisation.json`, and
`results/q2/frozen_prior.json`, `closure_dev_feature.json`,
`closure_eval_feature.json`, `closure_eval_bland.json`,
`holdout_dev_seed-01001_feature.json`. The family tables and the single corrected
held-out run have different protocols; do not silently combine them.

Observed weak improvement from 15→35→60 modes does not by itself prove a physical
information limit. Forward-model mismatch, imperfect constraints, optimizer
stopping and phase identifiability must first be separated. Neither a clean
negative nor a successful practical MFBD product is established.

## 21.3 Code-evidenced deficits to resolve before further claims

These are either directly visible implementation gaps or explicitly labelled
audit questions. Fixes must be tested, not presumed to improve closure.

1. **Constraint enforcement:** `estimators._project_e2a` clips positive, applies
   spectral support, then clips again. The final clipping can restore power
   outside support; this is not an exact projection onto the intersection. Use
   a converged constrained method with feasibility and optimality diagnostics,
   equally for E2a, E2b and A1o. Reassess oracle sensitivity after the correction.
   A bounded CPU diagnostic during this review confirms the mechanism: a 32×32
   unit impulse projected with circular support below 0.12 cycles/pixel remains
   nonnegative but has out-of-support Fourier norm / total Fourier norm ≈0.187657.
   This is a regression example, not a measurement of error in the seed tables.
2. **Forward/adjoint consistency audit:** the simulator convolves a padded scene
   before detector integration/cropping, whereas estimator and MFBD paths use
   crop-sized FFT products and circular Fourier registration. Quantify the
   resulting boundary, sampling and noise-covariance mismatch with noiseless
   fixtures; share the correct forward/adjoint or explicitly validate an interior
   approximation. A scalar spatial-average variance is a frozen approximation,
   not the exact heteroscedastic detector likelihood.
3. **Phase fitting and shifts:** `fit_frame_alpha` drops optimizer termination
   status; its freeze-tip/tilt branch does not cover vectors with only two modes.
   Preserve frozen coefficients in every dimension, expose gradient norms and
   iteration limits, and test nonzero phases as well as near-zero ones.
   `tip_tilt_from_shifts` uses a centroid Jacobian calibrated at a single step;
   validate accuracy across actual shifts and high-mode perturbations. Do not
   confuse measured image motion with independently known physical pupil tilt.
4. **Exposure and identifiability:** `PupilForward` fits snapshot PSFs to
   finite-exposure simulation. Measure this approximation's ceiling before adding
   modes or compute time. Control piston, object/PSF flux, global shift and colour
   gains so interchangeable parameters cannot drift. Keep truth-derived shifts,
   variances and transfers out of the real-data solver interface.
5. **Frozen model selection:** `run_q2_family` reads a frozen E2-star name without
   using it; per-file oracle-gap selection can affect the blind fit's prior when
   μ is nonzero. Freeze the blind architecture and prior using development data
   only. The §18 per-case E2-star rule may label the reporting denominator, but
   must never feed evaluation truth back into D. At the current μ=0 this does
   not imply that the recorded numerical table changed.
6. **Invalid results and provenance:** `closure_C` admits negative oracle gaps;
   `aggregate_q2` drops non-finite closures and can declare a pass on an incomplete
   finite subset. Require positive, well-conditioned opportunity, complete unique
   seed/crop/regime coverage and valid fits before certification. Record invalid
   cases and reasons instead of discarding them. `_to_jsonable` must preserve
   booleans rather than serialize them through the integer branch; use explicit
   nulls/status for undefined quantities rather than non-standard JSON NaN.
7. **Method certification:** `certify_development_validations` checks the seed
   family and pass flags but needs a complete compatibility fingerprint before
   copying method-level certification to other files. Bind certificates to
   physics, grid, exposure integration, padding, augmentation, operator version
   and source/config hashes; retain local diagnostics independently.
8. **Unfinished evidence:** complete the §3 practical-ranking/G1/G2 low-frequency
   augmentation check, required held-out coverage and optimizer diagnostics;
   turn the existing resource CSVs into the §12 plots. Recheck what each stored
   pass flag actually certifies. A test passing against its own simplified model
   is not independent validation of that model.

---

# 22. Final application contracts

## 22.1 Scope and honest operating modes

The final application must open real captures, reconstruct on a CPU without a
discrete GPU, optionally accelerate on supported GPUs, display a progressively
updated output in Qt6, and save scientific-depth images. Field rotation and
planetary rotation must work together; Saturn must not use a single spherical
warp for its globe and rings.

Expose two reconstruction families with the same input/output contract:

- **Validated baseline:** calibrated, quality-weighted registration and stacking,
  with detector-aware colour reconstruction and geometry-aware accumulation.
  Conventional debayer-then-stack remains a labelled comparison/quick-look path.
- **Physical reconstruction:** joint object/PSF inference using the validated
  detector and scene operators. Label experimental until the applicable
  scientific and real-data checks pass. A baseline-only milestone is useful but
  is not completion of this roadmap's raw-CFA and rotation requirements.

No algorithm may invent detail beyond optical support, infer trustworthy
orientation from an uninformative disc, or quietly fall back to a different
scientific objective. Unsupported or ambiguous inputs require a clear choice or
diagnostic. The source files are read-only.

## 22.2 Architecture and ownership

Retain the headless Python engine and CLI. Add PySide6/Qt6 Widgets as a thin
front end, with a raster image canvas so even the GUI does not require hardware
OpenGL. PySide6 provides Qt6 Python bindings; deployment is a separate build
step, not something the end user runs. See the [official Qt for Python overview](https://www.qt.io/development/qt-framework/python-bindings).

Proposed package boundaries (new unless explicitly marked existing):

| Boundary | Responsibility and contract |
|---|---|
| `io/ser.py`, `io/video.py`, `io/source.py` | `FrameSource`: metadata, indexed raw reads, bounded batch iteration, timestamps and close/context-manager lifecycle. Wrap existing synthetic HDF5 input behind a separate adapter. |
| `calibration.py`, `detector.py` | Raw ADU interpretation, calibration/masks, detector integration, CFA/RGB response and noise likelihood; matched forward and adjoint. |
| `geometry/` | `SceneModel` implementations for flat reference, oblate rotating globe and Saturn layers; per-frame pose, visibility and confidence. |
| `operators/` | Composable render, blur, integration, sampling, crop and adjoint operations; explicit coordinate frames, units, padding and precision. |
| Existing `estimators.py`, `mfbd.py`, `kl.py` | Scientific algorithms refactored to consume operators/backend without GUI dependencies; retain historical R9 execution for comparison. |
| `backends/` | CPU reference and optional accelerator adapters; capability probes, memory planning and explicit fallback. |
| `pipeline/`, `jobs.py` | Immutable job configuration, stage scheduling, cancellation, checkpoints, bounded preview/progress events and worker ownership. |
| `result.py`, `io/export.py` | Floating linear result, coverage/variance/masks, reference time/orientation and provenance; scientific export separate from display mapping. |
| `gui/` | Source setup, controls, image/progress view and error recovery; Qt objects and painting remain on the GUI thread. |
| `packaging/`, `.github/workflows/` | Native standalone build recipes, dependency locks, licenses, artifact smoke tests and release manifests. |

`ObservationMetadata` must distinguish measured, header-supplied, inferred and
user-overridden values. `ReconstructionConfig` is serializable and immutable for
a running job. `ReconstructionResult` carries units, channel order, validity mask,
reference epoch, transforms, processing history and actual backend/precision.
Progress and preview events have a job ID and monotonically increasing sequence
number so cancelled jobs cannot overwrite a new view. Truth arrays are available
only to simulation/evaluation, never through the production `FrameSource`.

Version document, experiment, input schema, result schema and operator contract
separately. Checkpoints/cache entries include input identity, configuration,
operator version, precision and software version; incompatible entries are
rejected rather than reused. Large captures are streamed/memory-mapped and
revisited in bounded batches; neither all frames nor all per-frame OTFs may be
required in RAM or VRAM.

## 22.3 CPU and optional GPU execution

The NumPy/SciPy float64 implementation is the numerical reference and the default
on the current development machine. Implement an explicit `CPU / Auto / GPU`
choice, CPU thread limit and RAM/VRAM budgets. Start with one controlled level of
parallelism; do not multiply process pools by unrestricted BLAS/FFT threads.

Planned accelerator adapter: PyTorch for supported CUDA devices on Windows/Linux
and MPS/Metal on macOS. This is a backend choice to qualify, not an assertion that
every required dtype, complex FFT, interpolation or gradient is supported on
every device. MPS is the documented macOS GPU backend; check build/device
availability and run the actual operator probes before selecting it. See
[PyTorch's MPS documentation](https://docs.pytorch.org/docs/stable/notes/mps.html).

Probe forward, adjoint, FFT, reductions, indexing, gradient and allocation at the
chosen dtype. Prefer a consistent whole-stage CPU fallback to repeated hidden
device transfers. `Auto` must recover to CPU after an unsupported operation,
driver failure or out-of-memory event, starting from the last valid checkpoint.
An explicit GPU request must show the failure and offer CPU continuation, not
crash or quietly claim GPU processing. Never silently drop frames, shrink the
requested output or change regularization to fit device memory.

Use float64 reference tests; qualify float32/mixed precision against them with
development-frozen tolerances on objective, flux, gradients and reconstruction
metrics. Record CPU/GPU differences and all fallbacks. GPU CI may be skipped on
ordinary CPU runners, but a GPU release requires real-device end-to-end evidence.
No CUDA/Metal import or availability check may prevent CPU-only startup.

## 22.4 SER and one-shot-colour special handling

Implement metadata-first raw SER ingestion. The v3 format has a 178-byte header,
unsigned 1–8-bit samples in bytes or 9–16-bit samples in words, and an optional
per-frame timestamp trailer. `ColorID` identifies mono (0), RGGB (8), GRBG (9),
GBRG (10), BGGR (11), RGB (100) or BGR (101); other colour IDs must not be
misclassified as Bayer RGB. Preserve the file's top-left sample origin, declared
bit depth and channel order. Header integers are little-endian; image byte order
is a separate field. See the [SER v3 specification](https://free-astro.org/images/5/51/SER_Doc_V3b.pdf).

The specification and common writers interpret the image-endianness flag in
opposite ways. Default to the documented ecosystem convention (0 means
little-endian), record the interpretation, provide an explicit override and
test both conventions with golden files. Do not silently byte-swap based only
on a plausibility heuristic. See [Siril's interoperability explanation](https://siril.readthedocs.io/en/latest/file-formats/SER.html#specification-issue-with-endianness).

Validate dimensions/counts against actual file length before allocation; support
large offsets, reject incomplete payloads by default and offer explicitly
labelled recovery of complete frames. Preserve significant bits without
per-frame stretching. Treat corrupt timestamps, duplicates, nonmonotonic timing
and unknown exposure duration separately. SER does not supply all telescope,
gain, mount or planet-pose parameters needed by the physical model; request
missing values or use clearly labelled estimates. Record cadence separately from
exposure, and never substitute local clock time for an unknown UTC epoch.

The original discussion's special handling is **joint reconstruction of R/G/B
latent images directly from the undebayered samples**, with CFA sampling inside
the observation model. It is not a request to demosaic every frame first. See
the original [Planetary Image Reconstruction discussion](https://chatgpt.com/c/6a9a4c5d-619c-83a8-964f-036c1e771523)
(account access may be required).

Required behaviour:

1. Use a valid Bayer `ColorID` automatically. If a writer labels raw CFA as mono,
   let the user choose mono or the four Bayer patterns with colour previews.
   An optional statistical suggestion may carry confidence, but texture and
   narrowband data cannot uniquely identify a pattern; never force a guess.
2. Track CFA parity through integer crops, flips and rotations. The physical CFA
   stays fixed in **detector coordinates**; subpixel alignment/field derotation
   acts on the rendered latent scene, not by interpolating a raw mosaic and then
   pretending it has the original independent Bayer samples.
3. Keep both green sublattices as distinct measurements of one green radiance
   field. Joint RGB inference uses channel-specific response/optical support,
   noise and calibration. Share atmospheric optical path difference, scaling
   phase by wavelength; do not blindly share a numerical phase or PSF across
   wavelengths. Start with declared effective bandpasses and qualify their
   approximation before adding spectral quadrature.
4. Apply bias/dark/flat, bad-pixel and saturation masks in detector/CFA space.
   Model gain/offset and exposure explicitly. Missing calibration yields a
   labelled approximate-noise mode, not invented electron counts. Avoid free
   per-channel gains that can absorb arbitrary changes in the latent object.
5. Use natural subpixel diversity without claiming optical super-resolution.
   A green-derived proxy may help alignment/ranking, but the reconstruction must
   retain red and blue measurements. Compare with a conventional demosaiced
   baseline on identical frames, geometry, support and photon budgets.
6. Include mosaics of all four patterns, odd crop origins, flips, subpixel shifts,
   simultaneous rotation, saturated pixels and narrowband scenes in tests.
   Measure channel flux, false colour, edge artifacts and held-out raw residuals,
   not only the apparent sharpness of an RGB display.

RGB/BGR SER bypasses CFA sampling but still needs linearity and channel-order
checks. Legacy AVI support remains a separate adapter with a declared tested
codec/pixel-format list; bundle its decoder rather than requiring external
FFmpeg. Compressed/nonlinear video is labelled unsuitable for the calibrated
physical likelihood unless its transformations are known.

## 22.5 Field rotation and planetary surface rotation

These are distinct operators and may both be present in an alt/az capture.
Field rotation changes the sky's orientation on the detector; surface rotation
changes visible planetary longitude with foreshortening and occlusion. A single
2-D image rotation does not correct both.

Use an explicit reference epoch and coordinate conventions: body-fixed surface,
projected sky and detector. Render a time-dependent oblate globe into the sky,
apply sky-to-detector attitude/translation, blur in the appropriate detector
optical frame, integrate over pixel area/exposure, then apply CFA sampling and
noise. An asymmetric pupil or instrument feature must have its own orientation
convention; do not rotate the atmospheric PSF merely because the sky rotates.
Compose geometry before resampling, avoiding repeated lossy image warps.

For expected raw data, the extended model is schematically

\[
\mu_k = b_k + g_k\,\mathcal C_{\rm CFA}\mathcal P
\int_{t_k}^{t_k+T_k}
\left[h_{k,t,\lambda} *
\mathcal A_{k,t}\mathcal V_{t,\lambda}(S)\right]dt,
\]

where \(S\) is the latent scene, \(\mathcal V\) renders surface/ring visibility,
\(\mathcal A\) is sky-to-detector pose, \(\mathcal P\) integrates detector pixels,
and \(\mathcal C_{\rm CFA}\) selects calibrated colour responses (identity/channel
selection for mono/RGB). Units must distinguish source rate, integration duration
and detector gain. The quadrature and adjoints must account for both atmospheric
and geometric motion; freeze-mid-exposure geometry only after measuring its
error. This is a new operator, not a retroactive change to the R9 fixture.

Estimate disc/ellipse centre, scale, pole angle and frame poses from the sequence,
with manual adjustment and confidence. Field-angle estimates can combine image
features with an optional mount/location/UTC model; metadata is initialization,
not infallible truth. Near-circular featureless discs cannot constrain roll.
Handle angle unwrapping, changing rotation centres, missing frames and fast
rotation near the zenith. Do not require online ephemeris access: support manual
pose/time parameters and bundled offline ephemeris data with documented coverage
if that initializer is provided.

Map globe radiance in body coordinates with visibility masks and limb weighting;
back-project only observed surface areas into the reference view. Do not fill
unseen longitudes with inferred detail or treat black warp borders as data.
Start with a declared rigid rotation rate and oblate geometry; longer Jupiter
clips require a separately qualified latitude-dependent rate/evolution model or
a warned duration limit. Rates, pole direction and apparent diameter must be
adjustable when absent from the file. Show common coverage and uncertainty.

Test field-only, spin-only and combined motion against independently rendered
truth, including zero motion, limbs entering/leaving, angle wrap and insufficient
texture. Controls must use the same geometry. Full-disc spatially varying seeing
requires validated local/tile PSFs and seam-free overlap, or a clearly bounded
single-PSF mode; the existing 128-pixel crop result does not establish this.

## 22.6 Saturn: globe and rings are separate scene layers

Provide a Saturn scene model, not an ellipse fed into Jupiter's disc warp.
Represent an oblate rotating globe plus a projected equatorial annulus with
inner/outer radii, ring opening angle, pole orientation and independent radiance.
Use a depth/visibility model for the near ring in front of the globe and the far
ring behind it. Include ring transmission/opacity at overlaps, plus globe/ring
shadow masks or a qualified illumination
model; uncertain regions can be masked rather than forced into the PSF fit.
Instrumental blur acts on the correctly composited sky scene, not on independently
stacked cutouts pasted together after deconvolution.

Field rotation acts on the entire composite. Globe surface rotation must **not**
drag rings with it. Initially treat rings as a static, axisymmetric radiance
profile over a qualified short interval; do not claim that physical ring
particles are stationary: ring orbital speeds vary with radius, as described in
[NASA's ring-plane discussion](https://science.nasa.gov/missions/hubble/hubble-views-saturn-ring-plane-crossing/).
Resolved azimuthal features require an independent
radius-dependent orbital/evolution model or masking/shorter intervals, not the
globe's spin rate. Rings, globe and background retain separate coverage and
regularization so a sharp ring edge cannot masquerade as globe detail.

Use ring-aware pose fitting and quality masks; do not let bright ansae dominate
every quality score or shrink the ROI until rings disappear. Handle low-opening
and edge-on rings as poorly conditioned geometry, with manual override and
conservative masking. Moving moons are separate tracked/masked sources and must
not become stationary surface features. Validate open, nearly closed and edge-on
rings, both signs of opening, globe occlusion, shadows and simultaneous field/
surface rotation. Assess globe and ring-region residuals independently.

## 22.7 Qt6 interaction, continuous image and progress

Minimum workflow: open capture → inspect metadata/CFA and calibration → choose
target/geometry and reference time → choose CPU/Auto/GPU and resource limits →
run/cancel → inspect and save. Include input-frame, progressive-output and
coverage views; zoom/pan, histogram, channel/display stretch and before/after
comparison must not mutate the scientific result.

Display the latest actual accumulated/reconstructed output while processing,
not only source thumbnails or an animation of the final image. Publish a first
baseline image after the first usable batch, refresh as batches accumulate, then
publish solver iterates at bounded intervals (target 2–5 Hz when new states are
available). Each snapshot is immutable or safely copied; throttle/downsample
preview transport, not science processing. Label baseline, intermediate and final
states. Keep the reference view and display mapping stable unless the user
changes them; a changing stretch must not mimic improving detail.

The progress bar reports actual stage work: scan/calibration, pose estimation,
accumulation, reconstruction passes and finalization. Show frame/batch counts,
iteration/limit, elapsed time, estimated remaining time when meaningful, backend
and warnings. Multiple passes over all frames are not 100% completion after the
first pass. Use an indeterminate stage only when its work count is genuinely
unknown; 100% means a final result has been published, not numerical convergence.

Run numerical work in an owned worker process; keep Qt widgets and painting on
the GUI thread and transfer events through a bounded channel. Coalesce old
previews without dropping errors or final/cancellation events. Cancellation is
checked between bounded work units; on cancel, error, window close or parent
exit, close frame sources and release shared buffers/device allocations, then
join owned workers. After a bounded grace period terminate only this job's
identified children. Never kill unrelated Python processes. Resume from a
compatible checkpoint; a new job cannot inherit an old job's result messages.

## 22.8 Scientific export

Required choices: **16-bit unsigned PNG**, **16-bit unsigned TIFF**, and **32-bit
IEEE floating-point TIFF**, each supporting mono or RGB. Here “16/32-bit TIFF”
means these two explicit encodings, not an ambiguous 16-bit-float default.

Export from the floating linear reconstruction, never the Qt preview or an
8-bit screenshot. For integer output, show and record a fixed black/white mapping,
rounding and clipping counts; do not rescale each channel/frame independently.
Default to preserving linear scientific intensity. Offer a separately labelled
display-rendered export with explicit transfer function/colour metadata; avoid
accidental double gamma. Float TIFF preserves scale and dynamic range without
normalizing to 0–1. Document validity masks/NaN policy and any resampling.

Use bundled, tested encoders (plan: a dedicated 16-bit PNG writer and `tifffile`
for TIFF; any chosen compression codec must also be bundled). The
[PNG specification](https://www.w3.org/TR/png-3/) defines 16-bit grayscale and
truecolour samples; [tifffile's documentation](https://github.com/cgohlke/tifffile)
covers integer/floating-point and multi-sample TIFF output. Validate actual
PNG bit depth and TIFF BitsPerSample/SampleFormat, channel order, endianness and
pixel values with an independent reader. Test values on both sides of 255,
ramps to 65535, RGB channels, finite float values above 1 and negative values
where the selected result permits them. Read-back equality must hold for
lossless integer output after the declared mapping, and float32 within its
representational precision.

Write atomically, ask before overwrite, handle full disks/permissions, and keep
the in-memory result on export failure. Save reference epoch, input fingerprints,
CFA interpretation, calibration, geometry, device/precision and algorithm settings
in appropriate metadata plus a JSON sidecar. Coverage/uncertainty can be separate
TIFF pages/files with explicit descriptions. Saving an intermediate snapshot is
allowed, but must be labelled incomplete and must not stop processing.

## 22.9 Standalone Windows, Linux and macOS delivery

“Without external dependencies” means **no user-installed Python, Qt, numerical
libraries, image/video codecs, pip environment or command-line tools**. Bundle
the application runtime and needed redistributable libraries. Supported OS system
libraries and vendor GPU drivers remain platform prerequisites; the CPU build
must work without GPU drivers/toolkits. Do not promise a binary that works on
every historical OS or GPU.

Start with PyInstaller native per-platform builds and checked-in specifications;
keep an early PySide6 packaging spike to catch plugin/runtime issues. Qt's own
`pyside6-deploy`/Nuitka path is a documented alternative if measured build issues
justify switching, not a second mandatory release system. PyInstaller is not a
cross-compiler: build/test on each target OS. See the [PyInstaller manual](https://www.pyinstaller.org/en/stable/)
and [Qt deployment tool documentation](https://doc.qt.io/qtforpython-6/deployment/deployment-pyside6-deploy.html).

Planned artifacts: Windows x86-64 portable folder/installer; Linux x86-64
AppImage or bundled archive with a declared glibc baseline; macOS arm64 `.app`
in a disk image, with a separately tested x86-64 build if offered. Freeze minimum
OS versions and exact Python/Qt/backend versions after the packaging spike.
Do not claim universal macOS support from an arm64-only build. Test Linux Qt
platform plugins on supported X11/Wayland environments and use software rendering
where needed.

Ship an offline CPU-capable build for every supported platform. Optional larger
accelerated builds may bundle qualified accelerator runtimes; they must still
start and process on CPU when no compatible GPU exists. No first-run downloads.
Pin dependencies with hashes, audit redistributability and Qt/backend/codec
license obligations, include notices/SBOM, sign Windows artifacts and sign/
notarize macOS releases where credentials are available. Missing signing
credentials are a release action item, not a reason to bypass OS security.
Test on clean machines without development environments or network access.

---

# 23. Scientific controls for the expanded application

Keep the R9 monochrome oracle gate distinct from product acceptance:

1. Preserve the original seeds, thresholds and old results. Correct demonstrable
   numerical defects on development fixtures, freeze a new operator/configuration,
   then report paired old/new results and the reason for changes. Already viewed
   evaluation seeds are regression data for later redesigns, not fresh holdouts.
   Declare an additional untouched validation family before claiming generalization.
2. Never spend the predeclared Gate-1 inconclusive extension as a way to rescue a
   failed Q2 result. Apply §11 only to the unresolved Gate-1 classification after
   the oracle/validity audit. A clean negative stops the primary MFBD claim; it
   does not cancel the requested desktop product.
3. For real data, estimate shifts, seeing, noise and pose from measurements or
   declared calibration only. Synthetic truth stays in an evaluation-only module.
   Split training, model-selection and final assessment data before fitting;
   phase fitting on held-out frames must be labelled conditional. Use a separate
   pixel/time holdout when claiming predictive performance without refitting.
4. Compare baseline and physical solver on the same valid samples, colour model,
   geometry, reference time, support and calibration. Report frames rejected and
   why. Do not reward one method for different sharpening, clipping or coverage.
5. Add independently generated colour/rotation/Saturn fixtures and real capture
   split-half comparisons. Check photometry, raw-domain residuals, repeatable
   features, ringing, false colour, limb/ring seams and sensitivity to calibration.
   No real-data claim rests on a single attractive image.
6. Declare acceptable clip durations, field sizes, sampling and geometry limits
   empirically. Reject or warn outside them. Differential rotation, appearance
   evolution, rolling shutter and spatially varying atmosphere are model issues,
   not problems to hide with stronger sharpening. Add explicit models only where
   qualified, otherwise expose the limitation and use shorter intervals/masks.

---

# 24. Remaining code changes in dependency order

Each work package ends with focused tests, a source/results commit and an update
to the evidence table. These are **planned**, not implemented by this document.
Keep numerical corrections separate from GUI/packaging commits. Development on
the current machine uses CPU-sized fixtures; long jobs need bounded resource
settings, progress, an owned lifecycle and a reason to retain their results.

| ID | Work package | Depends on | Completion evidence |
|---|---|---|---|
| W00 | Versioned evidence/provenance and reproducible test tiers | Existing tree | Historical results preserved; manifests and short CPU test command documented. |
| W01 | Estimator/operator correctness and certification audit | W00 | Constraint, forward/adjoint, noise and certificate regression tests pass. |
| W02 | Q2 optimizer, selection, validity and diagnostics | W01 | No truth-driven blind selection; complete typed diagnostics and held-out suite. |
| W03 | Scientific requalification and decision report | W02 | Corrected controls and scoped Gate-1/Q2 outcomes, including failures, published. |
| W04 | Headless source/config/result/operator contracts | W00 | Synthetic adapter and existing CLI work through bounded interfaces. |
| W05 | Owned jobs, CPU resource budgets and checkpoints | W04 | Cancellation/close/error leave no owned workers; memory bounded. |
| W06 | Raw SER reader and calibration | W04 | Golden-format matrix and real header/sample tests pass without changing inputs. |
| W07 | CPU baseline and raw-CFA RGB reconstruction | W01, W05, W06 | Progressive mono/RGB/CFA results and matched baseline controls. |
| W08 | Qt6 shell and early native packaging spike | W04, W05 | GUI opens on CPU-only systems; a tiny synthetic job updates/cancels. |
| W09 | Field rotation and rigid globe surface model | W07 | Separate and combined rotation recovery with masks and confidence. |
| W10 | Saturn layered scene model | W09 | Globe/ring occlusion, independent motion and degenerate poses validated. |
| W11 | Production physical solver and full-disc qualification | W02, W07, W09, W10 | Raw-data operator, exposure, local PSFs and scientific limitations tested. |
| W12 | Optional accelerator backend | W01, W04, W05, W07 | Real-device parity, memory limits and CPU fallback verified. |
| W13 | Scientific PNG/TIFF export | W04, W07 | Exact depth/type read-back, atomic writes and provenance. |
| W14 | Complete Qt6 workflow | W08–W10, W13 | Real progressive image, truthful progress, controls and recovery end to end. |
| W15 | Legacy video adapter and large-capture hardening | W05–W07 | Declared AVI formats plus long/offline/low-memory tests. |
| W16 | Real-data comparisons and performance qualification | W03, W10–W15 | CPU and supported GPU benchmarks; independent captures and comparisons. |
| W17 | Standalone releases and final acceptance | W08, W16 | Clean-system, offline Windows/Linux/macOS acceptance matrix passes. |

W04–W10, W13–W15 and the packaging spike need not wait for a positive MFBD
result. W11's advanced atmospheric claims and any Q3 scale-up remain gated by
W03. W12 may start with the baseline and operator tests, but its full qualification
must include the later colour, geometry and Saturn paths. Dependencies indicate
code readiness, not authorization to spend unbounded compute in parallel.

## 24.1 W00–W03: repair the evidence before scaling

**W00:** add an experiment manifest and typed result schema with source hash,
config hash, input identities, seed coverage, protocol, tolerances and statuses.
Distinguish `diagnostic`, `valid`, `incomplete`, `invalid` and `gate_passed`.
Archive the current tables unchanged. Split fast CPU unit tests, bounded
integration tests, expensive scientific families and hardware/release tests.
Do not put expensive simulations in the default test suite.

**W01:** address §21.3 items 1, 2 and 7 in `estimators.py`, `optics.py`,
`evaluate.py`, `validate.py`, `hdf5io.py` and corresponding tests. Use a proper
intersection-constrained method (for example converged Dykstra projection within
a qualified solver, or a primal-dual formulation), not an extra arbitrary clip.
Measure positivity violation, out-of-support energy and a projected-gradient/KKT
residual. Preserve the E1/E2a0 identical-objective control. Test padded convolution,
crop/bin phase, flux and registration against an independent spatial calculation,
with dot-product adjoint tests and finite-difference gradients. Quantify the
constant-noise approximation before changing the frozen statistical model.
Bind method certificates to compatible configurations and test stale/mismatched
certificates explicitly. Complete low-frequency ranking/gap convergence checks.

**W02:** fix and test the small-mode frozen-tip/tilt case, report optimizer status,
gradient norm, objective trace, feasibility and actual iterations. Benchmark the
shift-to-pupil mapping over recorded displacement ranges without launching whole
seed families. Separate known-shift, estimated-shift and no-shift controls.
Measure snapshot-versus-exposure model error and phase-basis residual. Freeze
blind prior/model choice independently of evaluation truth; reject non-positive
or ill-conditioned closure denominators and incomplete seed families. Complete
two-start held-out diagnostics on all three development seeds, both crops, all
declared mode stages, with separate model-selection and assessment labels.
Add tests for non-finite fits, duplicate/missing seeds, all-invalid crops and
boolean/null JSON round trips.

**W03:** first run bounded development ablations that separate operator error,
constraints, optimization budget and phase model; do not simply raise M or N.
Freeze corrections, then regenerate only affected results into a new experiment
directory. Publish E1/E2a0 agreement, E2a oracle sensitivity, both seeing regimes,
metric validity, prior limitation and closure dispersion. Add the resource plots
with captured/used frames, photons, wall time and memory. Apply original stop and
extension rules honestly. Q3's 1k/5k/20k runs require a qualified Q2 pass and a
CPU/GPU resource forecast; otherwise leave the advanced solver experimental or
stop its primary claim and ship the validated baseline.

## 24.2 W04–W08: usable CPU-first vertical slice

**W04:** introduce the §22.2 contracts, unit/coordinate conventions and synthetic
adapter. Refactor incremental reads and accumulations without retaining the full
capture. Keep CLI and GUI clients of the same engine, and make synthetic truth
impossible to pass accidentally through a real source. Add schema migration/
rejection tests and deterministic cache keys.

**W05:** implement job states `queued → running → completed/failed/cancelled`,
cooperative cancellation, bounded event queues, checkpoint validation and exact
worker ownership. Cap CPU processes/threads and memory from explicit settings.
Test cancel during decode, registration, fitting and export; failed worker and
parent exit; immediate rerun; and Windows/macOS spawn semantics. No job may
continue consuming CPU after its results have been abandoned.

**W06:** build the SER parser and calibration path described in §22.4. Commit tiny
generated golden files or their deterministic generator, not private multi-GB
captures. Test mono/CFA/RGB/BGR, 8/16-bit storage and sub-word depths, both endian
conventions, absent/truncated trailers, large offsets, corrupt headers, Unicode
paths and odd ROI parity. Validate selected local raw frames against a trusted
reader without modifying the recordings. Supply explicit pattern/byte-order/
timing overrides with provenance.

**W07:** implement calibrated registration/quality estimation without truth,
robust rejection of saturation/corruption, incremental weighted accumulation and
coverage maps. Add a CPU raw-CFA joint RGB solve with fixed/estimated transfers
before coupling it to blind phase fitting. Matched adjoints handle sampling and
subpixel movement; confidence masks avoid unsupported colour fill-in. Compare
against demosaic-first stacking on synthetic and real subsets. Emit actual
baseline and iterative result snapshots through W05.

**W08:** add PySide6 Widgets entry point and source/config/result panes using a
raster display. Run a tiny synthetic source through the owned worker and show
an evolving image plus progress. Exercise cancel/close/restart. Build minimal
native packages on all three OS families now, including Qt platform plugins,
before adding large accelerator runtimes. Record support floors and build locks.

## 24.3 W09–W12: geometry and physical inference

**W09:** implement pose fitting, angle unwrapping, reference epoch, field rotation
and oblate-globe visibility/rotation as composable operators with tested adjoints.
Add manual geometry controls and degeneracy reporting. Implement geometry-aware
baseline accumulation first, then connect the raw-data likelihood. Validate
motion during exposure, long-clip limits and joint CFA/rotation sampling.

**W10:** implement globe/ring layers, depth ordering, radiance profiles and
illumination masks. Fit ring geometry jointly with field attitude, not through a
disc-only estimator. Test near/far occlusion, ansae, edge-on degeneracy, moving
moons and independent globe rotation. Publish region-separated metrics and
coverage. Add more detailed ring motion only if the static-short-clip validity
test fails on otherwise supported data.

**W11:** integrate colour, geometry, detector noise and exposure into MFBD with
shared physical wavefront variables and explicit gauges. Generalize from the
small synthetic crop to full discs using validated padding and, where needed,
local PSFs/overlap blending. Fit/use temporal exposure models with held-out checks;
do not assume independent snapshots reproduce long exposures. Compare local
baselines with the same geometry and assess seams at limbs and rings. Keep
geometry-only reconstruction available regardless of the MFBD science outcome.
Qualify differential surface rotation/evolution or enforce reported duration
limits; do not label a single-PSF rigid model universally applicable.

**W12:** port the dominant measured costs behind backend interfaces, not all code
indiscriminately. Batch/cache transfers within a VRAM budget; preserve CPU
implementations and deterministic test inputs. Qualify complex FFTs, warps,
adjoints and gradients on each supported device/dtype. Test forced CPU on a
GPU host, GPU unavailable at startup, unsupported operation and mid-job OOM.
Publish end-to-end speedup including I/O/transfers, not only a kernel benchmark.

## 24.4 W13–W17: complete workflow and release

**W13:** implement §22.8 encoders, integer mapping, float preservation, metadata
and atomic output. Use independent format inspection/read-back tests, including
RGB16 PNG (not only grayscale), float TIFF, save-while-running and interrupted
write recovery. Bundle all selected codecs in the packaging smoke test.

**W14:** connect real sources, device/resource selection, calibration/CFA overrides,
planet/Saturn geometry, live scientific preview, stage progress and save controls.
Show confidence, clipped pixels, current reference time and incomplete coverage.
Test UI responsiveness under full CPU load, stale events, worker errors, cancel
latency and source/result memory ownership. A processing error must leave the
last valid image inspectable/saveable and make the failure visible.

**W15:** add the separately qualified AVI adapter with bundled decoders and
linearity/pixel-format warnings. Stress captures larger than RAM, uneven timing,
low disk space, file disappearance, corrupted frames and process interruption.
Stream frame statistics and checkpoint sufficient state without rewriting input.
Bound preview/queue/cache growth and prevent nested-thread oversubscription.

**W16:** maintain a consented, documented real-data corpus covering mono,
one-shot colour, alt/az field rotation, significant surface rotation and Saturn.
Use independent nights/cameras and predeclared comparisons; anonymize observer
metadata when distributing fixtures. Publish split-half/raw-residual checks and
baseline comparisons without undocumented sharpening. Record CPU/GPU hardware,
frame dimensions/counts, precision, wall time, peak memory, first-preview latency,
cancel latency and reconstruction quality. Establish supported limits from those
measurements rather than promising real-time reconstruction on the current CPU.

**W17:** lock build dependencies, produce signed/notarized artifacts as applicable,
bundle licenses/SBOM and verify checksums. On clean offline target systems, open
mono and Bayer SER, reconstruct on CPU, view continuous progress, process the
rotation/Saturn fixtures, save all required formats, cancel/restart and exit with
no owned workers. Repeat supported accelerator cases on real hardware and test
fallback. Ship a quick-start guide, support matrix, known limitations and a
reproducible release manifest. No development Python/Qt/FFmpeg installation may
be part of the acceptance environment.

---

# 25. Release acceptance matrix

Freeze numerical tolerances and fixture sizes during development before using
the final acceptance corpus. Unit correctness and product acceptance are both
required; an attractive preview is not a substitute for either.

| Requirement | Required acceptance test |
|---|---|
| CPU optionality in practice | All core input, reconstruction, geometry, preview and export cases complete on a supported CPU-only machine with a bounded memory budget. |
| GPU optionality in practice | Qualified GPU backend performs reconstruction, agrees with CPU within frozen tolerances, and unavailable/failed GPU cases offer or perform explicit CPU fallback. |
| Qt6 GUI | Clean packaged startup, responsive zoom/controls, evolving output, truthful progress, cancel/save/restart and window-close cleanup. |
| Raw one-shot colour | Every Bayer ID/parity fixture reconstructs the correct channel orientation and flux; unknown pattern can be overridden; raw likelihood never consumes a prewarped/debayered mosaic. |
| Scientific export | Independent read-back confirms PNG uint16, TIFF uint16 and TIFF float32 for mono/RGB, correct mapping/metadata and safe failure on interrupted writes. |
| Alt/az plus surface rotation | Joint-motion fixtures recover a common reference view without limb fill-in or CFA-phase corruption; ambiguous orientation is reported. |
| Saturn | Ring and globe motion are separate; near/far occlusion and low-opening cases pass region-specific checks with honest coverage. |
| Large inputs and process lifetime | Capture exceeds RAM but memory stays bounded; cancellation/error/exit leaves no job-owned processes, files or buffers in use. |
| Standalone distribution | Windows, Linux and macOS artifacts pass offline clean-system tests without separately installed language/GUI/numerical/codec dependencies. |
| Scientific honesty | Old and corrected results remain distinguishable; invalid cases are not filtered into passes; MFBD claims match the actual gates and real-data evidence. |

The roadmap is complete only when the requested features pass this matrix, not
when Qt starts or Q2 exceeds one threshold. A baseline release may precede the
advanced solver, but must state exactly which planned features remain unfinished.

# 26. Immediate next implementation handoff

**Implementation follow-up (2026-09-05):** commit `cac7837` added W00 manifests
and a bounded W01 audit. Review found and corrected Dykstra dual warm-start and
stopping defects, an even-kernel spatial-oracle offset, unenforced/stale method
provenance, historical-manifest relabelling and a ranking-correlation shortcut.
Estimator operator 1.2 requires feasibility and a qualified projected-gradient
residual; old result tables remain estimator 1.0. Generation identities and
certificates now participate in Gate eligibility, with legacy/unknown identities
requiring explicit revalidation. These small CPU regressions do not complete
full development-family convergence checks or W02/W03 requalification. The
remaining handoff below applies to outstanding work, not a request to repeat
completed unit-level infrastructure.

Implement **W00 and W01 first**, beginning with failing regression tests for the
positivity/support intersection and certification compatibility, then the
forward/adjoint consistency audit. Keep R9 result files unchanged. Add small CPU
fixtures and record operator/version changes. Do not launch Q3 or full family
reruns until those diagnostics establish what must be regenerated. The next
vertical slice is W04–W08: open a real Bayer SER, process a bounded CPU batch,
and display the progressively built result in an owned, cancellable Qt6 job.

**Implementation follow-up (W09):** field rotation and a rigid oblate globe are
composable CPU operators with bilinear adjoints, pose unwrapping, disc/roll
degeneracy flags and geometry-aware baseline accumulation. CFA samples stay on
the detector lattice. Freeze-mid-exposure is the default and warns when limb
motion during \(T_{\rm exp}\) is large. Unseen longitudes are not filled.

**Implementation follow-up (W10):** Saturn is a layered scene, not a disc warp.
An oblate globe and an equatorial ring annulus share field attitude and have
independent radiance; globe spin does not drag the rings. Near-ring / globe /
far-ring depth, transmission, illumination/shadow masks, edge-on degeneracy and
moving-moon masks are implemented on CPU fixtures. Rings are static over a
short clip. Production MFBD with geometry and Q3 remain out of scope here.

**Implementation follow-up (W13):** full-resolution result/checkpoint export now
supports mono/RGB PNG16, TIFF16 and IEEE float32 TIFF, shared explicit integer
mapping, optional labelled display gamma, validity/coverage, provenance and atomic
image publication with immutable generation companions. Independent CPU read-back
and interruption regressions plus a local Linux frozen-codec smoke are implemented.
GUI save controls are implemented by the W14 follow-up below. Clean-system and
cross-platform qualification remain W17. W11 advanced inference still requires W03; GPU qualification requires supported
hardware. W02/W03 remain outstanding and Q3 does not start.

**Implementation follow-up (W14):** Qt now exposes capture, calibration, geometry
and Saturn settings, acknowledged full-resolution snapshots, input/result/coverage
views, stable display mapping and scientific save controls. Save runs independently
of processing and preserves its chosen snapshot. Stale events are rejected;
cancellation/failure retains the last usable image and close owns its workers.
CPU GUI integration, load-heartbeat and local frozen GUI/save smoke checks are
available. RAM/VRAM enforcement, checkpoint resume, parent-crash handling and
large-capture/cross-platform acceptance remain unfinished. W15 is the next
independent application step; W02/W03 and the Q3 gate remain unchanged.

Implementation follow-ups record bounded progress, not complete roadmap certification.

**Implementation follow-up (W15):** native AVI 1.0 DIB RGB24/gray8 decoding,
disk-backed frame indexing, bounded raw batches, pre-read cancellation, safe SER
reads and disk-spooled worker events are implemented. Local tests cover an 8 GiB
sparse SER, truncation/disappearance, rejected AVI formats, exact independent
FFmpeg read-back, bounded queue cleanup and killed worker polling. This is
partial W15 qualification: compressed/OpenDML formats, hard working-set limits,
resumable checkpoints and parent-crash cleanup remain unsupported. No external
codec is needed for the declared formats.
