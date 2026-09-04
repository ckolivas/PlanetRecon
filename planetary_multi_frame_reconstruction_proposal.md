# Planetary Multi-Frame Atmospheric Reconstruction
## Implementation specification — revision R9

**Status:** Gate-1 implementation freeze  
**Revision:** R9 — 2026-09-04
**Intended outcome:** Prompt 1 and Prompt 2 can now be handed to Codex without reopening the research tree  
**Primary application:** High-frame-rate monochrome planetary SER/AVI sequences

This is MOMFBD / short-exposure inverse imaging in an amateur-planetary regime, tested against lucky-imaging architectures. It is not a new inverse problem.

R9 accepts the R8 science tree and stop rules. It adds **no new research branch**. It corrects the bland-crop geometry contradiction exposed by Prompt 1 and retains the R8 implementation clarifications:

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

No future revision should add a research branch unless implementation exposes a contradiction in the locked physics or mathematics.

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

# 17. Prompt 2 scope — frozen but not yet full text

Prompt 2 may begin only after Prompt 1 passes.

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

before tiles or spherical rotation.

Primary engineering metric:

> practical reconstruction gain per GPU-hour.

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

# 21. Convergence status

R9 freezes:

- the primary claim;
- two mandatory seeing regimes;
- timing;
- source flux scaling;
- object generator;
- crop geometry requirements;
- practical ranking;
- decision percentile;
- primary metric;
- gap definitions;
- seed logic;
- pooled-seed thresholds;
- oracle controls;
- simulator convergence criteria;
- truth-file schema;
- Prompt 1.

There are **no open research-branch questions before Prompt 1**.

The next useful action is implementation.

A future revision is justified only if:

1. Prompt 1 exposes a numerical/physical contradiction in R9; or
2. a reviewer identifies a concrete flaw that would systematically bias G1/G2.

Otherwise revisions should stop and the programme should move to code.
