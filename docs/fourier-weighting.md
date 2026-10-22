# Fourier (frequency-domain) weighting

`--fourier-weighting` (desktop: **Fourier (frequency-domain) weighting**) weights
frames separately at every spatial frequency, after Garrel, Guyon & Baudoz
(2012, PASP 124, 861): coarse detail draws on nearly every frame, while the
finest detail comes mostly from the sharpest. It needs alignment points. The
stack percentage still limits which frames each point uses; with 100% every
screened frame can contribute, and the weighting decides how much at each scale.

## Why not weight each frame by its own spectrum

Garrel et al. weight every frame by its own Fourier amplitude. In
seeing-limited planetary video that amplitude cannot be measured: on 2000
frames of a Saturn L3 capture (single-frame SNR about 7.6), in 64 px
alignment-point tiles, the per-frame signal power above 0.08 cycles/px was
below the noise power, and averaging 200 frames per quality decile showed no
detectable dependence on quality above 0.12 cycles/px (signal at about 0.5%
of the noise power, below what a noise model can resolve). The stacks still
hold that detail (20,000 frames x 0.5% is about 100x the noise), so it has to be
measured after stacking, not per frame. Weighting by noisy per-frame
amplitudes would weight noise.

## Method

1. Each alignment point already ranks every screened frame by its local
   quality. The frames it stacks are split into equal-count rank bins (16, or
   fewer so each bin keeps at least 20 frames), and each bin into two halves
   of alternate 32-frame blocks (frames a few ms apart share seeing, so they
   stay in the same half). Every (bin, parity) has its own accumulator, filled with the
   ordinary per-pixel blended weights; outside the points the global quality
   rank decides the bin.
2. After stacking, each bin-stack is measured on narrow rings of radial
   frequency (0.01 cycles/px, cos^2 transitions, a partition of unity) inside
   a tapered window over the target:
   - transfer T: the cross-spectrum against the stack of all other bins over
     that reference's power. The noise of the two is independent, so T has no
     noise bias.
   - incoherent residual V: from the difference of the bin's two halves, which cancels all common signal whatever its blur and leaves noise
     plus random distortion from imperfect alignment (larger in poor frames).
3. Image weights T^p / V are the inverse-variance matched filter for p = 1:
   the best signal-to-noise at each ring, which is what sharpening exploits.
   T is smoothed across rings and made non-increasing over rank; V is smoothed
   across rings and fitted as a quadratic in log over rank (its bin-to-bin
   scatter is estimation noise). `--fourier-power` (default 4) raises T to a
   higher power to favour sharpness over noise; see the results below.
4. Each ring's weighted stack is formed from the accumulators (per-pixel
   normalisation as usual) and the image is the ordinary stack plus each
   ring's filtered difference from it. Equal weights reproduce the ordinary
   stack exactly.
5. The two halves give each result's signal-to-noise per ring from
   their Fourier ring correlation (SNR = 2 FRC / (1 - FRC) for the full
   stack); both the Fourier and the ordinary values are recorded in the
   provenance (`fourier_weighting`).

Previews during stacking show ordinary weights; the weighting is applied when
the stack completes. Checkpoints keep the bin accumulators and resume exactly.

With a Geometry motion model the same bins are accumulated in the
motion-compensated reference geometry: each point ranks frames at its
predicted position in that frame, and every frame is projected once, its
samples scattered into the ordinary sums and then into its rank-bin sums with
the bin weights (the fused CUDA scatter reuses the projected coordinates).
Halves alternate in 32-frame blocks across the whole capture, so both cover
the full rotation. Colour completion is not limited by globe and ring regions,
so recombining bins needs no region handling.

## Results: Saturn L3, 2026-10-17 12:33.1 (44,440 screened frames)

One stack at 100% with 16 rank bins (GPU, 23 min), re-weighted offline from
the saved accumulators. "Current" emulates the ordinary 50% quality-range
stack: each point's best 7 of 16 bins (44% of frames, the share the 50%
quality range kept), quality weighted. Ring SNR from the halves' Fourier ring
correlation, as a ratio to Current:

| cycles/px      | 0.08 | 0.10 | 0.12 | 0.15 | 0.18 | 0.20 | 0.23 | 0.25 | 0.28 |
|----------------|------|------|------|------|------|------|------|------|------|
| ordinary 100%  | 0.97 | 0.91 | 0.90 | 1.03 | 1.17 | 1.16 | 1.09 | 1.11 | 0.99 |
| Fourier p = 1  | 1.02 | 1.03 | 1.09 | 1.12 | 1.21 | 1.21 | 1.14 | 1.15 | 1.00 |
| Fourier p = 2  | 1.08 | 1.08 | 1.19 | 1.17 | 1.21 | 1.26 | 1.22 | 1.21 | 1.05 |
| Fourier p = 4  | 0.94 | 0.83 | 0.89 | 0.89 | 0.96 | 1.12 | 1.18 | 1.24 | 1.12 |
| Fourier p = 6  | 0.70 | 0.66 | 0.72 | 0.68 | 0.70 | 0.88 | 0.98 | 1.17 | 1.17 |
| Fourier p = 8  | 0.57 | 0.60 | 0.66 | 0.60 | 0.57 | 0.71 | 0.78 | 1.05 | 1.20 |

The stack's SNR falls to about 2 at 0.25 cycles/px. Each version was sharpened
by eye: p = 4 looked best; 1 and 2 were softer; 6 and 8 were visibly noisier
with only slightly sharper Cassini division. The finest rings, where p = 4
leads, decide the visible result, so the default is 4. Above about 0.25
cycles/px a single bin-stack is too noisy for its transfer to be measured,
and the weights there drift back towards equal.

## Results: all six Saturn L3 captures, 2026-10-17

Fourier p = 4 at 100% against each capture's own former 50% quality-range
stack (emulated by the best bins holding that capture's retained share), ring
SNR ratio from the halves' Fourier ring correlation:

| capture | 50% range kept | 0.08 | 0.10 | 0.12 | 0.15 | 0.18 | 0.20 | 0.23 | 0.25 | SNR at 0.25 |
|---------|----------------|------|------|------|------|------|------|------|------|-------------|
| 12:20.6 | 32%            | 1.06 | 1.03 | 1.00 | 1.03 | 1.08 | 1.05 | 1.10 | 1.20 | 4.8 |
| 12:26.6 | 28%            | 1.11 | 0.99 | 0.93 | 1.02 | 1.07 | 1.04 | 1.12 | 1.22 | 2.6 |
| 12:33.1 | 44%            | 0.94 | 0.83 | 0.89 | 0.89 | 0.96 | 1.12 | 1.18 | 1.24 | 2.0 |
| 12:39.9 | 47%            | 1.03 | 0.97 | 1.21 | 1.27 | 1.36 | 1.39 | 1.23 | 1.21 | 1.6 |
| 12:46.6 | 22%            | 1.33 | 1.19 | 1.38 | 1.52 | 1.51 | 1.49 | 1.66 | 1.14 | 0.8 |
| 12:52.8 | 20%            | 1.43 | 1.37 | 1.80 | 1.99 | 1.92 | 1.90 | 1.51 | 1.89 | 1.1 |

The finest rings gain 1.1-1.9x on every capture; the gain is largest where the
50% quality range kept fewest frames, and p = 4 trades a little mid-frequency
SNR on the capture where it kept most (12:33.1). Sharpened with the same
Planetary Tools workflow (automatic amounts to a target noise), p = 4 was the
clear winner by eye on 12:20.6, 12:26.6 and 12:33.1. Automatic detail
measures did not reproduce those judgements: a 2x-binned gradient energy that
matches AutoStakkert!4's quality values preferred the 50% stacks, and a
fine-scale Laplacian preferred p = 6-8, which looked noisier.

### With the Saturn motion model (12:33.1)

Stacked at 100% with the Saturn model (geometry from preprocessing, Saturn
rotation preset, latitude from SER UTC; GPU, 34 min). Ring SNR ratio to the
emulated 50% stack from the same run:

| cycles/px     | 0.08 | 0.10 | 0.12 | 0.15 | 0.18 | 0.20 | 0.23 | 0.25 |
|---------------|------|------|------|------|------|------|------|------|
| Fourier p = 1 | 1.01 | 1.07 | 1.04 | 1.18 | 1.27 | 1.10 | 1.10 | 1.15 |
| Fourier p = 2 | 1.04 | 1.11 | 1.14 | 1.24 | 1.26 | 1.15 | 1.16 | 1.19 |
| Fourier p = 4 | 0.90 | 0.87 | 0.91 | 0.99 | 1.00 | 1.03 | 1.16 | 1.22 |

The same pattern as without a motion model. On this capture the absolute
ring SNR with and without the Saturn model is similar (p = 4 at 0.25
cycles/px: 2.08 and 2.02): alignment points already follow most of the
rotation over the 6-minute capture.
