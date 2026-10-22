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
   fewer so each bin keeps at least 20 frames), and each bin into even and odd
   frames. Every (bin, parity) has its own accumulator, filled with the
   ordinary per-pixel blended weights; outside the points the global quality
   rank decides the bin.
2. After stacking, each bin-stack is measured on narrow rings of radial
   frequency (0.01 cycles/px, cos^2 transitions, a partition of unity) inside
   a tapered window over the target:
   - transfer T: the cross-spectrum against the stack of all other bins over
     that reference's power. The noise of the two is independent, so T has no
     noise bias.
   - incoherent residual V: from the difference of the bin's even and odd
     halves, which cancels all common signal whatever its blur and leaves noise
     plus random distortion from imperfect alignment (larger in poor frames).
3. Image weights T^p / V are the inverse-variance matched filter for p = 1:
   the best signal-to-noise at each ring, which is what sharpening exploits.
   T is smoothed across rings and made non-increasing over rank; V is smoothed
   across rings and fitted as a quadratic in log over rank (its bin-to-bin
   scatter is estimation noise). `--fourier-power` (default 1) raises T to a
   higher power to favour sharpness over noise.
4. Each ring's weighted stack is formed from the accumulators (per-pixel
   normalisation as usual) and the image is the ordinary stack plus each
   ring's filtered difference from it. Equal weights reproduce the ordinary
   stack exactly.
5. The even and odd halves give each result's signal-to-noise per ring from
   their Fourier ring correlation (SNR = 2 FRC / (1 - FRC) for the full
   stack); both the Fourier and the ordinary values are recorded in the
   provenance (`fourier_weighting`).

Previews during stacking show ordinary weights; the weighting is applied when
the stack completes. Checkpoints keep the bin accumulators and resume exactly.
