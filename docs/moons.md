# Moons

**Stack moons separately** (`--moon-stacking`, `ReconstructionConfig(moon_stacking=True)`;
on by default) finds moons clear of the planet's glow, tracks them and stacks each on its
own position. It applies to ordinary stacking, with or without alignment points. It does
not apply with planet rotation models (field, surface, combined, Saturn), where the result
provenance records `moons.skipped`.

## Why moons blur when registered with the planet

On 2026-11-21-1018_9-CK-IR642-Sat (16" at f/12, 0.11"/px, 360 s, 175 fps) Rhea, Dione,
Tethys and Enceladus lie 30-45" from Saturn. Two things move them against the planet:

- **Seeing tilt.** They are far outside the patch of sky whose tilt the planet's
  registration removes, so each frame tilts them by its own amount. Measured with their
  own centroids, the bright moons scatter about the planet by 2.6-2.9 px rms per axis
  (0.3"); frames 70 ms apart are uncorrelated, and the planet's frame quality does not
  predict a moon's tilt.
- **Orbital drift.** 0.6-1.1 px per 3 minutes for these moons relative to Saturn. Over the
  6-minute capture this is smaller than the tilt; it grows with capture length and is
  larger for inner moons near conjunction (Mimas can move several pixels in 3 minutes).

Registered with the planet, the moons stacked to 10.9-11.9 px FWHM. Stacking them on their
own positions gives about 5.0-5.5 px.

## Method

1. **Finding moons.** Up to 400 frames spread through the selection (the global frame
   selection when alignment points rank frames locally) are registered with the planet's
   global shift and averaged. Compact sources at least 10 times the sky noise are
   candidates. A source is left with the planet's registration when its stacking window
   reaches the planet's glow (a smooth background more than 2% of the planet's brightness
   above the sky; this covers the planet, the rings and moons such as Mimas lost in their
   glow), when it lies within reach of an alignment point, or when its window would leave
   the field covered by every calibration frame. The provenance lists each excluded
   source and why.
2. **Measuring positions.** In each frame a moon's position is a Gaussian-weighted
   centroid (weight sigma a quarter of its planet-registered FWHM) with a plane fitted to
   the window border removed, so sky and glow gradients do not pull it.
3. **Drift.** A robust straight line (a quadratic for captures of 10 minutes or more)
   through the calibration positions relative to the planet gives each moon's drift and
   its position at the output epoch (the capture midpoint). A source whose positions on
   the detector vary less than half as much as its positions relative to the planet is
   fixed to the sensor (a hot pixel or defect left by the planet-registered average) and
   is not stacked.
4. **Noise.** A centroid strays on noise alone, and on a faint moon it strays about as
   far as the seeing moves it. Following those positions would shift each frame until its
   noise lines up, sharpening the noise into a false point (the effect described in
   [alignment points](alignment-points.md)). The noise is measured directly: in every
   calibration frame a copy of the brightest moon's profile (its own stack on its measured
   positions), scaled to each moon's flux, is placed in clear sky and found again from a
   start as far off as the moons scatter. Without clear sky the noise comes from the
   difference of centroids on the two halves of a checkerboard (which share the seeing but
   not the noise), or the whole scatter is treated as noise.
5. **Shared tilt.** Nearby moons share part of their tilt. Each moon's deviation from its
   drift is regressed on the deviations of the moons whose seeing exceeds their noise. A
   neighbour's centroid noise is independent of this moon's pixels, so the shared part
   costs no noise imprinting. On the sample, Tethys's tilt predicts 0.8 px rms of
   Enceladus's, 2.6" away.
6. **Own tilt.** What the neighbours do not predict is followed with the share
   k = s^2 / (s^2 + n^2), where s^2 is the moon's remaining seeing variance and n^2 its
   noise variance. Bright moons get k of about 0.97-0.99; a moon whose centroid noise
   exceeds its seeing gets k near 0 and follows its drift and its neighbours alone.
   Positions further than four standard deviations from the prediction are ignored.
7. **Stacking.** For every frame stacked, each moon's window (radius twice its
   planet-registered FWHM, at least 12 px) is sampled bilinearly at the moon's estimated
   position and accumulated with the frame's weight, the same quality weight the stack
   uses outside the alignment points. Bayer captures accumulate the raw CFA samples by
   colour. Frames whose window leaves the detector are skipped for that moon.
8. **Compositing.** Each moon's stack replaces the planet-registered moon at its output
   position, fully within 70% of the window radius and feathered to nothing at 95%. Where
   windows overlap, each pixel goes to the nearer moon, handing over smoothly along the
   line halfway between them. Coverage, validity and everything outside the windows are
   those of the ordinary stack; total moon flux is unchanged.

Moon definitions and window sums are part of the stack checkpoint, so a resumed stack
continues the same moons. Checkpoints written before moon stacking existed do not resume.

## Provenance

`moons` in the result provenance records the calibration frame count, output epoch, the
excluded sources and those too faint to track, and for each stacked moon: where it was
detected and its detection signal-to-noise ratio, planet-registered and stacked FWHM,
drift (px per minute, y then x), seeing jitter, centroid noise and how it was measured,
the tilt shared with neighbours, the gain k, its output position, and how many frames
followed its measured tilt or its prediction alone.

## Validation

2026-11-21-1018_9-CK-IR642-Sat, all 57,145 frames selected at 85% quality range with
alignment points and Fourier weighting (CUDA), against the same stack without moon
stacking:

| Moon | FWHM with planet | FWHM on own positions | Peak | Gain k | Centroid noise | Shared tilt |
|---|---|---|---|---|---|---|
| Rhea | 11.6 px | 5.2 px | 1.97x | 0.99 | 0.23 px | 0.8 px |
| Tethys | 12.2 px | 5.1 px | 2.26x | 0.98 | 0.39 px | 1.6 px |
| Dione | 12.3 px | 5.2 px | 2.40x | 0.97 | 0.44 px | 1.6 px |
| Enceladus | 11.6 px | about the same | 1.19x | 0 | 3.0 px | 1.1 px |

Flux within 10 px of each bright moon changes by 1-2%, and the image outside the moon
windows is identical to the ordinary stack. Enceladus's own centroid is all noise (3.0 px
against a scatter no larger), so it follows its drift and the tilt it shares with Tethys
2.6" away: a modest gain, and no false core. Mimas, over the rings beside the globe, keeps
the planet's alignment. On a 15,780-frame subset the bright moons stacked to 5.1-5.5 px,
and moon stacking added about 8% to the run time (three passes over 400 frames, then a
centroid and a small window per moon per frame). On Bayer captures with alignment points
the centroids originally used a CPU demosaic of every frame (40 ms per frame on a 1428x574
capture); they now use windows of the registration plane already on the GPU, copied to the
host together, and each moon's CFA window is sampled directly (45 us, from 320 us).
