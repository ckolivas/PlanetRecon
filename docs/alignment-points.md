# AutoStakkert-style alignment points

Choose **Alignment points → AutoStakkert-style alignment points** on the Capture
tab (CLI `--local-alignment --alignment-method alignment_points`), then **Stack**
with Motion model **None**. Settings:

- **Alignment point size** (`--ap-size`, default 64 px). Points are placed every
  half size on regions whose reference mean is at least 10% of the reference
  maximum and whose band-passed texture is at least 0.1 of the median point.
- **Local quality** is always used. Each point ranks every screened frame by
  its own sharpness and stacks its best N frames, where N is the number of
  frames the global quality-range/count selection keeps. Point selections are
  blended with normalised Gaussian footprints (sigma 0.6 x spacing); the global
  selection applies outside the points. Quality weighting still applies to the
  contributing frames. When every screened frame is selected (for example
  100%), each point would use every frame, so ranking is skipped; the result is
  unchanged.

There is no option to disable local quality or narrow its candidates: both
were measured to lose detail (below), so they would trade quality for speed.

The method:

1. A reference is averaged from up to 3000 of the best selected frames after
   global alignment, then averaged again after aligning those frames with
   alignment points to the first mean. The origin stays anchored to the best
   frame. (AutoStakkert!4 builds its reference from thousands of frames as
   well; the Saturn R session used 6,183.)
2. Each frame is registered globally, then every point searches +/-6 px for its
   normalised cross-correlation peak on the band-passed proxy (Gaussian 1.5 px
   minus Gaussian 8 px), refined with a two-dimensional quadratic. A point is
   accepted if the peak is interior, NCC >= 0.5 and the curvature is positive.
3. The dense field is a per-pixel weighted least-squares fit of the accepted
   shifts. Each point is weighted by its peak curvature (information matrix), so
   a point on a ring edge or the limb constrains only the direction across the
   edge. A weak prior (0.01 x median curvature) returns the field to global
   motion away from points. Accepted shifts are applied at full strength.
4. The raw detector samples are resampled once with that field (unchanged
   bilinear backprojection, including raw CFA support).

CUDA runs the correlations, field solve, proxy filtering, local quality and
reference resampling; peak decisions use the shared CPU code. CPU and CUDA
fields agree to 1e-9 px in the tests. Checkpoints bind the point size.

## With a motion model (Surface, Combined)

Alignment points combine with the Surface and Combined motion models; Field
and Saturn are refused (use circular or square alignment for them).

1. **Reference.** Up to 3000 of the best selected frames are predicted into the
   anchor frame's geometry by the globe model, including each frame's measured
   camera drift, and averaged; that derotated mean is refined once with
   alignment points.
2. **Matching.** For each frame the derotated reference is predicted at the
   frame's own time and pose, and an alignment-point matcher is built on it,
   using only points whose whole search footprint is observed (newly rotated-in
   longitudes are excluded). The measured seeing field is in frame coordinates,
   so it is inverted directly; composing a field through the model instead
   fails at the limb, where the model's mapping is discontinuous.
3. **Stacking.** Each raw sample is scattered once through the model into the
   reference-epoch geometry, removing rotation and seeing in one resampling.
4. **Local quality.** Points on the derotated reference are ranked in every
   screened frame at their predicted positions in that frame; the blend weights
   sit at the points' reference-epoch positions and apply at each destination
   pixel of the scatter.

Jupiter (75 s capture, about 2.5 px of rotation at the disc centre over the
capture), 63%, independent half-stack SNR:

| method | 0.04–0.08 | 0.08–0.12 | 0.12–0.16 | 0.16–0.20 | 0.20–0.25 | 0.25–0.30 |
|---|---|---|---|---|---|---|
| None + circular | 633 | 100 | 15.7 | 5.1 | 1.26 | 0.15 |
| Surface + circular | 1,030 | 133 | 15.3 | 3.8 | 0.94 | 0.13 |
| None + alignment points | 828 | 133 | 20.4 | 7.1 | 1.86 | 0.15 |
| Surface + alignment points, no local quality | 1,020 | 141 | 20.9 | 7.3 | 1.79 | 0.19 |
| Surface + alignment points | 983 | 145 | 21.1 | 7.3 | 1.83 | 0.20 |

Surface derotation improves the coarse scales (+19–23% at 0.04–0.08 over None
with alignment points) and the finest band (+29% at 0.25–0.30); between 0.12
and 0.25 cycles/px the two are within a few percent. On this short capture
local quality adds nothing measurable with Surface. Surface processing is much
slower: about 0.22 s per frame (drift tracking about 80 ms, a matcher built on
each frame's predicted reference, and the scatter) against about 0.02 s per
frame with Motion model None.

## Why the earlier stacks were poor

The investigation compared PlanetRecon with the AutoStakkert!4 stacks in the
`AS_P*` folders and with the AS4 session file.

- **Frame selection was not the cause.** AS4's per-frame qualities for Saturn R
  (from its session file) rank-correlate with PlanetRecon's preprocessing
  scores at Spearman 0.997; 89% of the best 21% coincide.
- **The local matcher discarded most local motion.** On real frames only
  15–28% of circular-matcher points passed its gates at the finest scale,
  even with median NCC 0.98 on Saturn: 27–38% had their peak on the edge of the
  +/-3 px search (the points with the largest seeing motion), 15–34% had
  NCC < 0.8, and 21–36% failed the 0.005 peak-uniqueness margin, which rejects
  the broad, smooth correlation peaks typical of planetary detail. Accepted
  corrections were also shrunk to 45–70% of the measured value on average by
  the confidence normalisation. Uncorrected seeing motion blurs the stack.
- **The 64-frame template was too noisy.** Stronger local alignment against it
  imprints its noise: half-stack differences showed coherent blotches on the
  low-contrast globe and limb until the reference used thousands of frames.
- **Global-only selection.** AS4 ranks frames per alignment point.

AS4's unsharpened output also looks smoother for a second, separate reason.
On Saturn R its fine-scale power is 150–900x below the noise floor that
averaging its ~5,800 frames must leave, while PlanetRecon's matched the
prediction from raw frame differences exactly:

| band (cycles/px) | predicted floor, N≈5,700 + bilinear | PlanetRecon | AS4 |
|---|---|---|---|
| 0.30–0.36 | 80 | 85 | 1.9 |
| 0.36–0.42 | 61 | 61 | 0.38 |
| 0.42–0.50 | 43 | 50 | 0.056 |

Motion cannot remove noise below that floor, so AS4's pipeline applies some
low-pass smoothing (most likely a side effect of its resampling or blending).
That changes how a stack looks before sharpening, not the recoverable detail,
and it is not reproduced here.

## Measurements

Recoverable detail is measured as per-band SNR from two fully independent half
stacks (alternate selected frames, each with its own reference and matcher,
anchored to adjacent frames so a rotating planet shares one geometry; halves
densely co-registered before comparison). SNR is invariant to linear filtering,
so interpolation or smoothing differences cannot flatter a method. Values are
for the whole stack; higher is better. The prototype used for these runs is in
`tools/ap_evaluation/` (`xstack.py`, `snr.py --dense`).

Saturn R (mono), 21% of frames (5,350):

| method | 0.08–0.12 | 0.12–0.16 | 0.16–0.20 | 0.20–0.25 | 0.25–0.30 |
|---|---|---|---|---|---|
| circular multiscale (previous) | 2,030 | 233 | 74 | 6.6 | 0.51 |
| alignment points, 3,000-frame reference | 2,040 | 289 | 84 | 9.3 | 0.48 |
| + local quality (equal weights) | 2,640 | 438 | 137 | 12.0 | 0.65 |

Saturn, colour (pipp), 63% (10,498 frames):

| method | 0.08–0.12 | 0.12–0.16 | 0.16–0.20 | 0.20–0.25 | 0.25–0.30 |
|---|---|---|---|---|---|
| circular multiscale | 1,550 | 134 | 35.3 | 10.0 | 1.02 |
| alignment points, 3,000-frame reference | 2,210 | 162 | 39.5 | 12.0 | 1.22 |

Jupiter (colour), 63% (2,105 frames):

| method | 0.08–0.12 | 0.12–0.16 | 0.16–0.20 | 0.20–0.25 |
|---|---|---|---|---|
| circular multiscale | 111 | 14.5 | 4.9 | 1.18 |
| alignment points (48 px, 64-frame reference) | 114 | 18.9 | 6.8 | 1.75 |
| 64 px, 1,000-frame reference + local quality | 133 | 20.5 | 7.2 | 1.87 |

Mars IR642, 14% (14,944 frames):

| method | 0.04–0.08 | 0.08–0.12 | 0.12–0.16 | 0.16–0.20 | 0.20–0.25 |
|---|---|---|---|---|---|
| circular multiscale | 4,820 | 232 | 69 | 21.3 | 7.3 |
| alignment points, 3,000-frame reference | 14,000 | 918 | 190 | 32.6 | 6.7 |

Further checks:

- Restricting local quality to the globally best frames loses detail. On
  Saturn R at 21% (equal weights; SNR at 0.12–0.16 / 0.16–0.20 / 0.20–0.25
  cycles/px): no local quality 289 / 84 / 9.3 from 5,350 frames; candidates
  limited to 1.5x the selection 378 / 107 / 11.8; 2x 399 / 115 / 13.2; 3x 413 /
  119 / 14.1; every screened frame 438 / 137 / 12.0 from about 24,000 frames.
- Linear quality weights beat equal weights by 3–4% both with and without
  local quality, so quality weighting remains on.
- 48 px and 64 px points perform within a few percent; 64 px is the default
  (AS4's smallest grid size in the Saturn R session).
- A synthetic capture with known truth (tip/tilt, smooth local distortion,
  varying blur, Poisson and read noise, 8-bit) confirms a smaller gain for
  alignment points over circular (SNR +10–14% at 0.08–0.20 cycles/px); the
  synthetic seeing is milder than the real captures.
- Visually, with identical wavelet sharpening, the alignment-point stacks with
  local quality are much closer to the AS4 stacks than the previous method:
  sharper ring edges, limb and belt structure, and far less mottling on Saturn R.

## Cost and progress

With CUDA the whole per-frame path stays on the device: registration-plane
demosaic, global correlation (only the peak stencil is copied back), proxy
filtering, point correlations, field solve, local quality, backprojection and
the weighted sums. Sums are merged into the authoritative CPU accumulators at
the end of every batch; if CUDA fails, unmerged frames are recomputed on CPU.
The global shifts measured while ranking local quality are reused for stacking.

The setup phases report progress: **Building alignment reference** (two passes
over up to 3000 frames) and **Ranking frames by local quality** (every
screened frame). Local quality then stacks nearly every screened frame, because
each point keeps its own best frames.

Measured on the RTX 5070 for a 47,708-frame 1124x466 colour Saturn capture
(about 45,000 screened, 50% quality range): reference 56 s, local-quality ranking
236 s (5.2 ms/frame), stacking about 14 ms/frame, roughly 15 minutes in total
(before ranking shifts were reused for stacking).
Before these
optimisations the same capture needed about 36 ms/frame for ranking and 80 ms/frame
for stacking, mostly CPU demosaicing and CPU accumulation.
