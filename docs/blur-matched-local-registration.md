# Estimating reference blur before local alignment

The regional-gate experiment did not consistently improve the result. This
experiment changes the matching reference instead: estimate how much additional
blur matches a frame, then measure local motion against that reference. It does
not change production registration, frame brightness, frame selection, output
sampling or stack filtering.

## What is new

The earlier [known-blur experiment](local-warp-validation.md) supplied the
simulated blur to the production matcher and rebuilt its AP eligibility. Here
blur is inferred from each observation and the spatially coherent estimator is
used. AP centres, eligibility, spline design, response kernels and spatial
support are fixed at the original reference. Only reference proxies, centred
matching templates and their strengths change, including reverse matching.

The additional Gaussian blur bank is sigma 0 through 2 pixels in steps of 0.25.
Reference predictions are rendered into sensor coordinates; the observed proxy
is fixed when scoring candidates. A positive gain between 0.5 and 2 and an offset
are fitted only to the score, never applied to the original detector samples.
The smallest residual sum of squares selects a discrete blur value. No mixture,
output smoothing, temporal mean subtraction or extra accept/reject gate is used.

Two automatic policies use the same bank:

- **Global pose:** estimate blur with only the supplied global translation.
- **Local pose:** estimate blur after the original coherent fit, then recompute
  the coherent fit using the chosen template.

Synthetic controls also include the exact known blur as an oracle diagnostic.
The final field still uses the existing bending penalty, robust AP fit and
geometric guards. In particular, neither automatic policy guarantees zero
motion in noisy stationary frames.

## Blur-only diagnosis

Thirty stationary frames cover Saturn and a textured disc, blur sigma
0/0.5/1/1.5/2, and either no noise or independent Gaussian noise with sigma 2 ADU
at seeds 9017 and 9018. Noise magnitude is illustrative; it is not a measured
sensor-noise model for this capture.

On the Saturn frames with no added noise, RMS displacement over the fixed planet mask is:

| Additional blur sigma px | Original coherent | Global-pose blur match | Local-pose blur match |
|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 |
| 0.5 | 0.08743 | 0 | 0.08737 |
| 1.0 | 0.20169 | 0 | 0.12578 |
| 1.5 | 0.39371 | 0 | 0.19402 |
| 2.0 | 0.66419 | 0 | 0.28172 |

There is no true motion or added noise here; the fixed reference, including any
residual structure in it, serves as synthetic truth. The original matcher interprets
changes in the blurred structure as displacement. At the correct global pose,
blur selection recovers the supplied value and removes this false motion.
After the biased local fit, it selects blur values 0.25/0.75/1.25/1.75 instead:
the warp has already explained away part of the blur. This directly demonstrates
confounding between motion and blur in this model.

The noiseless textured scene also acquires false motion (0.10669 px RMS at sigma
2), and both automatic policies remove it there. Noise remains a separate
problem: in Saturn's sigma-1.5, seed-9017 frame, matching the exact blur reduces
displacement RMS from 0.77700 to 0.49864 px, but does not make it zero. The second
noise realization changes from 0.71950 to 0.56202 px.

## Known-motion stacks

The established four cases, two scenes and two noise seeds give 768 frames.
Clean-image RMSE below compares stacks of the clean twins, warped using the
displacements estimated from their noisy counterparts, against the known-motion
oracle. All values are input ADU.

| Saturn case | Seed | Coherent | Global-pose blur | Local-pose blur | Known blur |
|---|---:|---:|---:|---:|---:|
| Stationary with changing blur | 9017 | 0.242217 | 0.196927 | 0.189423 | 0.187598 |
| Stationary with changing blur | 9018 | 0.269778 | 0.259020 | 0.246409 | 0.245216 |
| Motion and blur | 9017 | 0.241927 | 0.180845 | 0.160489 | 0.152608 |
| Motion and blur | 9018 | 0.253322 | 0.243942 | 0.204655 | 0.219376 |
| Offset motion and blur | 9017 | 0.262709 | 0.242038 | 0.211321 | 0.224841 |
| Offset motion and blur | 9018 | 0.294940 | 0.313898 | 0.289141 | 0.270333 |

Local-pose blur matching improves all six clean-image comparisons, although it
does not eliminate false stationary motion. Global-pose matching regresses the
offset-motion repeat. On the textured scene, local-pose matching reduces moving
clean-image error by about 23–32%; global-pose matching regresses both offset
cases. With no extra blur, changes are negligible and neither policy fixes the
underlying noisy stationary-frame errors.

The measured blur itself is confounded by motion. In the primary moving Saturn
case, blur-sigma error is 0.336 px RMS at the global pose and 0.167 px at the
local pose; in the offset case it is 0.433 versus 0.143 px. Thus the method that
best identifies pure blur in stationary noiseless images is not always the best
one when genuine motion is present.

Exact sharpening gives much smaller, mixed changes. Local-pose matching changes
Saturn's moving-case error from 0.021594 to 0.021497 at the first seed and from
0.021685 to 0.021577 at the second, but changes the offset-motion repeat from
0.022237 to 0.022509. These are errors in linear 0–1 image units against the
identically sharpened noiseless oracle; they include structure and artifacts.

The clean/noisy twins also allow an exact raw-error decomposition: total error
equals clean geometry error plus the noise carried through the selected warp.
In the primary moving Saturn case, that noise component has RMS 0.19638 ADU for
coherent and 0.19543 ADU for local-pose matching. The large clean-geometry gain
therefore does not provide a corresponding reduction in the stacked noise
component. This measurement is possible in simulation; the real capture does
not provide a noiseless twin.

## Blur-range sensitivity

A separate CPU score-only probe expands the global-pose bank to sigma 4 for 96
evenly selected capture frames. Twenty-three select the original upper value
of 2, but only one prefers a value beyond it: sigma 2.25 at selection position
2,475, reducing fitting loss by 0.139%. None selects 4. This check does not
change the replay parameters or create an additional stack. Its original-bank
choices are checked against the CUDA replay for all 96 positions.

## Full 5,738-frame result

The unchanged coherent baseline reproduces the previous full replay to
1.14e-13 ADU; coverage agrees within 1e-10. All 96 CPU range-probe selections
agree exactly with the corresponding CUDA choices. Every selected frame is
included once per method, with the original scalar weights and export mapping.

After the exact sharpening recipe:

| Method | Upper-disc variation % | Lower-disc variation % | Left ring width px | Right ring width px |
|---|---:|---:|---:|---:|
| Coherent baseline | 2.62672 | 2.73750 | 7.39574 | 6.04747 |
| Global-pose blur match | 2.72946 | 3.02445 | 7.48988 | 5.99245 |
| Local-pose blur match | 2.82094 | 2.91529 | 7.45078 | 6.12067 |

Variation is robust high-pass MAD divided by the local median, not isolated
sensor noise. Compared with coherent, global-pose matching raises it by 3.9%
and 10.5%; local-pose matching raises it by 7.4% and 6.5%.

The fixed-window right-ring width looks better for global-pose matching, but
that result does not hold across the common-centroid sensitivity windows.
Across 75 overlapping windows on each side, global-pose matching broadens the
median left/right transition by 0.177/0.077 px, with wider transitions in
93%/80% of windows. Local-pose matching broadens them by 0.117/0.068 px, with
wider transitions in 93% on both sides. These windows are sensitivity checks,
not independent statistical replicates or calibrated resolution estimates.

The global-pose selector chooses sigma 2 on 1,742 of 5,738 frames (30.4%); the
local-pose selector does so on 515 (9.0%). The separate wider-bank probe above
qualifies what reaching that boundary means; it rarely prefers a larger value
in the sampled frames. It cannot establish that the restricted blur model is
physically correct.

For context, the original user-sharpened PR image measures 2.96676/2.97506%
upper/lower variation; the supplied AS manual-64 image measures
0.57321/0.64187%. Those files use their own references and transfer responses,
so they are not a controlled estimator comparison. The large original gap
clearly remains.

**Decision:** retain this as a diagnostic and do not promote either policy.
Blur can produce false local motion without any added noise; template matching
can reduce that error on controlled images. Sequential blur and motion fits
remain coupled, noisy stationary frames still acquire false shifts, and the
full real result regresses. The evidence points toward testing a joint
blur/motion fit with explicit uncertainty in weak patches, rather than tuning
the order of these two fits or adding another accept/reject rule.

Local review artifacts are `out/saturn-blur-matched-full/comparison.png`, the
three raw PNG/NPZ stacks, exact `sharpened/` outputs and `analysis.json`.

## Reproduction and verification

```sh
PYTHONPATH=. OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python tools/blur_bias_probe.py --out out/blur-bias-probe.json

PYTHONPATH=. OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python tools/blur_matched_controls.py \
  --out out/blur-matched-saturn-controls --scenes saturn
```

Use `--scenes texture` for the second scene and `--seed 9018` for the repeat.
The four cases each have 48 frames and reuse the established clean/noisy twins,
motion, global translations and weights. Run the existing PlanetaryTools
sharpening harness on each output's PNGs, then
`tools/analyse_blur_matched_controls.py --out OUTPUT_DIRECTORY`.

The full replay uses `tools/refit_saturn_experiment.py --blur-matched
--sample-count 5738`, disjoint `--start`/`--count` ranges, then `--combine`.
Raw frames are backprojected once per method with their original absolute
brightness. The exact user recipe remains Wavelet 27/0/0/0 followed by Adaptive
Deconvolution 15.6 with Contrast Adaptive enabled.

The focused CPU/CUDA suite passes 32 tests, including blur selection with
gain/offset, baseline parity, fixed AP geometry, exact-blur zero-motion
recovery, blank-reference fallback, and existing coherent/validation/shard
checks. Baseline stacks and noiseless-oracle export hashes are checked against
the previous experiments. The numerical record is collected by
`tools/collect_blur_matched_results.py` into
[`results/registration/blur-matched-local-registration.json`](../results/registration/blur-matched-local-registration.json).
`tools/probe_blur_range.py --out out/blur-range-probe.json` reproduces the
score-only range sensitivity check on CPU with eight threads.

## Limits

Additional isotropic Gaussian blur is a restricted model of seeing. It cannot
make a soft reference sharper or represent spatially varying/asymmetric blur.
Local deformation can bias the global-pose blur estimate; false local motion can
bias the local-pose estimate. Neither estimate is independent of the pixels
used for alignment. The original reference-gradient response kernels remain
fixed, an approximation for blurred templates. One capture and two synthetic
scenes do not establish general performance. Real-image fine-scale variation
contains structure and artifacts as well as noise; ring transition widths are
not calibrated resolution measurements.
