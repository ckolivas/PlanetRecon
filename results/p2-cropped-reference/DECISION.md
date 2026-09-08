# Reference reaches full-count accuracy; original cap stability still fails

All four fits completed in 1185.14 seconds with unchanged source/input hashes.
The exact-crop FFT backend and explicit 900-second full-count allowance exercised
the existing iteration caps without changing the objective or numerical target.

| Frames | Cap | Iterations used | Independent distance bound | Outcome |
|---:|---:|---:|---:|---|
| 25 | 750 | 310 | 9.958545e-6 | Certified |
| 25 | 1500 | 310 | 9.958545e-6 | Certified |
| 500 | 750 | 750 | 9.205073e-4 | Iteration limit, incomplete |
| 500 | 1500 | 1410 | 9.859417e-6 | Certified |

The 500-frame, 1500-cap scene is the first full-count scene in this sequence to
meet the unchanged fresh independent CPU distance criterion. This is numerical
accuracy for the fixed known-transfer objective, not scientific truth or a prior
qualification. The 750-cap scene remains incomplete. Their latent/detector
changes, 4.19399986e-4 / 2.58208313e-4, exceed 1e-4. The overall endpoint study
therefore retains `incomplete` status; it must not be relabeled a pass.

The 25-frame outputs are identical across caps and reproduce the earlier bound.
Optimizer wall times are 14.77/14.65 seconds at 25 frames and 364.97/685.17 seconds
at 500 frames. Reference cap units are iterations; periodic certificate products
and initial/final gradients add work. Local timings are not universal speed
claims. The companion comparison includes tested normal-call accounting.

Next explicitly freeze the certified 1500-cap stage as the lower endpoint and
run a fresh 3000-cap reference solve from zero. Recompute both independent CPU
certificates and require the original latent/detector stability limits. This
checks stability at a numerically adequate lower budget without rewriting the
failed 750/1500 experiment, changing tolerances or warming the new solve from
the saved result. Source/input identities and stage checksums must match. The
full 60-selection family and scientific gates remain separate requirements.
