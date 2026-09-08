# Full-resolution noise-weight sensitivity

All 12 development cases (three seeds, D/r0 4/8, feature/bland crops) complete at
500 frames and 128×128 detector resolution. Both S10 and S100 are tested with
unchanged membership and E1 regularisation. The oracle per-pixel variance uses
the simulator expectation plus read-noise variance; it is a frozen weighted
quadratic, not a fitted Poisson likelihood or a proposed real-data interface.

All 48 weighted solves converge at 256/512 iteration caps, requiring 25–137 CG
iterations. Measured normal-equation relative residuals are 5.44e-10–9.96e-10.
Doubling the budget changes no image. The separate dense normal-equation oracle
and constant-variance Fourier reduction pass. Wall time is 985.95 seconds with
two worker processes (two CPU threads each), alongside other scientific jobs.

Per-pixel weighting changes the image by 11.1–47.6% relative to the scalar-weight
result and worsens both E_H and image MSE in all 24 tested subset/crop cases.
Bland-crop high-band scores remain ill-conditioned diagnostics. For feature
crops, E_H ratios (weighted/scalar) range from 1.004 to 3.963; no sharpening or
other post-processing is involved.

This is not evidence that detector variance is spatially constant. Realized
noise divided by the known variance has mean square 0.9983–1.0045, consistent
with the simulator's noise scale. The deployed circular crop-forward model
disagrees strongly with the padded optical simulator, particularly near crop
edges. See `../r10-full-crop-model-audit/report.json`: its whole-crop model
discrepancy is 41.85–2711.66 in units of expected noise variance, falling to
0.0212–0.4371 in the central 64×64 region. The metric window alone leaves
4.54–489.07. Noise weighting emphasizes different parts of this biased model.

Keep the production weighting unchanged. The next model step is a shared padded
scene / detector integration / crop operator (or a separately validated interior
approximation), followed by repeated noise sensitivity. Cropping away 32 pixels
is not implemented as a workaround. This diagnostic does not qualify MFBD, Q2,
or Q3.

Reproduce with the project venv:

```sh
.venv/bin/python tools/audit_noise_weights.py --inputs out/r10-full-gate1/inputs --out NEW_DIRECTORY
.venv/bin/python tools/audit_crop_model.py --inputs out/r10-full-gate1/inputs --out ANOTHER_NEW_DIRECTORY
```

Reports retain input hashes, source identities, frame indices, photon counts,
solver diagnostics and per-case resource measurements. No capture pixels are
included in this evidence.
