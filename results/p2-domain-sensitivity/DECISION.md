# Latent-domain pilot at the fixed selected prior

All full/0/32/64-margin cases passed both solver budgets and independent CPU
certificates, with 110 iterations per solve. Training, variance, native sampling
and the selected SUM prior 0.0003 were held fixed; every retained margin cell was
unknown and the exterior was zero. No truth filled the margins.

| Detector margin | Detector-image relative change from full scene | Selection residual score |
|---|---:|---:|
| Original full scene | 0 | 3.235328 |
| 0 pixels | 0.363923 | 5.979288 |
| 32 pixels | 8.12e-6 | 3.235441 |
| 64 pixels | 6.04e-15 | 3.235328 |

The 64-pixel result passes the 1e-4 exact-domain control by a wide margin. This
follows the finite 512-cell PSF's coupling range, not an empirically convenient
crop. With zero-centred positive ridge and no smoothness, disconnected exterior
cells optimize to zero; retaining their unknowns cannot change the observed-field
optimum. The independent small-domain test verifies this argument numerically.

The crop-only scene is strongly biased even though its optimizer converges.
32 pixels happen to approximate this case well, but no general permission to
truncate the scene there is inferred. Use the exact 64-pixel domain for the
predeclared sampling and photon pilots. This one-case result is not full-family
or real-capture boundary qualification. Study wall time was 37.16 s.
