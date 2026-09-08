# Full-count projection/CG remains incomplete

Both independently started 500-frame fits reached the original 300-second wall
limit before either product cap. The study completed in 714.66 seconds including
setup and independent verification, with unchanged source/input identities and
no execution exception. The numerical outcome is a failure to certify.

| Product cap | Products used | Accepted updates | Optimizer wall time | Independent distance bound |
|---:|---:|---:|---:|---:|
| 750 | 276 | 51 | 301.80 s | 2.793714 |
| 1500 | 277 | 51 | 302.65 s | 2.793714 |

The extra product in the larger-cap run did not produce an accepted update.
Both runs retain the same feasible image; zero latent/detector change is not
convergence because both independent certificates miss 1e-5 by a wide margin.
The wall policy retains transforms/final checks already in progress. All
accepted and inner traces, including unfinished directions, remain archived.
See [accepted progress and independent final bounds](comparison.png).

Passing both 25-frame caps did not transfer to full count. This bound is worse
than the earlier incomplete reference and diagonal Newton bounds at their own
limits; differences in backend/resource setup prevent an isolated timing claim,
and none of those full-count fits is qualified. Do not adopt the new method,
promote Q3, change the prior, or loosen the convergence criterion.

An independent compute improvement is now being qualified: exact detector-crop
FFT padding can omit discarded convolution work while retaining the full
forward/adjoint in exact arithmetic. Small CPU/CUDA controls pass, including
independent fitted-scene certificates in supported native/coarse configurations.
The window inverse is native-cell-only; coarse-cell checks use the supported
diagonal inverse and explicitly verify rejection of the unsupported combination.
A predeclared full-count operator/resource profile is required before solver use.

Next use that profile and the retained trajectories to declare further execution
work. Prefer a controlled comparison including the earlier reference/diagonal
methods; this failed projection run does not justify choosing it as the sole
candidate. Any new backend, budget or protocol requires a new identity. Keep
historical failures and the independent numerical target unchanged. Full-family,
prior/sensitivity and scientific qualification remain separate requirements.
