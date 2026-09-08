# Latent-cell sampling pilot

Native, 2×2 and 4×4 cell representations all passed both budgets and independent
CPU certificates at 110 iterations per solve. Coarse integrated cell flux was
uniformly deposited onto the native optical grid; PSFs, native optical integration
and detector pixels were unchanged. The ridge coefficient was divided by coarse
cell area, preserving the same squared-density prior rather than imposing a
sampling-dependent penalty.

Detector-image relative changes from native sampling were 0.001503 and 0.007598.
Selection residual scores were 3.23533 (native), 3.22637 (2×2), and 3.19175 (4×4).
These differences are below the pilot's 1% material-image flag but do not prove
that a coarser basis preserves fine information or real-capture resolution.
Keep native sampling for subsequent scientific qualification until complete
family and information-loss controls justify a change. Study wall time was 27.37 s.
