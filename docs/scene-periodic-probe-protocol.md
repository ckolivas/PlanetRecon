# Coupled periodic inverse on a frozen objective

Prospective protocol, 2026-09-09. The earlier diagonal probes and bounded
Newton-CG fits leave full-count convergence unresolved. Test one non-diagonal
positive inverse without changing the frozen iteration142 or the forward model.

For each native-cell, zero-translation observation, average its exposure PSFs
and integrate the detector footprint before forming the squared Fourier modulus.
Replace spatial detector weights by their per-channel mean and subsampling by
density 1/b^2 only inside the preconditioner. Its periodic symbol is

    ridge + periodic_smoothness_symbol
          + sum_k mean_weight_k * flux_k^2 * abs(FFT(integrated_psf_k))^2 / b_k^2.

The positive ridge makes this real convolution symbol strictly positive; the
inverse is self-adjoint positive definite. Restricting its application as P C^-1 P
on the free subspace preserves positive definiteness there. Finite crop,
nonuniform noise, Bayer aliases and nonperiodic boundaries remain in the exact
Hessian. This approximate inverse is neither a forward-model replacement nor a
certified bound on the true Hessian. Oversized kernels must wrap, not truncate.
Reject coarse cells or translated exposures for this initial implementation.

Before the full probe, verify symmetry/positive eigenvalues on dense full and
reduced subspaces, an independently constructed toroidal convolution inverse,
integration-before-squaring, sampling density, wrapped kernels and bounded PCG
against dense principal-system solves. Invalid curvature must fail explicitly.

Use the same original reference checkpoint, input and manifest identities as
the frozen-conditioning study. Define the free set by x>0 or g<0 using the
independent CPU gradient. The normal-cone residual is supported on this set.
Run four probes from zero: full/diagonal, full/periodic, reduced/diagonal and
reduced/periodic. Each has at most32 Hessian products; a recursive relative
residual <=1e-12 may stop the linear probe early but cannot qualify the scene.
Record every product and terminal correction locally; never update the scene.

Use the qualified retain-first3GiB CUDA cache with at least1GiB extra free
headroom, two BLAS threads and eight ordered independent CPU workers with one
FFT thread each. Total wall budget600s including setup, inverse construction and
verification, checked between operations; active operations/checks may cross it.
Before each interpretation, recompute the full H*y independently on CPU, check
CUDA agreement <=1e-10, and separately record full and free-subspace residuals.
Only the full H*y can enter the existing arbitrary-correction bound. Reduced
residuals and recursive traces are not scene certificates.

The already proved floor prevents1e-5 qualification of this saved iterate with
that bound family, even if a linear probe improves. Compare achieved independent
linear residuals under equal32-product caps to decide whether this one inverse
merits a new constrained-update experiment. Preserve failures, thresholds and
historical solver identities. No automatic prior/budget expansion, production
adoption, full-family or scientific-gate qualification follows.
