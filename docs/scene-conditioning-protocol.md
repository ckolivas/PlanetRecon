# Frozen-iterate conditioning and certificate diagnostic

Prospective protocol, 2026-09-08. Inspect the preserved reference 500-frame,
750-cap checkpoint at iteration142 for seed1001, D/r0=4, feature crop. Validate
the checkpoint checksum, embedded original protocol, input and manifest identity.
Do not overwrite it, resume its solver or replace its historical incomplete result.

Keep the same full observations, native scene, margin64, observed scalar variance
and SUM prior.0003. Use the qualified eight-worker independent CPU reference with
one FFT thread per worker and two BLAS threads, and one CUDA evaluator with a
256MiB retained-spectrum cache. One study runs at a time. Total budget600s, checked
between operations; a running operation or final CPU check can cross a deadline.

Record the gradient's normal-cone residual, active-cell fraction, majorizer/ridge
range and guaranteed scaled conditioning bound. Probe H*y=r from zero for at most
32 matrix products with (a) identity and (b) existing diagonal-majorizer scaling.
Store every iteration's elapsed time, matrix count, recursive residual and
directional Rayleigh quotients atomically. These quotients are probes, not certified
extreme eigenvalues or a proof of a condition number. Retain the correction array
locally, but never apply it as a new nonnegative scene.

For each terminal correction, independently recompute H*y on CPU, checking CUDA
product agreement to1e-10 relative tolerance. Evaluate the bound below using the
CPU gradient and product; recursive CG residuals cannot qualify it. Compare the
bound with the original1e-5 relative scene-distance requirement without changing
the threshold. Passing this diagnostic does not reclassify the old reconstruction
or adopt a new certificate in production. Any failure or exhausted probe budget
remains visible. No prior tuning, full-family expansion or Q3 follows.

## Bound derivation and independent controls

Let F be the frozen quadratic with Hessian H >= lambda I, lambda = positive ridge,
and x >= 0. Take n in the normal cone at x: n_i=-max(g_i,0) when x_i=0 and n_i=0
otherwise, where g=gradient F(x). Then r=g+n. For every feasible z,
F(z)+n·(z-x) <= F(z). The unconstrained minimum of this quadratic lower function
is F(x) - (r^T H^-1 r)/2. Hence the constrained objective gap is at most this
energy/2, and distance squared is at most energy/lambda.

For **any** correction y, let e=r-H*y. The exact identity

    r^T H^-1 r = 2 r^T y - y^T H y + e^T H^-1 e
               <= 2 r^T y - y^T H y + ||e||^2/lambda

provides a bound even when CG is unfinished. Use the minimum of this bound and
the old ||r||^2/lambda bound, with a conservative scalar reduction/subtraction
roundoff guard. This is exact-arithmetic algebra plus numerical controls, not
formal interval verification of floating FFTs. Test against independent dense
constrained optima, exact dense Hessian inverses, poor corrections, active bounds
and partial CG before the full-grid diagnostic. Do not infer that a tighter
bound will necessarily certify the preserved iterate.
