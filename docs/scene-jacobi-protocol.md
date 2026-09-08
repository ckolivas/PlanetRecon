# Frozen-iterate Hessian-diagonal probe

Prospective protocol, 2026-09-08. The identity and existing majorizer probes in
`results/p2-frozen-conditioning` completed without tightening the original bound.
Test one further scaling on exactly the same saved iteration142 and objective.
All input, independent CPU verification, resource and certificate requirements in
[the original protocol](scene-conditioning-protocol.md) remain in force, except
that this invocation runs **only Jacobi scaling**, for at most32 CG products.
The complete invocation has a600s wall budget, including diagonal construction.
No extension of that budget or the1e-5 tolerance follows automatically.

Compute diag(H) for native cells and zero exposure translations. Average exposure
PSFs, integrate the detector pixel footprint, then square the effective kernel
and correlate with detector weights. Squaring before integration would omit
cross terms. Add the exact ridge and nearest-neighbour smoothness diagonal.
Validate against dense basis-column products across odd/even kernels, cropping,
flux, masks, exposure averaging, mono, RGB, all Bayer patterns and CFA offsets.
Reject coarse cells or nonzero exposure translations rather than approximating.

Construct the diagonal using eight ordered CPU workers, with one FFT thread each
and two BLAS threads. Record construction cost, worker scope, diagonal extrema
and majorizer-to-diagonal percentiles. Use the diagonal only to precondition the
linear correction probe; it is not a safe step majorizer. Never update the scene.

Compare independently checked terminal correction bounds and32-step traces with
the two previous probes. Euclidean CG residual need not decrease monotonically;
directional quotients are not certified spectral endpoints. The fixed-residual
certificate floor already rules out1e-5 certification by this family at this
saved iterate, even for an exact inner solve. This remains a conditioning
diagnostic, not a new qualified reconstruction or justification to tune priors.
