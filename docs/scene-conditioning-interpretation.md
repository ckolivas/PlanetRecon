# What the fixed-residual gap probes can establish

This algebraic limit interprets the diagnostic; it is not a relaxed acceptance
criterion. For its fixed normal-cone residual r and Hessian bounds
lambda I <= H <= L I, inversion reverses the matrix order, so

    r^T H^-1 r >= ||r||²/L.

Even an exact correction H^-1 r therefore cannot make the distance certificate
sqrt((r^T H^-1 r)/lambda)/max(||x||,1) smaller than

    raw_relative_distance_bound / sqrt(L/lambda).

This is a lower limit on the **reported bound in this certificate family**, not
a lower bound on the actual reconstruction error. It also does not rule out
other certificates that exploit the constraints differently. If this limit
exceeds1e-5, the fixed-normal-cone/global-ridge gap strategy cannot certify that
particular saved iterate at1e-5, regardless of how many inner CG steps are used.
A limit below the target does not prove that a short probe can attain it.

Use the proved majorizer maximum for L, not a sampled Rayleigh quotient. The
diagnostic's directional quotients and residual traces may help choose a later
preconditioner study; they must not silently replace a valid eigenvalue bound.
Independent dense Hessian tests check this limit against exact inverse energies.
