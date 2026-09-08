# Gradient-projection/CG candidate

This audit-only candidate addresses active-face identification before a coupled
Newton correction. It is not integrated into reconstruction or GUI defaults.
Small independent optima establish correctness on those controls, not speed or
full-count convergence. Original objective, ridge, stopping certificate and
scientific gates are unchanged.

The method is inspired by GPCG: gradient projection identifies a feasible face,
then conjugate gradient approximately minimizes the quadratic on that face.
The Argonne account also distinguishes reduced residual stopping from quadratic
progress stopping. [Benson and Moré, P768, section 3](https://ftp.mcs.anl.gov/pub/tech_reports/reports/P768.pdf).
PETSc provides a gradient-projection phase before its reduced solve, with bounded
projection iterations. [PETSc GPCG source](https://petsc.org/main/src/tao/quadratic/impls/gpcg/gpcg.c.html).
This is an independent implementation of a related candidate, not a port or a
claim to reproduce either complete algorithm.

## Fixed candidate rules

1. Begin at a finite nonnegative scene. Use the existing independent global-ridge
   relative distance bound and its unchanged `1e-5` target for success.
2. At each cycle, take at most eight projected-gradient steps. Direction is minus
   the nonnegative-cone residual divided by the proven diagonal majorizer. Use
   its exact quadratic minimizing scalar as the first trial, then up to twelve
   halving trials with the existing `1e-4` sufficient-decrease condition. A
   feasible majorizer step remains the fallback. Stop the projection phase when
   the active set is unchanged or decrease falls below 0.1 of the largest prior
   decrease in that phase. These controls are explicit design choices.
3. On the resulting face, restrict CG to strictly positive pixels. Bound pixels
   can be released by the next gradient-projection phase. Keep the previous
   solver's 32-product/0.1-relative-residual inner policy to isolate the change
   in outer constraint handling. An optional positive inverse affects only CG.
4. Project and backtrack the Newton direction, with the same descent safeguard.
   Refresh the full gradient every cycle. Initial/final gradients are recorded
   separately; all inner, projection, line-search and refresh Hessian products
   consume the declared product budget. Reuse a direction product only when the
   actual step is exactly the scaled direction.
5. Record each accepted feasible update, objective change, active-set changes,
   products and inner traces. Preserve rejected/interrupted inner progress. A
   budget stop returns the last accepted scene. Fresh certification determines
   final success; no exact internal resume is claimed.

Unlike the cited account, this candidate uses a diagonal-scaled projection
direction, a fixed bounded phase, and the existing residual-based CG policy;
it does not implement adaptive inner accuracy or continuing CG on the same
face. Its usefulness on this reconstruction problem must be measured.

Before a capture-scale experiment, freeze a new protocol/source identity, keep
prior inputs and numerical thresholds, and declare budgets. Do not use a failed
periodic fit, a good fixed-state linear residual, or a changing active mask as
proof that this candidate will converge faster.
