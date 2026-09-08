# First full-optical-grid numerical/noise pilot

The full 1152×1152 optical scene was reconstructed from three 128×128 feature
observations (frames 0, 249, 499 of development seed 1001, D/r0=4). All scene cells
were unknown, including margins. The prior was zero, ridge 0.001 in optical-cell
electron units; no truth was supplied to initialization or optimization.

Scalar-variance observed/noiseless runs both certified after 40 iterations, at
both 100/200 caps, with identical doubled-budget images. The observed pilot used
about 388 MB process peak RSS and 52.58 s wall time for setup and both budgets.
Other CPU work ran concurrently; this is not a controlled backend benchmark.

Spatial-variance runs exhausted both iteration caps. Their relative error upper
bounds at 200 iterations remained about 0.0087 against the declared 1e-5 tolerance.
These results remain incomplete. The observed scalar versus spatial detector
image change (~3.3%) and truth errors are descriptive and cannot select a model.
Frozen expected-plus-read variances are oracle diagnostics, not a real-data
variance estimator or an exact Poisson likelihood.

The global curvature bound is 0.0152 for scalar weights and 1.9183 for spatial
weights. A direct Hessian-row probe gave a 1.6048 upper bound, with most cells far
below it (median data row sum 9.52e-5). Merely substituting that smaller global
bound would not explain or remove the large conditioning spread. The next
specific hypothesis is a provable diagonal majorizer of the same Hessian,
followed by the same certificate, budgets and independent dense controls.
No prior, objective, convergence threshold, frame selection or scientific gate is
relaxed. Numerical qualification of this pilot is not full-family P2 acceptance.
