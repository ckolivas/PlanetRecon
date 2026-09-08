# Window and state diagnostic: valid, no fitted-scene qualification

All twelve declared 32-product probes completed in 615.61 seconds under the
900-second diagnostic budget. Source/input hashes are unchanged. Independently
computed CPU Hessian products agree with CUDA to at most 2.27e-14 relative error.
The original 25-frame late state still meets the unchanged distance criterion
(9.958545e-6). Neither saved scene was updated.

| Frames | State | Diagonal reduced residual | Periodic | Window |
|---|---|---:|---:|---:|
| 25 | zero | 9.175878e-5 | 7.866770e-5 | 1.073746e-4 |
| 25 | late | 0.208789 | 0.068868 | 0.032249 |
| 500 | zero | 1.181387e-4 | 1.068967e-4 | 6.201944e-5 |
| 500 | late | 1.623201 | 0.171700 | 0.147893 |

Values are reduced linear residual norms divided by the initial residual norm,
not image error or a convergence certificate. At 500 late frames, the independent
full residual ratios remain 2.420465, 1.763822 and 1.791114 respectively. The
window candidate improves the reduced residual but not the full residual over
the original periodic inverse in that case. The zero-state right-hand sides are
much easier for all three inverses; their behavior does not predict late-state
conditioning. See [the trajectories](comparison.png).

The detector-window fraction is 0.280701754. At the three selected low Fourier
frequencies, the window symbol differs from independently measured energies by
less than 0.083%, while the original periodic symbol overstates them by roughly
a factor of 3.55–3.57. The fourth sampled frequency is ridge-dominated. These
four frequencies per count do not bound the full spectral error, aliasing or
finite-boundary coupling. Interior dense tests and this actual cropped check
support the density correction as an approximation, not an exact inverse.

CPU/CUDA free masks differ at 228/146 pixels for 25 zero/late states and 527/255
for 500 zero/late states, out of 933,888 pixels. CPU masks were used throughout;
no threshold or mask was changed to suppress the discrepancies. Backward
agreement of initial gradients satisfies the declared 1e-10 target. This records
mask sensitivity to floating arithmetic, without a separate perturbed-mask fit
or a claim that it explains the solver regression.

## Exploratory projection check

After the declared study, unit directions were inspected geometrically using
`d = -correction` and the frozen scenes. At zero, nonnegative projection removes
0.58%/4.29%/4.15% of direction norm for 25 diagonal/periodic/window frames, and
3.32%/22.09%/22.02% for 500 frames. The coupled inverse therefore changes the
proposed feasible step materially despite its good linear residual. At the late
500 state, removed norm is 20.67%/24.69%/24.29%.

These are exploratory 32-product unit directions, not the actual earlier
optimizer's 0.1-stopped inner directions or accepted line-search steps. No
objective was evaluated on these projected proposals and no proposal was applied.
The measurements suggest investigating active-face identification, but do not
establish it as the cause of the periodic optimizer's regression.

## Next decision

Keep both incomplete periodic fits and original certified references unchanged.
A separate gradient-projection/CG candidate now passes small independent optimum,
feasibility, descent, budget, safeguard and CPU/CUDA controls. Next predeclare a
25-frame ablation with the same window inverse for the existing Newton and new
projection-phase solver, each from zero at the existing two product caps. This
separates the window change from outer constraint handling. Do not proceed to
another 500-frame fit unless the 25-frame candidate passes its original
independent certificate and doubled-budget image stability. No full-family,
prior, scientific gate, production integration or Q3 claim follows here.

JSON traces, independent checks, exploratory analysis and source identities are
archived; local correction arrays remain ignored. `checksums.json` covers all
other files in this evidence directory.
