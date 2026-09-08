# Exact cropped-FFT backend passes full-count operator/resource profile

All three declared modes passed on all 500 observed frames in 104.67 seconds.
Source/input hashes are unchanged. The independent CPU reference checked the
normal product, linear term, objective and every frame's forward prediction on
the predeclared nonnegative probe with high-frequency content. Cropped-mode
errors are at most 9.60e-16 relative, below the unchanged 1e-10 threshold.

| Backend | Cache limit | FFT dimensions | Retained spectra | Median normal time |
|---|---:|---|---:|---:|
| Full padding, retain first | 3 GiB | 1536 × 1440 | 181 | 1.052072 s |
| Exact detector crop, retain first | 3 GiB | 1024 × 1024 | 383 | 0.435658 s |
| Exact detector crop, retain first | 4 GiB | 1024 × 1024 | 500 | 0.390903 s |

At the same cache limit the cropped backend is about 2.41 times faster in these
local measurements. The 4 GiB mode retains all spectra and is about 2.69 times
faster than the full-padding baseline. There are only three sequential normal
measurements per mode; these are not universal speed guarantees or a fitted
reconstruction benchmark. [Timings and allocator peaks](comparison.png).

The retained full-sequence spectra occupy 4,202,496,000 bytes instead of
8,859,648,000. The 4 GiB mode's measured Torch allocated/reserved peaks are
4,253,155,328 / 5,299,503,104 bytes. These exclude driver memory; process high-water
RSS is 4,366,696,448 bytes, not an isolated per-mode allocation. Each mode met its
predeclared additional 1 GiB free-device headroom check. Cache limits cover PSF
spectra only. The cache snapshots in the cropped rows include the extra forward
verification sweep as well as the three normal and objective measurements.

The reduction follows the proved retained-crop folding bounds. PSFs, scene
variables, precision, nonperiodic forward model, detector integration and adjoint
are unchanged in exact arithmetic. Floating agreement is independently measured,
not inferred solely from that algebra. Thirty-six small CPU/CUDA controls also
pass, including edge alias counterexamples, masks, exposure averaging, all CFA
patterns, cell bases and supported solver/inverse combinations. Four profile
error controls ensure nonfinite products cannot disappear in a maximum reduction.

## Next decision

The exact-crop backend is qualified for a new audit solver-study identity with
explicit 4 GiB retention and 1 GiB headroom checks on this machine. It is not a
production default. Revisit the earlier reference and diagonal Newton candidates
under the same backend and objective, retaining independent certificates and
budget stability; the failed full-count projection method is not established as
the best choice. Keep all old failures and distinguish changed execution cost
from improved numerical convergence. Predeclare any subsequent budget change
using these measured costs and retained trajectories, rather than altering the
completed experiments. Full numerical family, scientific sensitivity and Q3
requirements remain open.
