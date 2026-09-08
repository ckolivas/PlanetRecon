# Window and starting-state diagnostic

Prospective protocol, 2026-09-09. The original periodic inverse improved a
near-solution linear probe but regressed constrained fits from zero. Compare
diagonal, original periodic and window-averaged inverses without fitting an image.

Use seed1001, D/r0=4, feature, frozen observed 5%/100% selections (25/500 frames),
native cells, margin64, observed variance and SUM prior0.0003. For each count,
probe the zero state and the original reference late state: the certified25-frame
iteration310 result under cap750, and the incomplete500-frame iteration142
checkpoint under cap750. Bind the25-frame payload SHA256
`b98d85069d2d16e329491f6597c22dad66892cda74d5d19ff245d188ab49f9bb`, stage identity,
original protocol and selection; validate the500-frame embedded checksum and
identity as before. Recompute all gradients on independent CPU. Do not resume
or change either solver state or reclassify the historical outcomes.

Use the CPU free set x>0 or g<0 and its normal-cone residual. Run each of the
three preconditioners for at most32 reduced-Hessian products from zero:12 probes
in total. Retain each trace and terminal correction locally. Independently
recompute full H*y on CPU and require CUDA product relative error <=1e-10.
Record both free-subspace and full residual ratios, because reduced residuals
are not full residual or scene-distance bounds. Record CPU/CUDA free-mask
disagreement as a sensitivity measurement; do not change the CPU mask to obtain
better results. The existing1e-5 independent scene criterion remains unchanged.

Initial GPU gradient checks use backward error normalized by ||Hx||+||b|| at
threshold1e-10; also report the raw relative gradient discrepancy. This avoids
using a cancellation-sensitive gradient denominator near convergence. Only the
CPU gradient determines residuals, masks and initial certificates. Reconfirm
that the late25-frame CPU state meets the original1e-5 requirement.

Measure the exact Fourier diagonal through independent forward weighted energies
at four frequencies: (0,0), (1,0), (0,1), and (ny//8,nx//8). Compare the periodic
and window symbols with those energies. These four frequencies are diagnostics,
not a spectral bound or proof of a whole-operator approximation. Dense controls
cover interior and boundary footprints, nonperiodic smoothness, state checksums
and positive reduced inverses. An approximate inverse never replaces H.

One CUDA study uses the qualified retain-first3GiB cache and at least1GiB extra
free headroom, two BLAS threads and eight ordered CPU workers with one FFT thread
each. The total diagnostic budget is900s, including setup, two inverses per count,
384 maximum Krylov products, independent terminal products and Fourier-energy
checks. This is a new larger diagnostic design, not an increase to historical
300s fit budgets. Check the deadline between operations; active operations and
verification may cross it. Preserve incomplete outcomes if the budget runs out.

Compare performance across both starting states and counts before choosing
further solver work. No prior/threshold tuning, automatic fit retries, production
adoption, full-family expansion or scientific gate follows from this diagnostic.
