# Fixed SUM-prior photon-count pilot

Both 3- and 11-frame fits passed both budgets and independent CPU certificates,
using respectively 110 and 220 iterations per solve. The prior on summed frame
likelihood stayed 0.0003; its mean-likelihood implementation coefficient changed
from 0.0001 to 0.0003/11, as declared. The exact 64-pixel influence domain and native
scene sampling were held fixed. Additional training frames were disjoint from
selection and assessment frames.

The held-out selection residual score fell from 3.23533 to 1.05056, with a 5.504%
change in the detector reconstruction. This demonstrates the intended combination
of additional data and fixed-prior photon normalization for this development case;
it is not an isolated causal estimate of frame-count benefit at a retuned prior,
a scientific gap table, or a full-family result. No truth or expected frames were
used by the fit or score. Study wall time was 44.33 s, alongside the sampling pilot.
