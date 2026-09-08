# Observed full-development selection manifest

Prospective protocol, 2026-09-08. Freeze all 60 combinations of seeds1001–1003,
D/r0=4/8, feature/bland crops and 5/10/25/50/100% selections before their
known-transfer numerical solves. Counts are 25/50/125/250/500. This manifest
declares data selection, not solver budgets or scientific qualification.

Rank the raw observed detector frames by the existing mean squared four-neighbour
Laplacian, omitting two border pixels. Do not denoise, sharpen, register, use
expected-image values, latent truth, PSF scores or a truth-derived sky mask.
This explicitly differs from the legacy registered ranking; it is an observed
selection for the new numerical study, not a reissued legacy Gate-1 result.
Keep higher scores; ties follow the existing implementation (later frame first).
Store selected indices chronologically. Keep every required fraction, including
all frames regardless of their rank. Nonfinite scores/frames invalidate a case.

Record all scores, indices, observed electron sums (which include noise and may
include negative pixels), captured/used counts, input/code/runtime identities,
and mean observed scalar variances. Freeze SUM ridge .0003 solely as the numerical
reference; its mean-likelihood coefficient is .0003 / number of selected frames.
Do not infer true photon counts from the noisy observed sums.

This uses all development frames for ranking. It is not a held-out prediction or
final assessment protocol, and does not reserve the previously inspected pilot
assessment. A subsequent scientific protocol must provide separate untouched
assessment. Preserve the unbracketed prior decision and do not tune from this
manifest. No new solver pass, likelihood choice, production default or Q3 follows.
