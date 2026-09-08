# Cropped-FFT diagonal Newton: full product caps remain incomplete

The same diagonal Newton-CG v2 method now uses the qualified exact-crop FFT
backend and 4 GiB retention. All four fits completed in 1188.45 seconds with
unchanged source/input hashes and no execution exception. The new 900-second
full-count allowance let both declared product caps finish; the earlier
300-second failures remain unchanged.

| Frames | Cap | Products used | Accepted updates | Independent distance bound | Outcome |
|---:|---:|---:|---:|---:|---|
| 25 | 750 | 293 | 13 | 6.294593e-6 | Certified |
| 25 | 1500 | 293 | 13 | 6.294593e-6 | Certified |
| 500 | 750 | 750 | 25 | 4.044792e-3 | Product limit, incomplete |
| 500 | 1500 | 1500 | 48 | 3.936542e-4 | Product limit, incomplete |

The independently started 25-frame images are identical, reproducing the
previous qualified bound. Their optimizer wall times are 16.75 and 15.33 seconds.
The full-count fits take 360.37 and 690.73 seconds. These are local observations,
not isolated or general speed guarantees; initial/final and independent CPU
checks are recorded separately from the bounded inner/search/refresh products.

At full count, both distance bounds exceed 1e-5 and latent/detector changes are
6.358846e-4 / 4.587590e-4, exceeding the original 1e-4 stability limit. Increasing
execution time did not convert the product-limited results to passes. Feasible
objective descent and a lower upper bound do not qualify the fitted images.
The report, accepted and unaccepted traces, and descriptive inner analysis are
preserved. No prior, tolerance, active-set rule or solver arithmetic was tuned.

The separately declared reference invocation follows on the same backend and
objective. Its cap unit is iterations, unlike Newton's Hessian products, so a
same-number cap is not equal work. A reporting-only correction after this study
fixes reference deadline records to use `maxiter`; numerical solver/backend
sources and criteria are unchanged. Compare completed reference evidence before
choosing further execution work. Neither this endpoint study nor the compute
profile qualifies the full selection family, scientific prior, Q2/Q3 or production.
