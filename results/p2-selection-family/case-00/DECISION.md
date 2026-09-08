# First complete observed-selection case passes

Seed 1001, Dr0=4, feature crop passes all five frozen observed selections at
both independently initialized 1500/3000 iteration caps. Source/input identities
remained unchanged; no execution failures occurred.

| Selected frames | Iterations at each cap | Independent bound at each cap |
|---:|---:|---:|
| 25 | 310 | 9.95854544e-06 |
| 50 | 440 | 9.72389862e-06 |
| 125 | 710 | 9.30326251e-06 |
| 250 | 1000 | 9.99018789e-06 |
| 500 | 1410 | 9.85941713e-06 |

Every fit is feasible and meets the unchanged 1e-5 independent CPU distance
criterion. Both latent and detector differences are exactly zero for every
selection, meeting the unchanged 1e-4 stability criterion. The complete case
took 2113.29 seconds under the declared 3600-second ceiling.

The cumulative checker independently revalidates the recorded evidence and
counts 1 of 12 cases and 5 of 60 selections as passed. Eleven cases remain
missing. This is a numerical qualification of one known-transfer case; it
does not select the scientific prior, qualify blind reconstruction or authorize
Q3 or production adoption. Historical failed studies remain unchanged.

Next execute case 1 (the same seed and seeing, bland crop) under the same
frozen protocol and source/runtime identity. Preserve failed outcomes as well
as successful ones; no implicit budget extension or rerun is authorized by
this numerical result.
