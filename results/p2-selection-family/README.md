# Complete observed-selection numerical family

Current cumulative evidence: [summary through case 0](summary-through-00.json).
**1 of 12 cases and 5 of 60 selections pass. The complete family is incomplete.**

Each case uses the frozen observed 5/10/25/50/100% selections and independent
reference fits at 1500/3000 iteration caps. Both fresh CPU error certificates
must be feasible and at most 1e-5; latent and detector changes must each be at
most 1e-4. No thresholds or historical failed outcomes have changed.

| Case | Seed | Dr0 | Crop | State |
|---:|---:|---:|---|---|
| 0 | 1001 | 4 | feature | [All five selections pass](case-00/DECISION.md) |
| 1 | 1001 | 4 | bland | Running; no final qualification yet |
| 2 | 1001 | 8 | feature | Pending |
| 3 | 1001 | 8 | bland | Pending |
| 4 | 1002 | 4 | feature | Pending |
| 5 | 1002 | 4 | bland | Pending |
| 6 | 1002 | 8 | feature | Pending |
| 7 | 1002 | 8 | bland | Pending |
| 8 | 1003 | 4 | feature | Pending |
| 9 | 1003 | 4 | bland | Pending |
| 10 | 1003 | 8 | feature | Pending |
| 11 | 1003 | 8 | bland | Pending |

[Case-0 work accounting](work-case-00.json) records 343, 486, 783, 1102 and
1553 optimizer normal products per cap at 25, 50, 125, 250 and 500 frames.
These include initial/final gradients and periodic certificate products, and
exclude construction, forward-objective calls and independent CPU verification.
Both caps stop at the same certified iteration in each selection. Local optimizer
times range from 13–14 seconds at 25 frames to 619 seconds at 500 frames;
the complete case takes 2113 seconds. Equal iteration or product counts do not
imply equal cost across selected-frame counts or hardware.

See the [prospective execution protocol](../../docs/scene-selection-family-protocol.md)
and [cumulative checker instructions](../../docs/scene-selection-family-summary.md).
Run one case at a time in manifest order, preserve failed attempts, and write
each cumulative summary to a new path. This family addresses known-transfer
numerical accuracy only; scientific prior selection, blind reconstruction,
production integration and Q3 remain separate requirements.
