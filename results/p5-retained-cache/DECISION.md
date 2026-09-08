# Full-count cache retention evidence

All three declared modes pass independent spatial CPU normal, linear and
objective checks on 500 frames. Source/input identities stayed unchanged; total
profile time was 118.53 seconds. Maximum relative error was 8.84e-16, below the
1e-10 requirement. No image reconstruction was performed.

| Cache mode | Retained spectra | Hits during 3 products + objective | Median normal time | Peak Torch allocated |
|---|---:|---:|---:|---:|
| LRU, 256 MiB | 15 | 0 | 1.3524 s | 374,439,936 bytes |
| LRU, 3 GiB | 181 | 0 | 1.3559 s | 3,315,843,072 bytes |
| Retain first, 3 GiB | 181 | 724 | 1.2048 s | 3,333,562,368 bytes |

The new policy gives about 11% less normal-product time than either LRU mode in
this local sequential measurement. Larger LRU capacity alone does not help:
sequential sweeps evict every entry before reuse while the complete sequence
does not fit. Fixed admission keeps the first 181 spectra and computes the rest
transiently. All modes retain float64/complex128 arithmetic and the same frame
reduction order. This is not a universal speedup claim.

All 500 spectra require 8,859,648,000 bytes, exceeding the available device memory
measured before this protocol. The tested 3 GiB capacity passed the additional
1 GiB free-headroom requirement before each mode. Retained bytes were
3,207,192,576, peak Torch reserved memory 3,569,352,704 and process peak RSS
4,237,426,688 bytes. Cache limits exclude workspaces, scene arrays and the runtime.
Independent CPU setup/normal/objective costs were 35.68/22.52/11.10 seconds with
eight ordered workers. These resource scopes must accompany subsequent budgets.

Six small CPU/CUDA controls cover exact cache arithmetic, bounded retention,
sequential hits, clearing and native/coarser RGB Bayer cells. Full-count results
qualify this retention policy for this operator profile. Default production and
historical audit backends remain unchanged. Adopt it only under a new study
identity with an explicit capacity and available-memory check.

An 11% per-product gain cannot by itself establish the missing full-count
convergence. The next numerical work should address coupled curvature with a
verified non-diagonal preconditioner and active-bound controls, while retaining
the exact forward operator and independent certificate. Do not automatically
relax tolerance, increase wall budgets, tune the prior or claim scientific gates.
