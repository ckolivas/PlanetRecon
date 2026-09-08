# Scientific CPU work-unit forecast

Fresh-process measurements at pupil samples 16/32/64 use 20 repetitions per
15/35/60-mode gradient, with two CPU threads. At the production 512x512 pupil
FFT and 128x128 detector grid, median gradients take 17.51/19.66/19.79 ms,
and the measurement process peaks at 178.5 MiB RSS. These are isolated gradient
measurements, not end-to-end reconstruction memory or timing.

At 100 gradient calls per frame per mode stage, a three-stage fit costs about
0.79 CPU work-unit hours for 500 frames; 1k/5k/20k forecasts are 1.58/7.91/31.64
hours respectively. At 1,000 calls those costs are ten times larger. The report
includes 10/100/1,000-call scenarios to expose sensitivity to optimizer work.
Multiple initializations, partitions, crops and seed families multiply the cost;
object updates, I/O, simulation and parallel efficiency are not included.

No GPU equivalence or speedup is inferred. No Q3 job was started or authorized.
Reproduce with `.venv/bin/python tools/benchmark_scientific_operator.py --out NEW_DIRECTORY`.
