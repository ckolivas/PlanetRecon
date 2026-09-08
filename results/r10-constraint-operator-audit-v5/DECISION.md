# Constrained object convergence

Version 1.5 adds the composite gradient restart of O'Donoghue and Candes
([primary paper](https://arxiv.org/abs/1204.3982)). Momentum resets when its
direction opposes the projected gradient step. The objective, support and
positivity requirements are unchanged. The archived v4 run retains the v3
1e-6 stopping tolerance and records a remaining 1.403e-4 oracle image error in
one full-support case, despite satisfying that stopping tolerance.

The v5 protocol tightens stopping to 1e-8, retains the independent oracle image
tolerance of 1e-4 and objective tolerance of 1e-8, and compares 128/512/1024
iteration caps. All six cases pass from both starts at 512; the largest relative
image error is 2.8072e-6, largest absolute relative objective gap 1.180e-12,
and maximum iterations 369. Raising the cap to 1024 returns identical solutions
and termination counts. All 16 composed operator controls still pass.

The resulting development object-solver defaults are frozen at tolerance 1e-8
and cap 512. This establishes bounded known-transfer quadratic convergence,
including the formerly incomplete full-support controls. It does not establish
global atmospheric identifiability, TV convergence or full-resolution family
acceptance. The independent oracle uses dense spatial convolution and SLSQP.
No evaluation seeds selected the solver settings and Q3 is not authorized.

Reproduce with `.venv/bin/python -m planetrecon.constraint_audit --out NEW_DIRECTORY`.
