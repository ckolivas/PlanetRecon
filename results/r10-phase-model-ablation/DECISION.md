This is a bounded development diagnostic. Q3 remains unauthorized.

The predeclared protocol covers physical frozen-flow screens for seeds
1001–1003, D/r0 = 4 and 8, eight frames per case, a 16-sample pupil diameter
and a 32×32 detector crop. All comparisons use identical detector integration,
cropping and normalization. No historical simulation or result was changed.

Across 48 frames, exact-midpoint versus exposure-average relative OTF error
has median 0.003104 and maximum 0.015858. Increasing exposure quadrature from
8 to 16 samples changes the transfer by median 0.0000355 and maximum 0.000209.

Midpoint phase-basis truncation alone has median relative OTF errors 0.424155,
0.154943 and 0.084345 at 15, 35 and 60 modes respectively. At 60 modes its
maximum is 0.242064; combined basis-plus-exposure error has median 0.081630.
The two error sources are not additive and can partially cancel.

This isolates the earlier combined diagnostic: basis truncation dominates
exposure error on these reduced grids. It does not establish that the full
resolution application is basis-limited, that fitting additional modes will
recover the object, or that optimization has converged. The remaining W01–W03
work is operator/constraint convergence, independently assessed model selection
and affected-family requalification. W11 production inference and Q3 remain
behind those gates; real-data acceptance still requires documented captures.

Reproduce with:

```sh
PLANETRECON_THREADS=2 .venv/bin/python -m planetrecon.phase_audit --out NEW_DIRECTORY
```

The report includes the protocol, source hash, all per-frame measurements and
wall time. Static representable-phase and deliberately varied-exposure controls
verify that the diagnostic separates the two effects.
