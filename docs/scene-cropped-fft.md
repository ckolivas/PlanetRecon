# Exact FFT padding for a retained detector crop

This experimental backend reduces discarded convolution work without shortening
PSFs, changing scene variables, assuming a periodic sky, changing precision, or
altering the detector operator. It is not yet qualified at full count or adopted
by a solver. Both CPU and CUDA retain float64/complex128 arithmetic and the
existing bounded retain-first spectrum policy.

## Padding derivation

In one axis, let the zero-extended scene and PSF lengths be `n` and `k`, and let
the retained full-convolution indices be the half-open interval `[s,t)`. Linear
convolution has support `[0,n+k-1)`. Choose a transform length satisfying

```
N >= max(n, k, t, n+k-1-s).
```

The first two inequalities embed both inputs without truncation. The third
embeds all retained outputs and excludes aliases from negative indices. For any
retained index `j >= s`, the fourth gives `j+N >= n+k-1`, so no nonzero higher
linear-convolution sample folds onto it. More distant multiples also lie outside
support. Applying this argument on both axes excludes every nonzero 2D folding
alias in the retained rectangle. Take the maximum over all frame crop intervals
and then round upward to an FFT-friendly length. Full-convolution padding remains
a valid special case; arbitrary smaller padding is not valid.

The forward operator is the retained circular convolution on that embedding,
which equals the original retained linear convolution in exact arithmetic.
Its adjoint must use the *same* crop embedding, conjugated PSF spectrum and scene
restriction. The inherited implementation does so. Detector integration, masks,
flux, exposure averaging, colour sampling and cell expansion/reduction are left
in the same order. Thus the normal operator and objective are unchanged in exact
arithmetic; different FFT roundoff is checked against the independent reference.

## Controls and limits

Controls compare independent forward, adjoint, dot-product identity and weighted
normal results at central and edge crops, odd/even kernels, masked exposures,
mono/full RGB and all four CFA patterns. A deliberately too-small FFT produces
measurable wrapped contamination in an edge impulse control. Cache byte limits
still use the actual complex128 spectrum dimensions, including zero-cache runs.
CUDA and cell-basis objective/gradient/certificate parity are separate controls.

The fast path still requires zero reference translations and common scene and
PSF shapes. The crop bounds are sufficient, not claimed to be minimal for every
mask or PSF with internal zeros. No full-count speed or convergence gain may be
claimed before a prospectively bounded independent operator/resource profile.
Changing a solver's FFT backend requires a new study identity and the original
independent numerical certificate; historical incomplete fits remain unchanged.
