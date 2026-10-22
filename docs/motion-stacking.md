# Motion compensation in Stack

Stack now honours the Geometry tab's None, Field, Surface, Combined or Saturn
selection. It enables screening and AutoStakkert-style
[alignment points](alignment-points.md), which work with every motion model.
The former Run button, which honoured the local alignment toggle, is retired. Required geometry must be resolved or supplied; an unresolved
motion model is never silently replaced with ordinary stacking.

After preprocessing, the Image tab draws the motion the current Geometry
settings predict (toolbar **Predicted motion**, on by default), for checking
by eye before stacking: an amber curved arrow along a latitude of the visible
hemisphere shows which way surface features move, and a cyan arc outside the
disc shows which way the field turns when its rate is known. Directions come
from the stack's own globe and field models; a nearly edge-on view is drawn
tilted to at least 17 degrees so the arc stays curved. The arrows follow
changes to the Geometry controls (for example a reversed rotation or a pole
flipped by 180 degrees).

The new **Automatic motion reference** choice defaults to **Good frames near
capture midpoint**. From the screened frames in the middle 20% of elapsed
capture time, it chooses the frame nearest the midpoint among the better half
by quality. Ties prefer quality, then frame index. If fewer than four screened
frames fall in that window, it expands to the four nearest available frames.
Without timestamps or supplied cadence, frame position determines the anchor;
physical rotation rates still require a real time scale.

The anchor fixes the geometry origin, so changing the optional stack
percentage/quality cutoff does not move it. The alignment-point reference
averages up to 3000 of the best selected frames, motion-aligned into the
anchor's geometry (the best screened frames when fewer than four are
selected). Those reference observations do not add extra contributions to the
output stack.

**Best frames across capture** restores the previous automatic anchor policy.
A nonzero manual reference frame overrides the automatic anchor and centres
the reference window on that frame. As before, zero means automatic. With no
preprocessing measurements, midpoint selection uses proximity alone and the
best-frame option uses the first usable frame.

Geometry discovery uses the same anchor as stacking. The interface already
defaults the output epoch to the capture's temporal midpoint when timing is
known; manual output epochs are preserved. The anchor and output epoch need
not coincide: the geometric mapping accounts for that difference. Manually
supplied detector centres should describe the selected anchor. Existing caches
retain their quality measurements, but geometry measured at the old anchor is
marked for refresh. Motion checkpoint identity includes the reference pool
and policy, preventing continuation with incompatible sums.

At each observation time, the model predicts the averaged reference. Alignment
points measure residual seeing against that prediction, using only
fully observed reference/search footprints. The residual field is inverted and
composed with the geometric mapping before the original detector samples are
scattered once. Missing or unreliable AP measurements retain the model motion.
There is no added output filter or per-frame brightness normalization. Cached
quality weights, or equal weights when disabled, apply to motion stacking too.

Surface, Combined and Saturn run projection, accumulation and alignment points
on CUDA. Field-only motion currently runs on CPU. The reference
prediction changes with each frame, so motion stacking does more work than the
fixed-reference ordinary stacker; this change makes no performance claim.

## Verification

The record below predates the removal of the square and circular matchers;
the tests now exercise alignment points with every motion model.

`tests/test_motion_stacking.py` covers elapsed-time midpoint selection with
irregular timestamps, rejected frames, sparse reference pools, manual anchors,
all four motion models with both AP methods, selection-independent references,
absolute brightness, one-frame output selections, checkpoint continuation and
refusal, old-cache refresh, missing reference support and CPU/CUDA agreement.
The integration regression run passed 406 tests (11 CUDA/hardware skips in the
CPU run); the separate CUDA run passed 46 tests with hardware checks enabled.
A final 56-test run passed after verifying that an invalid manual anchor leaves
the independent preprocessing results available (two CUDA skips).
An independently sampled rotating sphere has lower error against its known
midpoint view than ordinary stacking with either AP method. Older tests whose
truth images are in first/best-frame coordinates explicitly retain that policy.

A bounded real-capture smoke test also processed 16 screened observations
spanning `2024-09-27-1154_3-CK-R-Sat.ser`, using circular APs and Saturn motion.
Both CPU and CUDA retained all 16 frames. Maximum pixel disagreement was
3.654e-11 ADU. This used a fitted silhouette centre, approximate globe/inner-ring/
outer-ring radii of 97/145/221 pixels from prior cache measurements, cached
viewing latitude, the Saturn rotation preset and zero field rotation. A single
raw-frame globe/ring fit was unresolved, so those radii were supplied explicitly.
It is a functional check, not evidence of optimal geometry or better stacking
quality. Inputs, resolved configuration, snapshots and exports are recorded in
`out/motion-stack-integration/`, including `report.json`.
