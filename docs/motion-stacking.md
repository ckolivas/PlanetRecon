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

## Limb darkening, ring motion and field rotation

**Limb-darkening compensation.** A globe's brightness is its surface albedo
times limb darkening, which depends only on the emission cosine mu and so
stays fixed while the surface turns. Each globe sample moved from mu_src to
mu_ref is scaled by L(mu_ref) / L(mu_src) with L(mu) = mu^p (mu clamped at
0.2, where the limb is dominated by blur), as WinJUPOS applies when
derotating. Model predictions of a frame (drift tracking, alignment-point
references) apply the inverse; support channels are never scaled. p is fitted
per capture on the anchor frame over mu >= 0.5, with a separate brightness per
2-degree latitude band so belts do not bias it, excluding rings in front of the
globe and ring shadows (`limb_darkening_power` overrides it; 0 disables).
Without it the limb-darkening slope itself moves with the spin: when the stacked
frames are weighted away from the reference time (seeing improving late in a
capture), one limb gains a compressed, steeper edge that sharpening turns into
a bright rind, on the side set by the sign of that time offset. On synthetic
limb-darkened globes with only zonal belts and frames weighted late, the
compensation reduces the error where spin moves light 22x (Saturn) to 67x
(Surface), and the fit recovers p within 0.001.

**Ring motion (Saturn).** Ring material turns with the planet. Spokes form
near synchronous orbit and follow the planet's rotation: one tracked through
the sharpened stacks of 2026-10-14 (12:05-12:28 UT) moved at 0.561 degrees per
minute, against 0.563 for Saturn's rotation and 0.627 for orbital motion at its
radius. In a 6-minute capture such a spoke 40 degrees in from an ansa moves
about 6 px across the sky. The Saturn model turns the ring plane with the globe
(an exact affine map per frame, which carries uniform rings onto themselves).
Light fixed to the viewing or lighting geometry does not move: ring motion
fades in from 4 to 14 px clear of the globe's silhouette and its shadow on the
rings, and out from 4 to 12 px beyond the ring edges. Rings in front of the
globe follow the globe. Viewing-geometry ring asymmetries (the A ring's
azimuthal brightness pattern) are smooth enough that a few degrees of turn
does not affect them.

**Field rotation.** The rotation estimated from sampled ring or field angles
(a least-squares slope; ring angles folded to one half turn, since rings look
the same turned by 180 degrees) is applied only when significant at 95%
confidence by Student's t on its residual scatter. With three or four samples
that scatter is itself uncertain, so a fixed number of standard errors would
pass noise. Preprocessing's ring-axis estimate uses the same test.
Equatorial-mount captures therefore keep a fixed field instead of a few tenths
of a degree of spurious rotation.

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
