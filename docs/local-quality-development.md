# Development with local alignment and the upper 50% quality range

Updated 2026-09-10. This is the current direction following the user's selected
best settings. It supersedes earlier default and next-step instructions in the
historical development log. New GUI and CLI jobs already use these defaults
(commit `512be78`). Existing saved jobs retain their settings.

Use local patch alignment, Motion None, original linear Emil quality weights,
and scores strictly above `(capture best + capture worst) / 2`, followed by cached
screening. This is a quality-range cutoff, not half the frame count. Preserve the
best selected frame as the reference origin, normal template selection, original
CFA measurements, direct colour support and the existing local ambiguity/fold
guards. No automatic sharpening. Stronger quality weighting remains optional.

## Completed: reference-coordinate interpolation for local maps

The previous quadratic prototype supported global translations only. Building
its global-only resumable engine is deferred: it would not support the chosen
working mode. The new separate [local transport probe](../tools/cfa_local_transport_probe.py)
locates observed raw detector centres in the reference plane by inverting the
actual local pull map. It refuses maps outside a conservative contraction bound
or whose inverses do not converge. It uses the frozen radius-four triweight and
quadratic fit, then affine under an ordinary-stack iid-noise variance cap, then
ordinary colour completion when sample geometry cannot support either fit.
Only ordinary observed output channels are eligible. Direct CFA support remains
separate from reconstructed-image variance. Frame contributions are staged before
publication; cancellation or an invalid map leaves all preceding sums intact.

Twenty-one controls pass: four Bayer layouts, known row-shear maps, polynomial
and curved scenes, midpoint selection, an independent weighted least-squares
oracle for translation, raw-impulse variance and colour isolation, inverse refusal,
frame cancellation, chunk independence and exact cropped/full geometry agreement.
A preliminary check also matched the earlier frozen global prototype to 1e-10
on its complete interior; the committed control uses a self-contained independent
least-squares oracle. These are image-sampling controls, not physical detector
PSF or motion-estimation qualification. Local deformation changes pixel footprints;
that physical effect remains outside this interpolator's model.

The [fixed Jupiter pilot](../results/real-data/cfa-local-transport.json) uses 64
uniformly spaced entries among the 1,645 frames passing the upper-half range and
screening. It includes anchor 1947, uses the normal 64-frame selected template,
and shares the exact local maps and original linear weights between both arms.
The fixed central crop is y=164:260, x=280:376; ordinary support and completion
are calculated on the full detector. A shared registration from the full ordinary
stack positions the conventional unsharpened reference. Both comparisons fit
per-colour gain/offset on the same 6,400-pixel interior, without sharpening.

RMS difference falls from 0.00765598 to 0.00516725 (32.51% relative reduction),
and sigma-three highpass correlation rises from 0.633049 to 0.850667. All 27,648
crop channel samples support quadratic fits under the variance cap. The run took
79 seconds using GPU registration and CPU fitting. This passes the predeclared
requirement that both metrics improve. It is a small development crop, not proof
of improvement for the full selected stack; the conventional reference is not
independent truth. No application reconstruction code changed.

Private evidence is under `out/cfa-local-transport`: the prior protocol, exact
run script/source, logs, indices/template, hashes, float32 TIFFs and matched PNG
previews. `ordinary.tif` and `quadratic.tif` contain unsharpened crop intensities;
PNG previews share display scaling after the declared gain/offset comparison.
The [reproduction script](../tools/cfa_local_transport_pilot.py) requires the private
capture and its existing cache and refuses to overwrite a completed pilot.

## Next steps

1. Bound the local fitting cost and memory while preserving the qualified sample
   positions, weights, fit choices, variance and output. Do not use global parity
   grouping where local geometry differs. Include cancellation during inversion
   and final fitting, not just between staged frame contributions.
2. Declare a larger selected-frame confirmation before running it. Keep the same
   local/50% baseline, anchor, normal template rules and interpolation parameters.
   Check both the fixed crop and broader planetary regions; require both metrics
   to improve on common support. Do not infer a full-stack gain from the 64-frame
   result or tune window/weights against inspected references.
3. If confirmed, implement exact resume bound to source, selection/weights, dense
   local maps and execution policy. Failed GPU work must retry a complete frame
   without double counting. Then offer an optional GUI integration for user testing.

Physical motion modes and transformed detector footprints require separate
qualification. The earlier rejected global stronger-weighting/first-moment
integration stays reverted. Do not repeat completed global or rejected weighting
studies. The periodic continuation automation remains paused; this document does
not schedule background work.
