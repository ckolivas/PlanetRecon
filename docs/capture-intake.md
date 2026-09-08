# Independent capture intake

Use `tools/inventory_captures.py` to collect SER structure, effective colour mode,
timestamp validity, whole-file SHA256 and explicit report/fixture permissions.
This is intake evidence, not reconstruction or scientific qualification. Private
filenames, paths, absolute timestamps and observer/instrument/telescope fields
are omitted from the exported report. Keep its input specification under `out/`.

Each private-spec JSON entry supplies an anonymous `id` and local `path`, with
optional `target`, `night_group`, `camera_group`, `bayer_override` and
`interpretation_evidence`. A Bayer override requires an explicit reason. IR642
and L3 Mars must remain RGGB according to the owner's correction. Do not count
pseudo-monochrome Bayer data as a true mono capture.

Permission fields are `permissions.report_pixels` and
`permissions.redistribute_fixture`, each `unknown`, `allowed` or `denied`.
An allowed use requires `permission_evidence`; local availability alone does not
establish permission to distribute. Defaults remain unknown. Group identifiers
must come from verified capture provenance: filenames or similar-looking images
alone do not prove independent nights, cameras or conditions.

```sh
.venv/bin/python tools/inventory_captures.py --private-spec out/capture-intake/private-spec.json --out out/capture-intake/hashed.json
```

Use a new output path. `--metadata-only` skips reading image payloads and leaves
the whole-file hash null. Such a record cannot identify pixels or enroll a
qualification fixture. The tool verifies file stat identity before/after intake;
pixel identity in a hashed report comes from the SHA256, not from header fields.
Invalid, duplicate or reversed timestamps remain explicit and do not produce an
invented duration. A repaired timing series needs a separate documented source
and interpretation; it must not silently replace the original evidence.

For an invalid-timestamp intake, use `tools/diagnose_capture_timestamps.py` with
`--private-spec`, `--intake` and a new `--out` path. It rechecks whole-file hashes
and compares raw pixels in at most 64 uniformly spaced equal-timestamp pairs
per affected capture. Reports omit paths and pixels. Equal timestamps do not
imply equal images; this diagnostic neither removes frames nor repairs timing.

Before independent evaluation, obtain true mono Mars and additional verified
nights/cameras, field rotation, measurable globe rotation, ring geometries and
seeing/noise/calibration conditions. Bind any conventional comparison stack to
its parent capture and record frame selection, alignment, linear intensity
conversion, colour/debayer method and any sharpening. The unsharpened Jupiter
stack is a development comparator; its sharpened counterpart is separate.

Keep development captures separate from an untouched final assessment set and
freeze numerical/quality acceptance before tuning that set. A conventional
stack is not ground truth: include held-out raw residuals, split-half consistency,
coverage/colour checks and repeated independent captures. Unresolved permissions,
timing or provenance must remain visible in each category's assessment.
