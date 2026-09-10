# Planetary rotation presets

The GUI selects **Geometry → Planet rotation preset** from a clear planet name
in the capture filename, including the common `Jup` and `Sat` tokens. Ambiguous
names leave it unselected. This supplies a published bulk rotation period when
texture tracking cannot measure spin. The detected choice is displayed and can
be changed; manual choices, including **Measured / manual rate**, are preserved.
This does not change the selected motion model or ordinary stacking defaults. Leave **Surface rate override**
blank; an explicit value, including zero, takes precedence. The CLI equivalents
are `--rotation-planet saturn` and `--reverse-rotation`.

The offline table uses signed sidereal days from
[JPL Solar System Dynamics](https://ssd.jpl.nasa.gov/planets/phys_par.html),
retrieved 2026-09-10:

| Planet | Sidereal days |
| --- | ---: |
| Mercury | 58.6462 |
| Venus | -243.018 |
| Mars | 1.02595676 |
| Jupiter | 0.41354 |
| Saturn | 0.44401 |
| Uranus | -0.71833 |
| Neptune | 0.67125 |

The angular rate is `2π / (period_days × 86400)`. Radius does not change that
rate: the globe model projects it using the measured radius and viewing geometry.
The GUI displays `abs(rate) × radius` as the equator-on pixel-motion scale.
Negative periods describe retrograde rotation relative to the conventional
north pole. They do not identify the pole in an arbitrarily oriented capture.
At pole position angle zero, a positive model rate moves central texture left;
**Reverse preset rotation direction** changes that apparent direction. It does
not change an explicitly entered override.

These are bulk/body rotation priors, not fitted cloud speeds. Latitude-dependent
cloud winds can differ. A preset preserves the existing requirements for actual
timestamps or supplied cadence, and for the viewing geometry needed by the
chosen model. In particular, a ring ellipse alone does not identify the signed
planetary viewing latitude. This is distinct from the telescope's latitude on
Earth; the application does not require an Earth observing-site position.

Result metadata identifies preset use, source, signed period, reversal and the
radius used for pixel scaling. Checkpoints bind those choices. Existing default
jobs without a preset keep their previous checkpoint identities.

## Optional planetary viewing latitude

Leave **Planet-facing latitude override** blank for automatic geometry in Saturn
mode, or in Surface/Combined mode with a planet preset selected. Preprocess and
Run use the SER UTC capture midpoint and an Earth-centred
[JPL Horizons observer ephemeris](https://ssd-api.jpl.nasa.gov/doc/horizons.html).
No telescope latitude or longitude is needed. Only the target planet and UTC
are sent to JPL; images and file names remain local.

Horizons quantity 14 gives planetodetic latitude (the surface normal angle).
PlanetRecon converts it to the planetocentric viewing direction using Horizons'
reference equatorial and polar radii: `atan2((c/a)^2 sin(B), cos(B))`.
The signed result controls globe projection and Saturn ring opening. The camera's
pole position angle still comes from the image or user; an ephemeris cannot
identify north in an arbitrarily rotated or mirrored capture.

The first lookup needs internet access. A validated response is cached beside
the SER as `*.planetrecon-viewing.json`, keyed by planet and capture midpoint,
for subsequent offline use. Preprocessing displays the resolved latitude and
retains it in its report. The input stays blank so changing captures recalculates
the view. Result/checkpoint geometry records the source, epoch, radii and actual
latitude; an explicit override, including zero, always takes precedence.

Absolute trailer UTC takes precedence over the SER header UTC. A valid header
can supply the date when the trailer is absent; available duration/cadence places
the lookup at the midpoint. Automatic dates are supported from 1900 through 2100.
If no usable date or ephemeris is available, Saturn motion refuses to guess an
edge-on view and explains how to supply an override. Surface runs without a
known target/date retain their existing equator-on assumption.

This removes the manual latitude requirement for dated planetary captures; it
does not infer a globe radius from Saturn's full ring-system diameter. Globe and
ring radii remain required for Saturn motion compensation.

Preprocessing resolves the selected planet's viewing latitude even when Motion
model is still None, so geometry discovery is ready before choosing Surface.
Unresolved image spin does not disable a selected planet preset. Error messages
are selectable with mouse/keyboard and can be copied with Ctrl+C.
