# Planetary rotation presets

Select **Geometry → Planet rotation preset** to use a published bulk rotation
period when texture tracking cannot measure spin. Leave **Surface rate override**
blank; an explicit value, including zero, takes precedence. The CLI equivalents
are `--rotation-planet saturn` and `--reverse-rotation`.

The offline table uses signed sidereal days from
[JPL Solar System Dynamics](https://ssd.jpl.nasa.gov/planets/phys_par.html),
retrieved 2026-09-10:

| Planet | Sidereal days |
| --- | ---: |
| Mercury | 58.6462 |
| Venus | -243.018 |
| Earth | 0.99726968 |
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
