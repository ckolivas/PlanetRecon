# Saturn inner-ring profiles

`saturn_inner_profiles.npz` contains paired one-dimensional brightness profiles
from three 32-frame aligned averages of the user-supplied Saturn R capture.
These are diagnostic image samples, not fitted or invented ring radii.

The archive records the original frame indices, anchor, pixel-content digest,
viewing latitude and detector-to-analysis scale. Each `radii_N` array contains
analysis-pixel radii; `profiles_N` contains the median radial profile from each
ansa's angular samples, before the derivative/peak selection. Values are float32
copies of the estimator's original float64 inputs.

The old estimator paired a one-sided feature near radius 46 with the common
boundary near radius 54 and rejected every window. The regression requires the
same resolved boundary on both sides, without weakening the contrast gates or
inventing a missing edge. Full globe/ring fitting and GUI prefilling are also
covered by synthetic captures in `test_saturn_radius_discovery.py`.
