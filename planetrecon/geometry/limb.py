"""Limb-darkening compensation for moving globe light, as WinJUPOS derotation applies it.

A globe's brightness is its surface albedo times a limb-darkening factor that
depends on the emission cosine mu. Spin moves the albedo across the disc but
the limb darkening stays fixed to the viewing geometry. Moving a sample from
mu_src to mu_ref therefore scales it by L(mu_ref) / L(mu_src). Without this the
limb-darkening slope itself moves with the spin, and a stack whose frames are
not centred on the reference time gains a bright rind inside one limb.

L(mu) = mu ** power is fitted per capture on the globe (mu >= FIT_MU_MIN, clear
of the blurred limb), with a separate brightness per latitude band so belts do
not bias it. Below LD_MU_FLOOR the limb is dominated by optical blur rather
than limb darkening; mu is clamped there.
"""
import numpy as np

LD_MU_FLOOR = .2
FIT_MU_MIN = .5
FIT_LATITUDE_BAND_RAD = np.radians(2.)
FIT_MIN_PIXELS = 500
FIT_MIN_BAND_PIXELS = 20
MAX_POWER = 3.


def ld_ratio(mu_to, mu_from, power):
    """L(mu_to) / L(mu_from) for L(mu) = mu ** power (numpy arrays or torch tensors)."""
    return (mu_to.clip(LD_MU_FLOOR, None) / mu_from.clip(LD_MU_FLOOR, None)) ** power


def fit_limb_darkening(plane, globe, cx, cy, field_angle_rad=0., rings=None):
    """Fit L(mu) = mu ** power to one alignment plane of the globe.

    Excludes rings in front of the globe and ring shadows when ``rings`` is
    given. Returns {'power', 'pixels', 'rms', 'origin'}; power 0 (no
    compensation) when too little of the globe is usable.
    """
    from planetrecon.geometry.coords import detector_to_sky, detector_xy_grids
    from planetrecon.geometry.globe import field_rotate_sky, globe_hit
    plane = np.asarray(plane, dtype=np.float64)
    x, y = detector_xy_grids(*plane.shape)
    sx, sy = field_rotate_sky(*detector_to_sky(x, y, cx, cy), -float(field_angle_rad))
    _lon, lat, visible, mu, _t = globe_hit(sx, sy, globe, globe.reference_epoch_s)
    use = visible & (mu >= FIT_MU_MIN) & np.isfinite(plane) & (plane > 0)
    if rings is not None:
        from planetrecon.geometry.saturn import classify_layers
        info = classify_layers(sx, sy, globe, rings, globe.reference_epoch_s)
        use &= ~info['on_ring'] & ~info['ring_shadow']
    band = np.floor(np.where(use, lat, 0.) / FIT_LATITUDE_BAND_RAD).astype(np.int64)
    band = band[use]
    lx, ly = np.log(mu[use]), np.log(plane[use])
    # Remove each latitude band's mean: its own brightness (belts) drops out.
    keys, inverse, counts = np.unique(band, return_inverse=True, return_counts=True)
    keep = counts[inverse] >= FIT_MIN_BAND_PIXELS
    if np.count_nonzero(keep) < FIT_MIN_PIXELS:
        return {'power': 0., 'pixels': int(np.count_nonzero(keep)), 'rms': None, 'origin': 'unmeasured'}
    inverse, lx, ly = inverse[keep], lx[keep], ly[keep]
    count = np.maximum(np.bincount(inverse), 1)
    dx = lx - (np.bincount(inverse, lx) / count)[inverse]
    dy = ly - (np.bincount(inverse, ly) / count)[inverse]
    denominator = float(np.sum(dx * dx))
    if denominator <= 0:
        return {'power': 0., 'pixels': int(len(lx)), 'rms': None, 'origin': 'unmeasured'}
    power = float(np.sum(dx * dy) / denominator)
    rms = float(np.sqrt(np.mean((dy - power * dx) ** 2)))
    return {'power': float(np.clip(power, 0., MAX_POWER)), 'pixels': int(len(lx)), 'rms': rms, 'origin': 'fitted'}
