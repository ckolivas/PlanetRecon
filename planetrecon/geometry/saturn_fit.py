"""Image-derived Saturn globe and visible annulus boundaries.

The silhouette's moments only initialize the fit. Limb gradients measure the
globe and outer ellipse separately; the inner edge comes from both ansae.
Opening is unsigned: this fit cannot identify which pole faces the observer.
"""
import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d, label, map_coordinates
from scipy.optimize import least_squares
from scipy.signal import find_peaks

from planetrecon.geometry.fit import fit_disc_ellipse


def _ellipse_edges(image, initial, choose, contrast, *, lock_angle=False, axis_ratio=None):
    p = np.asarray(initial, dtype=float)
    for _ in range(3):
        cx, cy, a, b, angle = p
        theta = np.linspace(0, 2*np.pi, 256, endpoint=False)
        u, v = a*np.cos(theta), b*np.sin(theta)
        use = choose(u, v)
        u, v = u[use], v[use]
        if len(u) < 15:
            raise ValueError('Too little unobscured limb for a Saturn ellipse fit.')
        nx, ny = u/a**2, v/b**2
        norm = np.hypot(nx, ny)
        nx, ny = nx/norm, ny/norm
        c, s = np.cos(angle), np.sin(angle)
        x, y = cx+u*c-v*s, cy+u*s+v*c
        dx, dy = nx*c-ny*s, nx*s+ny*c
        reach = max(3., .045*a)
        offsets = np.linspace(-reach, reach, 81)
        values = map_coordinates(image, (y[:, None]+dy[:, None]*offsets,
                                         x[:, None]+dx[:, None]*offsets),
                                 order=1, mode='nearest')
        gradient = -np.gradient(values, offsets, axis=1)
        # Prefer the exterior edge to internal ring bands. Ignore weak peaks
        # and incomplete transitions at the ends of the search interval.
        peaks = ((gradient >= np.roll(gradient, 1, axis=1))
                 & (gradient >= np.roll(gradient, -1, axis=1))
                 & (gradient > .3*gradient.max(axis=1)[:, None]))
        peaks[:, :3] = peaks[:, -3:] = False
        index = len(offsets)-1-np.argmax(peaks[:, ::-1], axis=1)
        strength = gradient[np.arange(len(index)), index]
        good = peaks.any(axis=1) & (strength > max(.008*contrast, np.median(strength)*.25))
        x = x[good]+dx[good]*offsets[index[good]]
        y = y[good]+dy[good]*offsets[index[good]]
        if len(x) < 15:
            raise ValueError('Saturn limb edges are not resolved.')

        def residual(z):
            if axis_ratio is not None:
                cx, cy, a, pa = z
                b = a*axis_ratio
            else:
                cx, cy, a, b = z[:4]
                pa = angle if lock_angle else z[4]
            u = (x-cx)*np.cos(pa)+(y-cy)*np.sin(pa)
            v = -(x-cx)*np.sin(pa)+(y-cy)*np.cos(pa)
            return (np.hypot(u/a, v/b)-1)*min(a, b)

        lo = [cx-2, cy-2, a*.85, b*.8, angle-.08]
        hi = [cx+2, cy+2, a*1.15, b*1.2, angle+.08]
        indices = [0, 1, 2, 4] if axis_ratio is not None else list(range(4 if lock_angle else 5))
        fit = least_squares(residual, p[indices], bounds=(np.array(lo)[indices], np.array(hi)[indices]),
                            loss='soft_l1', f_scale=.3, max_nfev=100)
        if not fit.success or np.median(abs(residual(fit.x))) > 1.5:
            raise ValueError('Saturn limb ellipse fit is inconsistent.')
        if axis_ratio is not None:
            p = np.array([*fit.x[:3], fit.x[2]*axis_ratio, fit.x[3]])
        else:
            p = np.r_[fit.x, angle] if lock_angle else fit.x
    return p


def fit_saturn_geometry(image, *, opening_rad=None):
    """Return pixel-centre geometry, or an unresolved fit without guessed radii.

    Input is a mono/RGB estimation plane, never an unconverted Bayer mosaic.
    Radii refer to visible edges, not invisible material below the noise floor.
    """
    img = np.asarray(image, dtype=float)
    if img.ndim == 3 and img.shape[-1] == 3:
        img = .25*img[..., 0]+.5*img[..., 1]+.25*img[..., 2]
    failure = {'ok': False, 'ring_inner': None, 'ring_outer': None,
               'opening_rad': 0., 'degeneracy': ('saturn_geometry_unresolved',)}
    if img.ndim != 2 or min(img.shape) < 12 or not np.isfinite(img).all():
        return {**failure, 'reason': 'No finite Saturn estimation plane.'}
    try:
        return _fit_saturn(img, opening_rad)
    except ValueError as exc:
        return {**failure, 'reason': str(exc)}


def _fit_saturn(img, opening_rad):
    image = gaussian_filter(img, .7)
    sky, peak = np.percentile(image, [10, 99.8])
    contrast = peak-sky
    if contrast <= 0:
        raise ValueError('No Saturn contrast.')
    labels, _ = label(image > sky+.18*contrast)
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    mask = labels == counts.argmax()
    if counts.max() < 40 or mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any():
        raise ValueError('Saturn is too small or clipped by the capture boundary.')
    base = fit_disc_ellipse(np.where(mask, image, sky))
    cx, cy, angle = base['cx']-.5, base['cy']-.5, base['pa_rad']
    yy, xx = np.nonzero(mask)
    c, s = np.cos(angle), np.sin(angle)
    u, v = (xx-cx)*c+(yy-cy)*s, -(xx-cx)*s+(yy-cy)*c
    outer = float(np.percentile(abs(u), 99.7))
    # The first globe limb along the equator is inside the ring gap. Its
    # measured position initializes the globe, rather than the ring diameter.
    radii = np.linspace(.2*outer, .63*outer, 180)
    sides = np.array([-1, 1])[:, None]
    profiles = map_coordinates(image, (cy+s*sides*radii, cx+c*sides*radii), order=1)
    globe_a = float(np.median(radii[np.argmin(np.gradient(profiles, radii, axis=1), axis=1)]))
    envelope = []
    for pos in np.linspace(.6*outer, .85*outer, 12):
        use = abs(abs(u)-pos) < 1
        if use.any():
            envelope.append(np.max(abs(v[use]))/np.sqrt(1-(pos/outer)**2))
    if len(envelope) < 6:
        raise ValueError('Both Saturn ansae must be visible.')
    ring_b = float(np.median(envelope))
    ratio = None if opening_rad is None else abs(float(np.sin(opening_rad)))
    if ratio is not None:
        if not np.isfinite(ratio) or not .04 < ratio < .85:
            raise ValueError('Ring opening is too close to edge-on/pole-on for boundary estimation.')
        ring_b = outer*ratio
    ring = _ellipse_edges(image, [cx, cy, outer, ring_b, angle],
                          lambda u, v: abs(u) > .65*outer, contrast, axis_ratio=ratio)
    cx, cy, outer, ring_b, angle = ring
    if not .04 < ring_b/outer < .85 or ring_b < 2:
        raise ValueError('Ring opening is unresolved or too close to edge-on/pole-on.')
    # At low opening the minor-axis edge is undersampled. Measure the radial
    # extent again at the two tips, where detector sampling resolves it best.
    radii = np.arange(.85*outer, 1.2*outer, .2)
    angles = np.r_[np.linspace(-.15, .15, 13), np.linspace(np.pi-.15, np.pi+.15, 13)]
    u, v = np.cos(angles)[:, None]*radii, np.sin(angles)[:, None]*radii*ring_b/outer
    c, s = np.cos(angle), np.sin(angle)
    values = map_coordinates(image, (cy+u*s+v*c, cx+u*c-v*s), order=1)
    tips = []
    for side in np.array_split(values, 2):
        gradient = -np.gradient(gaussian_filter1d(np.median(side, axis=0), 2), radii)
        peaks, _ = find_peaks(gradient, height=.25*gradient.max(), prominence=.12*gradient.max())
        if not len(peaks) or gradient.max() < .008*contrast:
            raise ValueError('Outer ring edge is not resolved in both ansae.')
        tips.append(float(radii[peaks[-1]]))
    if abs(tips[0]-tips[1]) > max(3., .06*outer):
        raise ValueError('Outer ring edges disagree between ansae.')
    measured_outer = float(np.mean(tips))
    ring_b *= measured_outer/outer
    outer = measured_outer
    # Average within each ansa, then require the same inner edge on both
    # sides. Sharing the outer ellipse prevents an unconstrained inner tilt.
    radii = np.arange(globe_a*1.08, outer*.85, .2)
    if len(radii) < 12:
        raise ValueError('No resolved gap between globe and rings.')
    angles = np.r_[np.linspace(-.4, .4, 17), np.linspace(np.pi-.4, np.pi+.4, 17)]
    u, v = np.cos(angles)[:, None]*radii, np.sin(angles)[:, None]*radii*ring_b/outer
    c, s = np.cos(angle), np.sin(angle)
    values = map_coordinates(image, (cy+u*s+v*c, cx+u*c-v*s), order=1)
    inner_edges = []
    for side in np.array_split(values, 2):
        profile = np.median(side, axis=0)
        gradient = np.gradient(gaussian_filter1d(profile, 2), radii)
        peaks, _ = find_peaks(gradient, height=.25*gradient.max(), prominence=.12*gradient.max())
        if not len(peaks) or gradient.max() < .008*contrast:
            raise ValueError('Inner ring edge is not resolved in both ansae.')
        inner_edges.append(float(radii[peaks[0]]))
    if abs(inner_edges[0]-inner_edges[1]) > max(3., .06*outer):
        raise ValueError('Inner ring edges disagree between ansae.')
    inner = float(np.mean(inner_edges))
    inner_b = inner*ring_b/outer
    globe = _ellipse_edges(image, [cx, cy, globe_a, globe_a*.9, angle],
                           lambda u, v: ((np.hypot(u/outer, v/ring_b) > 1.12)
                                         | (np.hypot(u/inner, v/inner_b) < .88)),
                           contrast, lock_angle=True)
    gx, gy, globe_a, globe_b, _ = globe
    if (not 4 < globe_a < inner < outer or not .7 < globe_b/globe_a < 1.05
            or np.hypot(gx-cx, gy-cy) > max(2., .04*outer)):
        raise ValueError('Globe and ring boundary fits are inconsistent.')
    return {'ok': True, 'cx': float(gx+.5), 'cy': float(gy+.5),
            'radius': float(globe_a), 'semi_major': float(globe_a),
            'semi_minor': float(globe_b), 'pa_rad': float(angle),
            'flattening': float(max(0., 1-globe_b/globe_a)),
            'ring_inner': inner, 'ring_outer': float(outer),
            'opening_rad': float(np.arcsin(ring_b/outer)),
            'opening_origin': 'viewing latitude' if ratio is not None else 'ring ellipse',
            'degeneracy': ('low_opening',) if ring_b/outer < .2 else (),
            'method': 'separate globe limb and ring boundary gradients v1'}
