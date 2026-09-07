"""Bounded, observation-only geometry suggestions; never changes an active job.

Fit differential motion across a projected sphere, translation and image roll
jointly. Surface drift varies with visible depth, unlike tracking translation.
This is a small-angle initializer, not ephemerides or a physical pole identity.
"""
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates, shift
from scipy.optimize import least_squares

from planetrecon.calibration import apply_calibration
from planetrecon.geometry.fit import fit_disc_ellipse
from planetrecon.geometry.pose import source_times_s
from planetrecon.pipeline.align import phase_correlation_shift
from planetrecon.pipeline.preprocess import measurement_plane


def fit_projected_motion(reference, moving, center, radius, should_cancel=None):
    """Return surface displacement (x,y), CCW roll and nuisance translation.

    Coordinates are detector x-right/y-down. In GlobeParams' convention a
    positive surface rate at PA=0 moves texture to the left. The surface
    displacement coefficients are radius * projected angular displacement.
    """
    y, x = np.indices(reference.shape, dtype=float)
    x -= center[0]
    y -= center[1]
    r2 = (x*x + y*y) / radius**2
    mask = r2 < .85**2
    if mask.sum() < 200:
        return None
    # Suppress lighting gradients and noise; keep measured spatial texture.
    ref = gaussian_filter(reference, .7) - gaussian_filter(reference, 4)
    mov = gaussian_filter(moving, .7) - gaussian_filter(moving, 4)
    contrast = float(np.std(ref[mask]))
    if contrast < max(float(np.ptp(reference)) * .001, 1e-12):
        return None
    ref /= contrast
    mov /= max(float(np.std(mov[mask])), 1e-12)
    yy, xx = np.nonzero(mask)
    # Bound solver work independently of detector resolution.
    step = max(1, int(np.ceil(len(xx) / 6000)))
    yy, xx = yy[::step], xx[::step]
    u, v = x[yy, xx], y[yy, xx]
    depth = np.sqrt(1 - r2[yy, xx])
    target = ref[yy, xx]
    def residual(p):
        if should_cancel and should_cancel():
            raise InterruptedError('geometry discovery cancelled')
        a, b, roll, tx, ty = p
        return map_coordinates(mov, (yy + b*depth - roll*u + ty,
                                     xx + a*depth + roll*v + tx),
                               order=3, mode='nearest') - target
    dx, dy = phase_correlation_shift(reference, moving)
    initial = [0., 0., 0., np.clip(dx, -7, 7), np.clip(dy, -7, 7)]
    bounds = np.array([.3*radius, .3*radius, .15, 8., 8.])
    fit = least_squares(residual, initial, bounds=(-bounds, bounds),
                        loss='soft_l1', f_scale=.25, max_nfev=80,
                        ftol=1e-5, xtol=1e-5, gtol=1e-5,
                        x_scale=[radius, radius, 1, 1, 1])
    if not fit.success or np.any(abs(fit.x) > .98*bounds):
        return None
    singular = np.linalg.svd(fit.jac, compute_uv=False)
    if singular[-1] <= 0 or singular[0]/singular[-1] > 1e5:
        return None
    variance = float(np.mean(residual(fit.x)**2))
    covariance = np.linalg.pinv(fit.jac.T @ fit.jac) * variance
    return {'parameters': fit.x, 'stderr': np.sqrt(np.maximum(np.diag(covariance), 0)),
            'rms': np.sqrt(variance)}


def discover_geometry(source, config, selection, calibration=None, should_cancel=None):
    """Three short, aligned averages across accepted frames; at most 96 reads."""
    report = {'method': 'projected spherical texture motion v1', 'suggestions': {},
              'status': 'unresolved', 'notes': [], 'sample_indices': []}
    accepted = np.flatnonzero(selection.accepted)
    if len(accepted) < 12:
        report['notes'].append('At least 12 accepted frames are needed to estimate rotation.')
        return report
    try:
        times, time_origin = source_times_s(source, cadence_s=config.cadence_s)
    except ValueError as exc:
        report['notes'].append(str(exc))
        return report
    report['time_origin'] = time_origin
    averages, sample_times, centres = [], [], []
    scale = None
    for group in np.array_split(accepted, 3):
        # Compact windows avoid averaging away the very motion being measured.
        middle = len(group)//2
        window = group[max(0, middle-64):middle+64]
        chosen = np.sort(window[np.argsort(selection.measurements[window, 0], kind='stable')[-32:]])
        report['sample_indices'].append(chosen.tolist())
        total = None
        for index in chosen:
            if should_cancel and should_cancel():
                raise InterruptedError('geometry discovery cancelled')
            frame, _ = apply_calibration(source.read_raw(int(index)), calibration)
            plane, bin_scale = measurement_plane(frame, source.color_mode())
            factor = min(1., 256 / max(plane.shape))
            if factor < 1:
                yy, xx = np.meshgrid(np.arange(int(plane.shape[0]*factor))/factor,
                                     np.arange(int(plane.shape[1]*factor))/factor, indexing='ij')
                plane = map_coordinates(plane, (yy, xx), order=1, mode='nearest')
            scale = bin_scale / factor
            if total is None:
                anchor = plane
                total = plane.copy()
            else:
                dx, dy = phase_correlation_shift(anchor, plane)
                total += shift(plane, (-dy, -dx), order=1, prefilter=False, mode='nearest')
        average = total / len(chosen)
        disc = fit_disc_ellipse(average)
        if not disc['ok']:
            report['notes'].append('No stable disc centre could be fitted.')
            return report
        averages.append(average)
        centres.append((disc['cx']-.5, disc['cy']-.5))
        sample_times.append(float(np.mean(times[chosen])) + .5*config.exposure_s)
    stats = selection.summary['statistics']
    width, height = stats['width_px']['mean'], stats['height_px']['mean']
    angle = selection.measurements[accepted, 3]
    if config.field_rate_rad_s is not None and time_origin != 'inferred':
        angle = angle + config.field_rate_rad_s * (times[accepted]+.5*config.exposure_s-config.reference_epoch_s)
    axis = np.mean(np.exp(2j*angle))
    report['silhouette_axis_rad'] = float(np.angle(axis)/2)
    report['silhouette_axis_coherence'] = float(abs(axis))
    report['notes'].append('Pole direction is an image-coordinate convention, not identification of physical north.')
    suggestions = report['suggestions']
    # This is only an initial pole-axis hypothesis. Surface tracking supersedes
    # it when resolved; a crescent can exchange the major and polar axes.
    if abs(axis) > .9 and (1.02 < width/height < 1.35 or config.geometry_mode == 'saturn'):
        suggestions['pole_pa_rad'] = report['silhouette_axis_rad'] + config.field_angle0_rad
        report['orientation_origin'] = 'silhouette (major axis assumed equatorial)'
        report['notes'].append('Initial pole assumes the stable silhouette major axis is equatorial; phase can bias or interchange the axes.')
    suggestions.update(field_center_x=float(np.mean(centres, axis=0)[0])*scale+.5*bin_scale,
                       field_center_y=float(np.mean(centres, axis=0)[1])*scale+.5*bin_scale)
    # Rings/crescents do not identify a globe radius from the whole silhouette.
    if width/height > 1.35 and config.equatorial_radius_px is None:
        report['notes'].append('Rings or strong phase: supply the globe radius to estimate surface rotation.')
        return report
    radius_px = config.equatorial_radius_px or width/2
    if config.equatorial_radius_px is None:
        suggestions['equatorial_radius_px'] = float(radius_px)
    radius = radius_px / scale
    fits = []
    for i in (0, 1):
        dt = sample_times[i+1] - sample_times[i]
        if dt <= 0:
            return report
        # Fit nuisance tracking translation independently for each time interval.
        dx, dy = np.subtract(centres[i+1], centres[i])
        moving = shift(averages[i+1], (-dy, -dx), order=1, prefilter=False, mode='nearest')
        fit = fit_projected_motion(averages[i], moving, centres[i], radius, should_cancel)
        if fit is None:
            report['notes'].append('Texture or motion fit is insufficiently constrained.')
            return report
        fits.append(fit)
    deltas = np.diff(sample_times)
    rates = np.array([f['parameters'][:3]/dt for f, dt in zip(fits, deltas)])
    errors = np.array([f['stderr'][:3]/dt for f, dt in zip(fits, deltas)])
    report['fit_rms'] = [float(f['rms']) for f in fits]
    report['sample_times'] = sample_times
    rate = np.mean(rates, axis=0)
    surface = rate[:2]
    amplitude = float(np.linalg.norm(surface))
    uncertainty = max(float(np.linalg.norm(errors[:, :2], axis=1).max()),
                      float(np.linalg.norm(rates[0, :2]-rates[1, :2])))
    surface_ok = (amplitude > 3*uncertainty and amplitude*min(deltas) > .25
                  and max(report['fit_rms']) < .8)
    roll_error = max(float(errors[:, 2].max()), float(abs(rates[0, 2]-rates[1, 2])))
    roll_ok = abs(rate[2]) > 3*roll_error and abs(rate[2])*min(deltas)*radius > .25
    report['projected_surface_rad_per_time_unit'] = amplitude / radius
    report['projected_surface_uncertainty_rad_per_time_unit'] = uncertainty / radius
    report['apparent_roll_rad_per_time_unit'] = float(rate[2])
    report['apparent_roll_uncertainty_rad_per_time_unit'] = roll_error
    report['surface_resolved'] = bool(surface_ok)
    report['roll_resolved'] = bool(roll_ok)
    if surface_ok:
        pa = float((np.arctan2(surface[1], surface[0]) + np.pi/2) % np.pi - np.pi/2)
        known_field = config.field_angle0_rad
        if time_origin != 'inferred' and config.field_rate_rad_s is not None:
            known_field += config.field_rate_rad_s * (sample_times[1]-config.reference_epoch_s)
        if np.cos(pa - (config.pole_pa_rad-known_field)) < 0:
            pa += np.pi
        signed_rate = -float(surface @ np.array([np.cos(pa), np.sin(pa)])) / radius
        report['surface_direction'] = ('left' if surface[0] < 0 else 'right') + (' / up' if surface[1] < 0 else ' / down')
        lat = config.sub_obs_lat_rad
        if lat is None:
            lat = 0.
            suggestions['sub_obs_lat_rad'] = 0.
            report['notes'].append('Surface rate assumes an equator-on view (observer latitude 0°); true rate may be larger.')
        if abs(np.cos(lat)) < .2:
            report['notes'].append('Near pole-on view: surface rate cannot be recovered reliably.')
            return report
        signed_rate /= np.cos(lat)
        field_rate = float(rate[2] + signed_rate*np.sin(lat))
        epoch_angle = config.field_angle0_rad
        if time_origin != 'inferred':
            epoch_angle += (config.field_rate_rad_s if config.field_rate_rad_s is not None else field_rate) * (sample_times[1]-config.reference_epoch_s)
            suggestions['surface_rate_rad_s'] = float(signed_rate)
        suggestions['pole_pa_rad'] = float((pa + epoch_angle + np.pi) % (2*np.pi) - np.pi)
        report['orientation_origin'] = 'surface texture motion'
        report['status'] = 'estimated'
    else:
        field_rate = float(rate[2])
        report['notes'].append('Surface rotation is unresolved; no surface-rate prefill.')
    if roll_ok:
        report['roll_direction'] = 'counter-clockwise' if rate[2] > 0 else 'clockwise'
        # LOS spin and camera roll are degenerate without a viewing latitude.
        if time_origin != 'inferred' and (config.sub_obs_lat_rad is not None or not surface_ok):
            suggestions['field_rate_rad_s'] = field_rate
        report['status'] = 'estimated'
        if not surface_ok:
            report['notes'].append('Image roll may include line-of-sight spin; the field-rate suggestion treats it as camera rotation.')
    if time_origin == 'inferred':
        report['notes'].append('No timestamps or cadence: direction only; rates in degrees/second are not inferred.')
    return report
