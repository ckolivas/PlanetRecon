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
from planetrecon.geometry.shape import estimate_flattening


SATURN_VIEW_DEPENDENT_KEYS = ('sub_obs_lat_rad', 'surface_rate_rad_s',
                             'field_rate_rad_s', 'pole_pa_rad')


def without_assumed_saturn_view(estimate):
    """An inferred equator-on globe view is not a signed Saturn ring opening.

    Existing discovery records only suggest latitude via that assumption.
    Preserve the original record and independent centre/size suggestions.
    """
    suggestions = estimate.get('suggestions', {})
    if 'sub_obs_lat_rad' not in suggestions:
        return estimate
    return {**estimate,
            'suggestions': {key: value for key, value in suggestions.items()
                            if key not in SATURN_VIEW_DEPENDENT_KEYS},
            'status': 'unresolved',
            'notes': [*estimate.get('notes', []),
                      'Saturn needs a supplied signed viewing latitude; assumed equator-on latitude and dependent motion prefills are not applied.']}


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
    from planetrecon.geometry.viewing import resolve_viewing
    # Discover viewing geometry without changing the chosen stacking mode;
    # a temporary surface mode would invalidate local-alignment settings.
    effective, viewing = resolve_viewing(source, config, strict=False, for_preprocessing=True)
    report = _discover_geometry(source, effective, selection, calibration, should_cancel)
    if viewing is not None:
        report['viewing_geometry'] = viewing
        if 'sub_obs_lat_rad' in viewing:
            report['notes'].append(
                f"Automatic {viewing['planet'].title()} viewing latitude: "
                f"{np.degrees(viewing['sub_obs_lat_rad']):+.4f}° from JPL Horizons at capture UTC; no Earth-site coordinates needed.")
        else:
            report['notes'].extend(viewing['notes'])
            report['suggestions'] = {key: value for key, value in report['suggestions'].items()
                                     if key not in (*SATURN_VIEW_DEPENDENT_KEYS, 'flattening')}
            report['status'] = 'unresolved'
    return report


def _discover_geometry(source, config, selection, calibration=None, should_cancel=None):
    """Three short averages aligned to the best retained frame; at most 97 reads."""
    from dataclasses import replace
    from planetrecon.geometry.pose import capture_exposure
    exposure = capture_exposure(source, config.exposure_s)
    config = replace(config, exposure_s=exposure['value_s'])
    shape = estimate_flattening(selection, config)
    report = {'method': 'projected spherical texture motion v2', 'suggestions': dict(shape['suggestions']),
              'status': 'unresolved', 'notes': [], 'sample_indices': []}
    report['exposure'] = exposure
    report['flattening_estimate'] = {k: v for k, v in shape.items() if k != 'suggestions'}
    report['notes'].extend(shape['notes'])
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
    def read_plane(index):
        if should_cancel and should_cancel():
            raise InterruptedError('geometry discovery cancelled')
        frame, _ = apply_calibration(source.read_raw(int(index)), calibration)
        plane, bin_scale = measurement_plane(frame, source.color_mode())
        factor = min(1., 256 / max(plane.shape))
        if factor < 1:
            yy, xx = np.meshgrid(np.arange(int(plane.shape[0]*factor))/factor,
                                 np.arange(int(plane.shape[1]*factor))/factor, indexing='ij')
            plane = map_coordinates(plane, (yy, xx), order=1, mode='nearest')
        return plane, bin_scale, bin_scale / factor
    reference_index = selection.best_reference_index
    report['reference_index'] = reference_index
    anchor, bin_scale, scale = read_plane(reference_index)
    for group in np.array_split(accepted, 3):
        # Compact windows avoid averaging away the very motion being measured.
        middle = len(group)//2
        window = group[max(0, middle-64):middle+64]
        chosen = np.sort(window[np.argsort(selection.measurements[window, 0], kind='stable')[-32:]])
        report['sample_indices'].append(chosen.tolist())
        total = np.zeros_like(anchor)
        for index in chosen:
            if index == reference_index:
                total += anchor
            else:
                plane, _, _ = read_plane(index)
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
    if config.geometry_mode == 'saturn':
        from planetrecon.geometry.saturn_fit import fit_saturn_geometry
        ring_fits = [fit_saturn_geometry(average, opening_rad=config.sub_obs_lat_rad) for average in averages]
        report['saturn_geometry'] = {'fits': ring_fits, 'status': 'unresolved',
                                    'detector_pixels_per_sample': float(scale)}
        good = [fit for fit in ring_fits if fit['ok']]
        keys = {'equatorial_radius_px': 'radius', 'ring_inner_radius_px': 'ring_inner',
                'ring_outer_radius_px': 'ring_outer'}
        if len(good) >= 2:
            values = np.array([[fit[key] for key in keys.values()] for fit in good])
            typical = np.median(values, axis=0)
            stable = np.all(np.ptp(values, axis=0) < np.maximum(2., .06*typical))
            if stable:
                report['saturn_geometry']['status'] = 'estimated'
                for key, value in zip(keys, typical):
                    suggestions[key] = float(value*scale)
                suggestions['field_center_x'] = float(np.median([fit['cx']-.5 for fit in good])*scale+.5*bin_scale)
                suggestions['field_center_y'] = float(np.median([fit['cy']-.5 for fit in good])*scale+.5*bin_scale)
                # Recover the axis at the output epoch, retaining the chosen
                # pole branch. An ellipse alone does not identify north.
                angles = np.array([fit['pa_rad'] for fit in good])
                fit_times = np.array([t for t, fit in zip(sample_times, ring_fits) if fit['ok']])
                field_rate = config.field_rate_rad_s
                if len(good) == 3 and time_origin != 'inferred' and np.ptp(fit_times) > 0:
                    unwrapped = np.unwrap(2*angles)/2
                    elapsed = fit_times-np.mean(fit_times)
                    slope, intercept = np.polyfit(elapsed, unwrapped, 1)
                    residual = float(np.max(abs(unwrapped-(intercept+slope*elapsed))))
                    # Ring axes are measurable even with no azimuthal texture.
                    # Reject inconsistent or large inter-window rotations; the
                    # ellipse alone cannot resolve a half-turn alias.
                    if (residual < max(.003, 1/typical[-1])
                            and np.max(abs(np.diff(unwrapped))) < .15):
                        report['saturn_geometry']['field_axis_residual_rad'] = residual
                        suggestions['field_rate_rad_s'] = -float(slope)  # detector y is downward
                        report['roll_resolved'] = True
                        report['roll_direction'] = ('counter-clockwise' if slope < 0 else 'clockwise')
                        if field_rate is None:
                            field_rate = suggestions['field_rate_rad_s']
                        report['notes'].append('Field rotation measured from the ring-axis change between timed averages; '
                                               'a near-zero rate means the ring orientation stayed stable.')
                if field_rate is not None and time_origin != 'inferred':
                    angles += field_rate*(fit_times-config.reference_epoch_s)
                pa = float(np.angle(np.mean(np.exp(2j*angles)))/2 + config.field_angle0_rad)
                if np.cos(pa-config.pole_pa_rad) < 0:
                    pa += np.pi
                suggestions['pole_pa_rad'] = pa
                report['orientation_origin'] = 'measured ring major axis (pole branch preserved)'
                latitude = config.sub_obs_lat_rad
                ratio = float(np.median([fit['semi_minor']/fit['semi_major'] for fit in good]))
                if latitude is not None and abs(np.cos(latitude)) > .2:
                    polar2 = (min(ratio, 1.)**2-np.sin(latitude)**2)/np.cos(latitude)**2
                    if .75**2 < polar2 <= 1:
                        suggestions['flattening'] = float(1-np.sqrt(polar2))
                        report['flattening_estimate'] = {
                            'status': 'estimated', 'method': 'separate exposed Saturn globe limb',
                            'apparent_axis_ratio': ratio, 'latitude_rad': float(latitude),
                            'flattening': suggestions['flattening']}
                report['status'] = 'estimated'
                report['notes'].append('Globe and visible ring radii measured separately from limb edges in aligned averages. '
                    'Faint rings below the capture noise floor are not measured. Ring opening is unsigned; '
                    'the viewing latitude uses capture UTC or your override. Surface spin uses the selected '
                    'planet preset or an explicit rate; ring structure is not used to infer globe spin.')
                return report
        reasons = list(dict.fromkeys(fit.get('reason', '') for fit in ring_fits if not fit['ok']))
        report['notes'].append('Saturn globe/ring boundaries could not be measured consistently. ' + ' '.join(reasons))
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
            if config.geometry_mode == 'saturn':
                report['notes'].append('Saturn needs a supplied signed viewing latitude; an equator-on assumption would incorrectly make the rings edge-on.')
                return report
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
