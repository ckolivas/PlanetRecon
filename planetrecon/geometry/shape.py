"""Apparent silhouette flattening and conditional oblate-globe interpretation."""
import numpy as np


def estimate_flattening(selection, config):
    """Infer c/a from the projected minor/major ratio and viewing latitude.

    For an axisymmetric oblate globe, q² = sin²(B) + (c/a)² cos²(B).
    Unknown latitude permits only apparent flattening (the equator-on case).
    Illuminated silhouettes cannot independently identify phase or ring light.
    """
    report = {'status': 'unresolved', 'suggestions': {}, 'notes': []}
    values = selection.measurements[selection.accepted]
    if len(values) < 12:
        report['notes'].append('At least 12 retained silhouettes are needed for flattening.')
        return report
    ratios = values[:, 2] / values[:, 1]
    ratio = float(np.median(ratios))
    scatter = float(1.4826*np.median(abs(ratios-ratio)))
    report.update(apparent_axis_ratio=ratio, apparent_flattening=float(max(0., 1-ratio)),
                  axis_ratio_scatter=scatter, n_frames=len(values))
    if config.geometry_mode == 'saturn' or ratio < .75:
        report['notes'].append('Ring or strongly phased silhouettes do not determine globe flattening.')
        return report
    if not 0 < ratio <= 1.02 or scatter > .03:
        report['notes'].append('Silhouette axis ratio is insufficiently stable for flattening.')
        return report
    latitude = config.sub_obs_lat_rad
    if latitude is None:
        latitude = 0.
        report['status'] = 'apparent'
        report['notes'].append('Flattening assumes an equator-on view; unknown viewing latitude prevents a unique physical flattening.')
    else:
        report['status'] = 'estimated'
    s2, c2 = np.sin(latitude)**2, np.cos(latitude)**2
    polar2 = (min(ratio, 1.)**2-s2)/max(c2, 1e-30)
    if c2 < .04 or polar2 <= 0:
        report['status'] = 'unresolved'
        report['notes'].append('Viewing latitude is incompatible with the silhouette or too close to pole-on.')
        return report
    flattening = float(1-np.sqrt(polar2))
    report.update(flattening=flattening, latitude_rad=float(latitude),
                  flattening_scatter=float(ratio*scatter/(c2*np.sqrt(polar2))))
    report['suggestions']['flattening'] = flattening
    report['notes'].append('Assumes an oblate globe with its major silhouette axis equatorial; phase, limb darkening and seeing can bias the estimate.')
    return report
