"""Offline sidereal rotation presets; camera orientation remains observational."""
import math

# JPL Solar System Dynamics, Planetary Physical Parameters, retrieved 2026-09-10.
# Signed sidereal days; Venus and Uranus rotate retrograde relative to the
# conventional north pole. These are bulk/body periods, not cloud-wind fits.
SOURCE_URL = 'https://ssd.jpl.nasa.gov/planets/phys_par.html'
PERIOD_DAYS = {
    'mercury': 58.6462,
    'venus': -243.018,
    'earth': .99726968,
    'mars': 1.02595676,
    'jupiter': .41354,
    'saturn': .44401,
    'uranus': -.71833,
    'neptune': .67125,
}


def rotation_preset(planet, *, reverse=False, radius_px=None):
    period = PERIOD_DAYS[planet]
    rate = 2*math.pi / (period*86400) * (-1 if reverse else 1)
    return {
        'origin': 'planet_preset', 'planet': planet, 'source': SOURCE_URL,
        'sidereal_period_days': period, 'surface_rate_rad_s': rate,
        'reversed': reverse, 'equatorial_radius_px': radius_px,
        'equator_on_speed_px_s': None if radius_px is None else abs(rate)*radius_px,
        'note': 'Bulk sidereal rotation, not measured cloud motion. Pole orientation and viewing latitude '
                'are separate inputs. Positive rate at pole PA 0 moves central texture left; '
                'reverse for the opposite apparent direction. Radius scales pixel motion, not the period.',
    }
