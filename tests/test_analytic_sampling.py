import numpy as np
import pytest

from tools.analytic_sampling_probe import detector_image
from tools.analytic_motion_probe import integrated_motion


@pytest.mark.parametrize('blur', [0., .4, 1.])
def test_independent_quadrature_matches_exact_pixel_integrals(blur):
    features = np.array([[31., 13.2, 12.7, .65, .8], [100., 20., 22., 4., 5.]])
    exact = detector_image(features, (40, 48), (.6, -.4), blur)
    numeric = integrated_motion(features, (40, 48), 0., 0., blur, order=8, shift_xy=(.6, -.4))
    np.testing.assert_allclose(numeric, exact, atol=1e-10, rtol=0)


def test_continuous_blur_preserves_flux_and_background():
    features = np.array([[31., 30.2, 32.7, .65, .8]])
    expected = 31*2*np.pi*.65*.8
    for blur in (0., .4, 1.):
        image = detector_image(features, (64, 64), (.6, -.4), blur)
        assert np.sum(image-3.) == pytest.approx(expected, abs=1e-10)
