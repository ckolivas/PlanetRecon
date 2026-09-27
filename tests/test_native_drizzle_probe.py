import numpy as np
import pytest
from scipy.ndimage import map_coordinates

from tools.native_drizzle_probe import deposit


@pytest.mark.parametrize('shift', [(0., 0.), (.25, .5), (-.7, .35), (3.2, -4.4)])
def test_unit_drop_matches_bilinear_pull_with_boundary_coverage(shift):
    image = np.random.default_rng(10).normal(size=(25, 31))
    yy, xx = np.indices(image.shape, dtype=float)
    coords = np.array([yy+shift[1], xx+shift[0]])
    for result, source in zip(deposit(image, shift, 1.), (image, np.ones_like(image))):
        expected = map_coordinates(source, coords, order=1, mode='grid-constant', prefilter=False)
        np.testing.assert_allclose(result, expected, rtol=0, atol=1e-13)


@pytest.mark.parametrize('fraction', [0., .25, .5, 1.])
def test_interior_flux_is_preserved_and_integer_placement_does_not_blur(fraction):
    impulse = np.zeros((25, 31))
    impulse[12, 15] = 100.
    result, _ = deposit(impulse, (.35, -.7), fraction)
    assert result.sum() == pytest.approx(100., abs=1e-12)
    result, support = deposit(impulse, (2., -1.), fraction)
    assert result[13, 13] == pytest.approx(100.)
    assert np.count_nonzero(result) == 1
    assert support[13, 13] == pytest.approx(1.)
