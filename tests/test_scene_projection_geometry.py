import numpy as np
import pytest
from tools.scene_projection_geometry import projection_geometry


def test_projection_geometry_counts_released_and_newly_bound_coordinates():
    x = np.array([0., 2., 0., 3.]); d = np.array([1., -4., -1., 2.])
    a, b = x.copy(), d.copy()
    info = projection_geometry(x, d)
    assert info['crossing_fraction'] == .5
    assert info['newly_bound_fraction'] == info['released_fraction'] == .25
    assert info['removed_direction_relative_norm'] == pytest.approx(np.sqrt(5/22))
    assert info['projected_step_relative_norm'] == pytest.approx(np.sqrt(9/22))
    np.testing.assert_array_equal(x, a); np.testing.assert_array_equal(d, b)
    assert projection_geometry(x, np.zeros(4))['removed_direction_relative_norm'] == 0.


@pytest.mark.parametrize('x,d', [([-1.],[1.]), ([0.],[np.nan]), ([0.,1.],[1.])])
def test_invalid_projection_inputs_rejected(x,d):
    with pytest.raises(ValueError): projection_geometry(x,d)
