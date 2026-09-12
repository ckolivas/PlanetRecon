"""Translation fast paths preserve interpolation, boundaries and ownership."""
import numpy as np
import pytest
from scipy.ndimage import map_coordinates
from planetrecon.pipeline.local_align import pull


@pytest.mark.parametrize('shape', [(17, 23), (17, 23, 3), (1, 13), (11, 1, 2)])
@pytest.mark.parametrize('shift', [(0., 0.), (.375, -.625), (-2., 3.), (22.5, -16.75), (40., 40.)])
@pytest.mark.parametrize('nonfinite', [False, True])
def test_scalar_pull_matches_pixel_coordinate_oracle(shape, shift, nonfinite):
    image = np.random.default_rng(73).normal(size=shape)
    if nonfinite:
        image.flat[0] = np.nan
        image.flat[-1] = np.inf
    original = image.copy()
    y, x = np.indices(shape[:2], dtype=float)
    def sample(plane):
        return map_coordinates(plane, [y+shift[1], x+shift[0]], order=1,
                               mode='grid-constant', prefilter=False)
    expected = (sample(image) if image.ndim == 2 else
                np.stack([sample(image[..., c]) for c in range(image.shape[2])], axis=-1))
    actual = pull(image, shift)
    np.testing.assert_array_equal(actual, expected)
    assert not np.shares_memory(actual, image)
    actual.fill(9.)
    np.testing.assert_array_equal(image, original)
