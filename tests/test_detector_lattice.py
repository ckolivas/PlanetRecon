"""Performance paths preserve CFA sites and the local deformation inverse."""
import numpy as np
import pytest
from planetrecon.detector import BAYER_PATTERNS, cfa_labels


@pytest.mark.parametrize('pattern', list(BAYER_PATTERNS))
@pytest.mark.parametrize('shape', [(1,1),(13,22),(20,17)])
@pytest.mark.parametrize('origin', [(0,0),(1,0),(0,1),(-1,3)])
def test_cfa_lattice_matches_site_convention(pattern,shape,origin):
    expected = np.array([[BAYER_PATTERNS[pattern][(y+origin[1])%2][(x+origin[0])%2]
                          for x in range(shape[1])] for y in range(shape[0])])
    np.testing.assert_array_equal(cfa_labels(*shape,pattern,origin),expected)


