"""Independent per-pixel oracle for reusable, run-owned Bayer geometry."""
import numpy as np
import pytest

from planetrecon.detector import PreparedBilinearDemosaic, bilinear_demosaic


def oracle(raw, pattern, origin):
    h, w = raw.shape
    ox, oy = origin
    tiles = {'RGGB': 'RGGB', 'GRBG': 'GRBG', 'GBRG': 'GBRG', 'BGGR': 'BGGR'}[pattern]
    def colour(y, x):
        return tiles[((y+oy) % 2)*2 + (x+ox) % 2]
    output = np.zeros((h, w, 3))
    for y in range(h):
        for x in range(w):
            for channel, name in enumerate('RGB'):
                if colour(y, x) == name:
                    output[y, x, channel] = raw[y, x]
                    continue
                total, count = np.float64(0), 0
                for dy, dx in ((1,0), (-1,0), (0,1), (0,-1), (1,1), (-1,-1), (1,-1), (-1,1)):
                    yy, xx = y+dy, x+dx
                    if 0 <= yy < h and 0 <= xx < w and colour(yy, xx) == name:
                        total += raw[yy, xx]
                        count += 1
                output[y, x, channel] = total/max(count, 1)
    return output


@pytest.mark.parametrize('shape', [(1,1), (1,7), (6,1), (7,9), (8,10)])
@pytest.mark.parametrize('pattern', ['RGGB', 'GRBG', 'GBRG', 'BGGR'])
@pytest.mark.parametrize('origin', [(0,0), (1,0), (0,1), (1,1)])
def test_prepared_demosaic_matches_pixel_oracle(shape, pattern, origin):
    rng = np.random.default_rng(54)
    prepared = PreparedBilinearDemosaic(shape, pattern, origin)
    for nonfinite in (False, True):
        raw = rng.normal(100, 60, (shape[0], shape[1]*2))[:, ::2]
        if nonfinite:
            raw[0, 0] = np.nan
            raw[-1, -1] = np.inf
        before = raw.copy()
        raw.setflags(write=False)
        with np.errstate(invalid='ignore'):
            expected = oracle(raw, pattern, origin)
            actual = prepared(raw)
            ordinary = bilinear_demosaic(raw, pattern, origin)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(ordinary, expected)
        np.testing.assert_array_equal(raw, before)
        assert not np.shares_memory(actual, raw)


def test_prepared_geometry_has_bounded_private_storage():
    first = PreparedBilinearDemosaic((20, 30), 'RGGB')
    second = PreparedBilinearDemosaic((20, 30), 'BGGR')
    arrays = [a for entry in first._geometry for a in entry[1:]]
    assert sum(a.nbytes for a in arrays) == 22*20*30
    assert all(not a.flags.writeable for a in arrays)
    for a in arrays:
        assert all(not np.shares_memory(a, b) for entry in second._geometry for b in entry[1:])
    with pytest.raises(ValueError, match='shape'):
        first(np.zeros((30, 20)))
