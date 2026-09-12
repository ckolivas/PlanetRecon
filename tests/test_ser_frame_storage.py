"""Decoded frames retain their own bytes across seeks and source closure."""
import numpy as np
import pytest

from planetrecon.io.ser import SERSource, write_ser, COLOR_MONO, COLOR_RGGB, COLOR_BGR


@pytest.mark.parametrize('depth', [8, 12, 16])
@pytest.mark.parametrize('flag', [0, 1])
@pytest.mark.parametrize('convention', ['ecosystem', 'spec'])
@pytest.mark.parametrize('colour', [COLOR_MONO, COLOR_RGGB, COLOR_BGR])
def test_frame_storage_survives_reads_and_close(tmp_path, depth, flag, convention, colour):
    shape = (3, 7, 9, 3) if colour == COLOR_BGR else (3, 7, 9)
    frames = np.random.default_rng(22).integers(0, 1 << depth, shape, dtype=np.uint16)
    path = write_ser(tmp_path/'storage.ser', frames, pixel_depth=depth, color_id=colour,
                     little_endian_flag=flag, endian_convention=convention)
    with SERSource(path, endian_convention=convention) as source:
        decoded = [source.read_raw(i) for i in (2, 0, 1)]
    for frame, index in zip(decoded, (2, 0, 1)):
        expected = frames[index, ..., ::-1] if colour == COLOR_BGR else frames[index]
        assert frame.dtype == np.uint16 and frame.dtype.isnative
        np.testing.assert_array_equal(frame, expected)
        assert not any(np.shares_memory(frame, other) for other in decoded if other is not frame)
