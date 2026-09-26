import numpy as np
import pytest

from planetrecon.io.ser import write_ser
from tools.stack_transfer_probe import integer_translate, probe_shifts, verify_capture


def test_shifts_are_balanced_and_start_unshifted():
    shifts = probe_shifts()
    assert shifts.shape == (256, 2)
    np.testing.assert_array_equal(shifts[0], [0, 0])
    np.testing.assert_array_equal(shifts.sum(axis=0), [0, 0])
    assert np.max(np.abs(shifts)) == 3
    np.testing.assert_array_equal(shifts, probe_shifts())


@pytest.mark.parametrize('dy,dx', [(0, 0), (2, -3), (-3, 2)])
def test_motion_preserves_pixels_and_does_not_wrap(dy, dx):
    truth = np.arange(256, dtype=np.uint8).reshape(16, 16)
    shifted = integer_translate(truth, dy, dx)
    restored = integer_translate(shifted, -dy, -dx)
    np.testing.assert_array_equal(restored[3:-3, 3:-3], truth[3:-3, 3:-3])
    assert shifted.dtype == truth.dtype
    if dy > 0:
        assert not shifted[:dy].any()
    if dy < 0:
        assert not shifted[dy:].any()
    if dx > 0:
        assert not shifted[:, :dx].any()
    if dx < 0:
        assert not shifted[:, dx:].any()


def test_serialized_capture_recovers_known_truth(tmp_path):
    truth = np.random.default_rng(2).integers(0, 256, (24, 32), dtype=np.uint8)
    shifts = probe_shifts()
    frames = np.stack([integer_translate(truth, dy, dx) for dy, dx in shifts])
    path = write_ser(tmp_path/'motion.ser', frames)
    report = verify_capture(path, truth, shifts)
    assert report['verified_frames'] == 256
    assert report['known_shift_mean_max_error_adu'] == 0.
