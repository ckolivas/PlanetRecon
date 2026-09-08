import numpy as np
import pytest

from planetrecon.config import make_config
from planetrecon.mfbd import PupilForward, tip_tilt_from_shifts
from planetrecon.optics import centroid_px


@pytest.mark.parametrize('phase_rms', [0., .5, 1.])
def test_centroid_calibration_across_shifts_and_higher_modes(phase_rms):
    fwd = PupilForward.from_config(make_config(1001, 4., n_diam=16, pupil_pad_factor=8., eval_size=32))
    targets = np.array([[-1.5, 1.2], [0., 0.], [1.4, -.9], [-.2, .1]])
    rng = np.random.default_rng(55)
    bases = rng.normal(size=(4, fwd.n_modes))
    bases[:, :2] = 0.
    for base in bases:
        base *= phase_rms / np.std(fwd.phase(base)[fwd.mask])
    original = bases.copy()
    coefficients, info = tip_tilt_from_shifts(fwd, targets, phase_base=bases, return_info=True)
    np.testing.assert_array_equal(bases, original)
    bases[:, :2] = coefficients
    actual = np.array([centroid_px(fwd.psf_det(base)) for base in bases])
    np.testing.assert_allclose(actual, targets, atol=1e-6)
    assert all(row['success'] and row['centroid_error_px'] <= 1e-6 for row in info['frames'])


@pytest.mark.parametrize('targets', [[[np.nan, 0.]], [[100., 0.]], [1., 2.]])
def test_invalid_centroid_requests_are_not_silently_mapped(targets):
    fwd = PupilForward.from_config(make_config(1001, 4., n_diam=8, pupil_pad_factor=8., eval_size=16))
    with pytest.raises(ValueError):
        tip_tilt_from_shifts(fwd, np.array(targets))


def test_degenerate_pupil_cannot_supply_calibrated_tilt():
    fwd = PupilForward.from_config(make_config(1001, 4., n_diam=8, pupil_pad_factor=8., eval_size=16))
    fwd.modes[:, :2] = 0.
    with pytest.raises(ValueError, match='ill-conditioned'):
        tip_tilt_from_shifts(fwd, np.zeros((1, 2)))
