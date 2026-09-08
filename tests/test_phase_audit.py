import numpy as np

from planetrecon.config import make_config
from planetrecon.mfbd import PupilForward
from planetrecon.optics import instantaneous_psf
from planetrecon.phase_audit import decompose


def test_static_representable_phase_has_no_basis_or_exposure_error():
    fwd = PupilForward.from_config(make_config(1001, 4., n_diam=8, pupil_pad_factor=8., eval_size=16), n_modes=8)
    phase = fwd.phase(np.array([.2, -.3, .8]))
    psf = instantaneous_psf(fwd.amplitude, phase)
    report = decompose(fwd, phase, psf, psf, modes=(3, 8))
    assert report['exposure_only_relative_otf_error'] < 1e-12
    assert report['exposure_quadrature_relative_otf_error'] < 1e-12
    for row in report['basis']:
        assert row['basis_only_relative_otf_error'] < 1e-12
        assert row['phase_residual_rms_rad'] < 1e-12


def test_basis_and_exposure_controls_are_separated():
    fwd = PupilForward.from_config(make_config(1001, 4., n_diam=8, pupil_pad_factor=8., eval_size=16), n_modes=8)
    coeff = np.zeros(8)
    coeff[7] = 5.
    phase = fwd.phase(coeff)
    exact = instantaneous_psf(fwd.amplitude, phase)
    different = instantaneous_psf(fwd.amplitude, fwd.phase(np.array([5., 0.])))
    static = decompose(fwd, phase, exact, exact, modes=(3, 8))
    varying = decompose(fwd, phase, .5*(exact+different), .5*(exact+different), modes=(3, 8))
    assert static['exposure_only_relative_otf_error'] == 0.
    assert varying['exposure_only_relative_otf_error'] > .01
    assert static['basis'][0]['basis_only_relative_otf_error'] > .01
    for a, b in zip(static['basis'], varying['basis']):
        assert a['basis_only_relative_otf_error'] == b['basis_only_relative_otf_error']
    assert varying['basis'][1]['basis_only_relative_otf_error'] < 1e-12
