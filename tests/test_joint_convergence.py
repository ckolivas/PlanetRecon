import numpy as np

from planetrecon.config import make_config
from planetrecon.mfbd import PupilForward, d_tail, fit_converged, phase_stationarity, phase_start, fit_frame_alpha


def fixture():
    fwd = PupilForward.from_config(make_config(1001, 4., n_diam=8, pupil_pad_factor=8., eval_size=16), n_modes=3)
    y, x = np.indices((16, 16))
    obj = 1.+np.exp(-((x-8)**2+(y-8)**2)/10.)
    alpha = np.array([[.2, -.3, .4], [-.3, .1, -.2], [.1, .2, .6]])
    images = np.fft.ifft2(fwd.otfs(alpha)*np.fft.fft2(obj)).real
    return fwd, obj, alpha, images


def test_matching_joint_solution_stops_early_with_current_gradients():
    fwd, obj, alpha, images = fixture()
    fit = d_tail(fwd, images, np.ones(3), np.zeros((16, 16)), np.ones((16, 16)),
                 tv_mu=0., obj0=obj, alpha0=alpha, train_idx=np.arange(3), holdout_idx=np.array([], dtype=int),
                 m_grid=(3,), outer_iters=(5,), alpha_iters=20, freeze_tip_tilt=True)
    assert fit_converged(fit)
    assert fit['stages'][0]['n_outer'] < 5
    assert fit['stages'][0]['convergence']['phase']['max_relative_gradient_inf'] < 1e-8
    np.testing.assert_allclose(fit['object'], obj, atol=1e-7)


def test_phase_stationarity_must_be_recomputed_when_object_changes():
    fwd, obj, alpha, images = fixture()
    exact = phase_stationarity(fwd, alpha, obj, images, np.ones(3), np.arange(3), freeze_tip_tilt=True)
    changed = phase_stationarity(fwd, alpha, np.roll(obj, 2, axis=1), images, np.ones(3), np.arange(3), freeze_tip_tilt=True)
    assert exact['stationary']
    assert not changed['stationary']
    assert changed['max_relative_gradient_inf'] > 1e-8


def test_historical_failures_do_not_override_converged_final_pair():
    stage = {'object_info': {'converged': True}, 'convergence': {'converged': True},
             'phase_fits': [{'frames': [{'success': False}]}]}
    assert fit_converged({'stages': [stage]})
    stage['convergence']['converged'] = False
    stage['phase_fits'][0]['frames'][0]['success'] = True
    assert not fit_converged({'stages': [stage]})
    del stage['convergence']
    assert not fit_converged({'stages': [stage]})


def test_nonzero_phase_start_escapes_symmetric_stationary_point():
    fwd, obj, _, _ = fixture()
    obj_f = np.fft.fft2(obj-1.)
    image = np.fft.ifft2(fwd.otf(np.array([0., 0., 2.]))*obj_f).real
    _, zero_loss = fit_frame_alpha(np.zeros(3), fwd, obj_f, image, 1., n_iter=96, freeze_tip_tilt=True)
    start = phase_start(fwd, np.zeros((1, 2)), 3, 1001, perturb=True)[0]
    _, escaped_loss = fit_frame_alpha(start, fwd, obj_f, image, 1., n_iter=96, freeze_tip_tilt=True)
    assert zero_loss > .001
    assert escaped_loss < 1e-12


def test_phase_starts_are_reproducible_and_preserve_calibrated_tilt():
    from planetrecon import constants as C
    fwd, _, _, _ = fixture()
    tt = np.array([[3., -4.], [1., 2.]])
    first = phase_start(fwd, tt, 3, 1001, perturb=True)
    again = phase_start(fwd, tt[:1], 3, 1001, perturb=True)
    np.testing.assert_array_equal(first[:1], again)
    np.testing.assert_array_equal(first[:, :2], tt)
    np.testing.assert_allclose(np.linalg.norm(first[:, 2:], axis=1)/np.sqrt(fwd.mask.sum()), C.Q2_PHASE_START_RMS_RAD)
