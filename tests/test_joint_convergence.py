import numpy as np

from planetrecon.config import make_config
from planetrecon.mfbd import PupilForward, d_tail, fit_converged, phase_stationarity


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
