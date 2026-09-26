import numpy as np
from scipy.ndimage import map_coordinates
from scipy.ndimage import gaussian_filter
import torch

from planetrecon.backends.torch_circular import TorchCircularRegistration
from planetrecon.pipeline.circular_align import CircularMultiscaleRegistration
from tools.validate_local_warp_centring import make_frame, shear_coordinates, weighted_field_errors, run_case


def test_composed_shear_inverse_and_pull_sign():
    shape = (100, 120)
    y, x = np.indices(shape)
    a, b = 1.3, -.8
    xx, yy = shear_coordinates(shape, a, b)
    cy, cx = (np.asarray(shape)-1)/2
    original_y = yy-b*np.sin(2*np.pi*(xx-cx)/240+.4)
    original_x = xx-a*np.sin(2*np.pi*(original_y-cy)/160+.5)
    np.testing.assert_allclose(original_x, x, atol=2e-14)
    np.testing.assert_allclose(original_y, y, atol=2e-14)
    scene = 100+30*np.sin(x/7)+20*np.cos(y/9)
    frame, field, _ = make_frame(scene, a, b, 0.)
    restored = map_coordinates(frame, [y+field[1], x+field[0]], order=1)
    wrong_sign = map_coordinates(frame, [y-field[1], x-field[0]], order=1)
    roi = np.s_[8:-8, 8:-8]
    assert np.std((restored-scene)[roi]) < .1
    assert np.std((restored-scene)[roi]) < np.std((frame-scene)[roi])/10
    assert np.std((wrong_sign-scene)[roi]) > np.std((frame-scene)[roi])


def test_weighted_centring_loses_true_mean_even_with_perfect_registration():
    mask = np.ones((4, 5), dtype=bool)
    truth = np.zeros((3, 2, 4, 5))
    truth[:, 0] = np.array([-1., 0., 1.])[:, None, None]
    weights = np.array([.1, .2, .7])
    mean = np.einsum('n,nahw->ahw', weights, truth)
    result = weighted_field_errors(truth-mean, truth, weights, mask)
    np.testing.assert_allclose(result['total_vector_rmse_px'], .6)
    np.testing.assert_allclose(result['mean_error_vector_rms_px'], .6)
    np.testing.assert_allclose(result['temporal_error_vector_rms_px'], 0., atol=1e-15)


def test_zero_motion_generation_preserves_scene_and_brightness():
    scene = np.random.default_rng(34).uniform(5, 200, (80, 96))
    frame, field, blurred = make_frame(scene, 0., 0., 0.)
    np.testing.assert_allclose(frame, scene, atol=2e-13)
    np.testing.assert_array_equal(field, 0.)
    np.testing.assert_array_equal(blurred, scene)


def test_known_blur_template_recovers_zero_motion_without_changing_signal(tmp_path):
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        scene = 100+30*gaussian_filter(np.random.default_rng(91).normal(size=(80, 96)), 1.)
        engine = TorchCircularRegistration(CircularMultiscaleRegistration(scene), device='cpu')
        n = 8
        parameters = (np.zeros(n), np.zeros(n), np.full(n, .6), 0., np.ones(n))
        report = run_case(engine, scene, parameters, n, 10, np.ones(scene.shape, dtype=bool),
                          tmp_path, 'static', match_known_blur=True)
        assert report['variants']['local']['field']['total_vector_rmse_px'] < 1e-12
        with np.load(tmp_path/'static.npz') as data:
            np.testing.assert_allclose(data['local'], gaussian_filter(scene, .6), atol=1e-12)
    finally:
        torch.set_num_threads(previous)
