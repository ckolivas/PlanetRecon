import numpy as np
import pytest

from tools.analytic_sampling_probe import detector_image
from tools.elliptical_blur_probe import covariance, integrated_scene, EllipticalReference
from tools.fixed_circular_blur_probe import fit_circular


@pytest.mark.parametrize('sigma', [0., .5, 1.8])
def test_circular_limit_matches_independent_exact_integrals(sigma):
    features = np.array([[31., 13.2, 12.7, .65, .8], [100., 20., 22., 4., 5.]])
    actual = integrated_scene(features, (40, 48), np.eye(2)*sigma**2,
                              shift_xy=(.6, -.4), order=8)
    expected = detector_image(features, (40, 48), (.6, -.4), sigma)
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=0)


def test_rotated_elliptical_blur_preserves_flux_and_covariance():
    features = np.array([[31., 30.2, 32.7, .65, .8]])
    psf = covariance(1.8, .5, .7)
    image = integrated_scene(features, (64, 64), psf, order=8)-3.
    assert image.sum() == pytest.approx(31*2*np.pi*.65*.8, abs=1e-8)
    y, x = np.indices(image.shape)
    delta = np.array([x-30.2, y-32.7])
    measured = np.einsum('ihw,jhw,hw->ij', delta, delta, image)/image.sum()
    np.testing.assert_allclose(measured, psf+np.diag([.65**2, .8**2])+np.eye(2)/12,
                               atol=1e-5, rtol=0)


def test_blur_estimation_recovers_independent_scene_and_ignores_heldout():
    features = np.array([[100., 28., 25., 6., 4.], [35., 21., 26., 2., 2.5],
                         [40., 35., 30., 2.5, 2.]])
    reference = detector_image(features, (56, 64))
    psf = covariance(1.7, .6, .65)
    frame = integrated_scene(features, reference.shape, psf, shift_xy=(.3, -.4), order=8)
    estimator = EllipticalReference(reference)
    mask = np.zeros(reference.shape, bool)
    mask[8:-8:2, 8:-8:2] = True
    before = frame.copy()
    _, stats = estimator.fit(frame, (.3, -.4), mask)
    np.testing.assert_allclose(stats['covariance'], psf, atol=.012, rtol=0)
    assert stats['training_mse'] < 1e-5
    changed = frame.copy()
    changed[~mask] += np.random.default_rng(3).normal(0, 100, (~mask).sum())
    _, alternate = estimator.fit(changed, (.3, -.4), mask)
    np.testing.assert_array_equal(alternate['covariance'], stats['covariance'])
    np.testing.assert_array_equal(frame, before)


def test_fixed_circular_profile_recovers_blur_with_same_training_samples():
    features = np.array([[100., 28., 25., 6., 4.], [35., 21., 26., 2., 2.5]])
    reference = detector_image(features, (56, 64))
    frame = detector_image(features, reference.shape, (.3, -.4), 1.3)
    mask = np.zeros(reference.shape, bool)
    mask[8:-8:2, 8:-8:2] = True
    estimator = EllipticalReference(reference)
    _, stats = fit_circular(estimator, frame, (.3, -.4), mask)
    assert stats['sigma'] == pytest.approx(1.3, abs=.003)
    assert stats['training_mse'] < 1e-5
    changed = frame.copy()
    changed[~mask] += 200.
    _, alternate = fit_circular(estimator, changed, (.3, -.4), mask)
    assert alternate == stats
