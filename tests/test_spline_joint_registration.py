import numpy as np
import pytest
import torch
from scipy.ndimage import gaussian_filter, map_coordinates, shift

from planetrecon.pipeline.circular_align import CircularMultiscaleRegistration
from tools.spline_joint_registration import (
    SplineJointRegistration, spline_coefficients, spline_sample,
)


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_spline_matches_scipy_at_fractional_coordinates_and_boundaries(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    rng = np.random.default_rng(19)
    image = rng.normal(size=(29, 35))
    coords = rng.uniform(-50, 80, (2, 300))
    # Include exact integer locations, knots, edges, and reflected periods.
    coords[:, :7] = [[0, 28, -1, 29, 58, .5, 28.5], [0, 34, -1, 35, 70, .5, 34.5]]
    coeff = torch.tensor(spline_coefficients([image])[0], device=device)
    y, x = torch.tensor(coords, device=device)
    result = spline_sample(coeff, x, y)
    expected = map_coordinates(image, coords, order=3, mode='reflect')
    np.testing.assert_allclose(result.cpu(), expected, atol=1e-12, rtol=1e-12)


def test_spline_coordinate_and_blur_mixture_derivatives():
    rng = np.random.default_rng(40)
    bank = torch.tensor(spline_coefficients(rng.normal(size=(2, 12, 14))))
    x = torch.tensor([.25, 5.3, 12.8, -1.2], requires_grad=True, dtype=torch.float64)
    y = torch.tensor([1.2, 7.6, 10.4, -.3], requires_grad=True, dtype=torch.float64)
    fraction = torch.tensor(.4, requires_grad=True, dtype=torch.float64)
    assert torch.autograd.gradcheck(
        lambda a, b, f: spline_sample(bank[0]+f*(bank[1]-bank[0]), a, b),
        (x, y, fraction), eps=1e-6, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_fractional_translation_does_not_invent_local_motion(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    ref = 100+20*gaussian_filter(np.random.default_rng(94).normal(size=(80, 96)), 1.)
    engine = SplineJointRegistration(ref, CircularMultiscaleRegistration(ref),
                                     device=device, maxiter=30)
    for delta in ((.25, .5), (.6, -.4), (-.7, .35)):
        for sigma in (0., 1.):
            blurred = gaussian_filter(ref, sigma) if sigma else ref
            frame = shift(blurred, delta[::-1], order=3, mode='reflect')
            before = frame.copy()
            field, stats = engine.fit(frame, delta, .3, lambda: None)
            expected = torch.tensor(delta, device=device)[:, None, None].expand_as(field)
            torch.testing.assert_close(field, expected, rtol=0, atol=1e-6, check_dtype=False)
            assert stats['heldout_loss'] < 1e-10
            assert not stats['heldout_improves']
            assert not stats['fallback']
            np.testing.assert_array_equal(frame, before)
