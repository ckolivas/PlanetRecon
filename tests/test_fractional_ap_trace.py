import numpy as np
import torch
from scipy.ndimage import map_coordinates

from tools.analytic_sampling_probe import components, detector_image
from tools.fractional_ap_trace import cubic_sample, trace, observations
from tools.local_warp_trace import trace as original_trace
from tools.regularized_ap_fit import RegularizedAPFit
from planetrecon.pipeline.local_align import LocalRegistration


def test_cubic_preserves_integer_samples_and_affine_interior():
    y, x = torch.meshgrid(torch.arange(24, dtype=torch.float64), torch.arange(28, dtype=torch.float64), indexing='ij')
    image = 3.+2*x-.7*y
    torch.testing.assert_close(cubic_sample(image, y, x), image, atol=0, rtol=0)
    torch.testing.assert_close(cubic_sample(image, y[3:-3,3:-3]+.2, x[3:-3,3:-3]-.4),
        3+2*(x[3:-3,3:-3]-.4)-.7*(y[3:-3,3:-3]+.2), atol=1e-12, rtol=0)


def test_experimental_baseline_preserves_original_trace_and_composed_observations():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        ref = detector_image(components('resolved'), (128,160), blur=1.)
        model = RegularizedAPFit(ref, device='cpu', policies=['baseline'])
        proxy = LocalRegistration.proxy(ref+np.random.default_rng(817).normal(0, 2, ref.shape))
        expected, records = original_trace(model.engine, proxy, (.2,-.3))
        actual, details = trace(model.engine, proxy, (.2,-.3))
        for a, b, ar, br in zip(actual, expected, details, records):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
            np.testing.assert_array_equal(ar['measurements'], br['measurements'])
        for a, b in zip(observations(model.engine, actual, details, (.2,-.3)), model.observations(proxy, (.2,-.3))):
            np.testing.assert_allclose(a, b, atol=1e-12, rtol=0)
    finally:
        torch.set_num_threads(previous)
