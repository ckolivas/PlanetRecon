import numpy as np
import torch

from tools.analytic_sampling_probe import components, detector_image
from tools.ap_oracle_observations import patch_averages
from tools.ap_oracle_identifiability import orthogonal_operator
from tools.regularized_ap_fit import RegularizedAPFit


def test_truth_patch_averages_match_spline_design_and_axis_order():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        reference = detector_image(components('resolved'), (128, 160), blur=1.)
        model = RegularizedAPFit(reference, device='cpu', policies=['baseline'])
        engine, fitter = model.engine, model.fitters['baseline']
        coefficients = np.random.default_rng(419).normal(size=(fitter.ny*fitter.nx, 2))
        field = np.stack([fitter.by@coefficients[:, i].reshape(fitter.ny, fitter.nx)@fitter.bx.T for i in range(2)])
        np.testing.assert_allclose(patch_averages(engine, field), fitter.design@coefficients,
                                   atol=1e-12, rtol=0)
        offset = np.array([1.7, -.4])
        np.testing.assert_allclose(patch_averages(engine, field+offset[:, None, None]),
                                   patch_averages(engine, field)+offset, atol=1e-12, rtol=0)
        zero = patch_averages(engine, np.zeros_like(field))
        result, stats = model.field('baseline', (zero, np.ones(len(zero))), offset)
        np.testing.assert_allclose(result.numpy(), np.broadcast_to(offset[:, None, None], field.shape), atol=1e-12, rtol=0)
        assert stats['field_guard_accepted'] and not stats['fallback']
        qy, qx, operator, u, s, vt, rank, tolerance = orthogonal_operator(fitter)
        z = np.stack([(qy.T@v@qx).ravel() for v in field], axis=1)
        np.testing.assert_allclose(operator@z, patch_averages(engine, field), atol=1e-12, rtol=0)
        np.testing.assert_allclose(np.sum(z*z), np.sum(field*field), atol=1e-10, rtol=1e-14)
        null = z-vt[:rank].T@(vt[:rank]@z)
        np.testing.assert_allclose(operator@null, 0., atol=1e-12, rtol=0)
    finally:
        torch.set_num_threads(previous)
