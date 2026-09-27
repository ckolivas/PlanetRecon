import numpy as np
import pytest
import torch

from planetrecon.pipeline.local_align import LocalRegistration
from tools.analytic_sampling_probe import components, detector_image
from tools.regularized_ap_fit import RegularizedAPFit, POLICIES
from tools.ap_stability_screen import choose
from tools.ap_stability_validation import integrated
from tools.analytic_motion_probe import integrated_motion


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_shared_observations_preserve_baseline_and_do_not_mutate_confidence(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    reference = detector_image(components('resolved'), (128, 160), blur=1.)
    model = RegularizedAPFit(reference, device=device)
    frame = reference+np.random.default_rng(21).normal(0, 2, reference.shape)
    proxy = LocalRegistration.proxy(frame)
    observations = model.observations(proxy, (.2, -.3))
    before = tuple(a.copy() for a in observations)
    expected = model.engine.displacement(proxy, (.2, -.3), lambda: None)
    actual, _ = model.field('baseline', observations, (.2, -.3))
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=0)
    for name in POLICIES:
        result, stats = model.field(name, observations, (.2, -.3))
        assert bool(torch.isfinite(result).all())
    for a, b in zip(observations, before):
        np.testing.assert_array_equal(a, b)
    again, _ = model.field('baseline', observations, (.2, -.3))
    torch.testing.assert_close(again, actual, atol=0, rtol=0)


def test_stability_alone_cannot_select_inaccurate_or_fallback_solution():
    controls = {'offset': {name: dict(clean_image={'rmse_adu': .1}, field_rmse_px=.2,
        guard_rejections=0, insufficient_points=0) for name in POLICIES}}
    stability = {name: {kind: {'median_px': 1. if name=='baseline' else .1} for kind in ('fine', 'broad')} for name in POLICIES}
    for name in POLICIES:
        if name != 'baseline':
            controls['offset'][name]['field_rmse_px'] = .3
    assert choose(controls, stability)['selected'] is None
    controls['offset']['bend_1']['field_rmse_px'] = .2
    controls['offset']['bend_1']['guard_rejections'] = 1
    assert choose(controls, stability)['selected'] is None
    controls['offset']['bend_1']['guard_rejections'] = 0
    assert choose(controls, stability)['selected'] == 'bend_1'


def test_short_motion_generator_retains_exact_stationary_integrals_and_long_motion():
    features = np.array([[100., 20., 21., 3., 5.], [31., 13.2, 12.7, .65, .8]])
    shape, translation = (40, 48), (.3, -.4)
    stationary = integrated(features, shape, 0., 0., translation, (80., 120.), order=8)
    np.testing.assert_allclose(stationary, detector_image(features, shape, translation, 1.), atol=1e-10, rtol=0)
    long = integrated(features, shape, 1.2, -.6, translation, (160., 240.), order=8)
    np.testing.assert_allclose(long, integrated_motion(features, shape, 1.2, -.6, 1., order=8,
                                                     shift_xy=translation), atol=1e-12, rtol=0)
