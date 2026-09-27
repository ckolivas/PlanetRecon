import numpy as np
import pytest
import torch

from tools.reference_imprint_probe import probes, ReferenceStates, engine_for, metrics, response_summary, variants
from tools.analytic_sampling_probe import components, detector_image
from planetrecon.pipeline.local_align import LocalRegistration
from tools.analyse_reference_imprint import geometric_component


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def test_patterns_are_reproducible_centred_unit_rms_and_distinct():
    reference = detector_image(components('resolved'), (128, 160))
    before = reference.copy()
    patterns, mask = probes(reference)
    repeated, other_mask = probes(reference)
    np.testing.assert_array_equal(reference, before)
    np.testing.assert_array_equal(mask, other_mask)
    for name, pattern in patterns.items():
        np.testing.assert_array_equal(pattern, repeated[name])
        assert abs(pattern[mask].mean()) < 1e-14
        assert abs(np.mean(pattern[mask]**2)-1.) < 1e-14
    assert abs(np.corrcoef(patterns['fine'][mask], patterns['broad'][mask])[0, 1]) < .2


def test_paired_response_recovers_known_linear_gain_and_even_term():
    reference = detector_image(components('resolved'), (128, 160))
    patterns, mask = probes(reference)
    images = {}
    for key, (name, dose) in variants().items():
        images[key] = reference.copy() if name is None else reference+.2*dose*patterns[name]+.1*dose*dose
    result = response_summary(images, patterns, mask)
    for name, values in result.items():
        dose = float(name.split('_')[1])
        assert abs(values['response_per_reference_adu']['pattern_gain']-.2) < 1e-12
        assert abs(values['even_response_rms_adu']-.1*dose*dose) < 1e-12
    assert metrics(np.zeros_like(reference), patterns['fine'], mask)['correlation'] is None


def test_switching_reference_states_restores_identical_field_and_fixed_geometry():
    reference = detector_image(components('resolved'), (128, 160))
    patterns, _ = probes(reference)
    engine = engine_for(reference, 'coherent', 'cpu')
    states = ReferenceStates(engine, reference, patterns)
    geometry = [(l['x'].clone(), l['y'].clone()) for l in engine.layers]
    design = engine.fitter.design.copy()
    frame = reference+np.random.default_rng(8).normal(0, 2, reference.shape)
    proxy = LocalRegistration.proxy(frame)
    original = engine.displacement(proxy, (0., 0.), lambda: None)
    states.select('broad_1_plus')
    engine.displacement(proxy, (0., 0.), lambda: None)
    states.select('base')
    torch.testing.assert_close(engine.displacement(proxy, (0., 0.), lambda: None), original, atol=0, rtol=0)
    np.testing.assert_array_equal(engine.fitter.design.toarray(), design.toarray())
    for layer, (x, y) in zip(engine.layers, geometry):
        torch.testing.assert_close(layer['x'], x)
        torch.testing.assert_close(layer['y'], y)


def test_translation_component_is_distinguished_from_pattern_copying():
    reference = detector_image(components('resolved'), (128, 160))
    _, mask = probes(reference)
    gy, gx = np.gradient(reference)
    result = geometric_component(.2*gx-.3*gy, reference, mask)
    np.testing.assert_allclose(result['translation_tangent_xy_px_per_reference_adu'], [.2, -.3], atol=1e-14)
    assert result['residual_rms'] < 1e-14
    assert abs(result['explained_energy_fraction']-1.) < 1e-14
