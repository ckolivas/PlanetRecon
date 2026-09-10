"""Re-anchoring the local mean must not manufacture detector-edge features."""
import numpy as np
import pytest

from planetrecon.pipeline.local_align import LocalRegistration, build_template, pull
from test_local_observed_support import continuous_scene


def template_with_known_shifts(reference, frame, frame_shift, origin):
    # Isolate support handling from the precision of the correlation estimator.
    shifts = iter([frame_shift] * 4 + [origin])
    return build_template(reference, range(4), lambda _: frame,
                          lambda *_: next(shifts), 32.)


@pytest.mark.parametrize('origin', [(.4, -.7), (-.4, .7), (4., 0.),
                                   (-4., 0.), (0., 4.), (0., -4.)])
def test_reanchoring_preserves_constant_brightness_at_detector_edges(origin):
    reference = np.full((97, 112), 300.)
    actual = template_with_known_shifts(reference, reference, (0., 0.), origin)
    np.testing.assert_allclose(actual, reference, atol=1e-12, rtol=0)


@pytest.mark.parametrize('frame_shift,origin', [
    ((20., 0.), (-.6, .25)), ((-20., 0.), (.6, -.25)),
    ((0., 20.), (.25, -.6)), ((0., -20.), (-.25, .6)),
])
def test_unobserved_template_uses_best_frame_at_original_coordinates(frame_shift, origin):
    y, x = np.indices((97, 112), dtype=float)
    reference = continuous_scene(y, x)
    frame = np.full(reference.shape, 100.)
    actual = template_with_known_shifts(reference, frame, frame_shift, origin)
    observed = pull(np.ones_like(frame), frame_shift) > 0
    support = pull(observed, origin)
    assert (support == 0).any() and ((support > 0) & (support < 1)).any()
    np.testing.assert_allclose(actual[support > 0], 100., atol=1e-12, rtol=0)
    np.testing.assert_array_equal(actual[support == 0], reference[support == 0])


@pytest.mark.parametrize('origin', [(4., 0.), (-4., 0.), (0., 4.), (0., -4.)])
def test_template_padding_cannot_create_false_local_motion(origin):
    y, x = np.indices((192, 224), dtype=float)
    frame = continuous_scene(y, x)
    reference = continuous_scene(y+origin[1], x+origin[0])
    actual = template_with_known_shifts(reference, frame, (0., 0.), origin)
    np.testing.assert_array_equal(actual, reference)
    flow = LocalRegistration(actual).displacement(reference, (0., 0.))
    np.testing.assert_array_equal(flow, np.zeros((2, *reference.shape)))
    # Prior re-anchoring darkened the edge despite identical observed texture.
    prior = pull(frame, origin)
    old_flow = LocalRegistration(prior).displacement(reference, (0., 0.))
    assert np.hypot(*old_flow).max() > 1e-4
    assert np.linalg.norm(pull(reference, old_flow)-reference) > .01


def test_fully_observed_template_interior_is_unchanged():
    y, x = np.indices((97, 112), dtype=float)
    frame = continuous_scene(y, x)
    actual = template_with_known_shifts(frame, frame, (0., 0.), (.4, -.7))
    prior = pull(frame, (.4, -.7))
    np.testing.assert_array_equal(actual[1:-1, 1:-1], prior[1:-1, 1:-1])
