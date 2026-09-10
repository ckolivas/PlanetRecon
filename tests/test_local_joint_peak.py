"""Known tilted peaks and continuous scenes expose coupled local-offset bias."""
import numpy as np
import pytest

from planetrecon.pipeline.local_align import LocalRegistration


@pytest.mark.parametrize('cross', [-.012, .012])
@pytest.mark.parametrize('offset', [(.35, -.3), (-.25, .4)])
def test_tilted_local_peak_recovers_joint_centre(cross, offset):
    y, x = np.indices((160, 192), dtype=float)
    reference = 300+30*np.cos(x/3)+35*np.sin(y/4)
    matcher = LocalRegistration(reference, use_cuda=True)
    matcher.texture_valid = np.ones(len(matcher.templates), dtype=bool)
    matcher.strength = np.ones(len(matcher.templates))
    cy, cx = np.indices((7, 7), dtype=float)-3
    dx, dy = cx-offset[0], cy-offset[1]
    # Independently specified concave quadratic with a known continuous maximum.
    costs = .95-.5*(.018*dx**2+2*cross*dx*dy+.02*dy**2)
    matcher._cuda_costs = lambda image: np.repeat(costs[None], len(matcher.templates), axis=0)
    actual = matcher.displacement(reference, (0., 0.))
    roi = np.s_[matcher.ys[0]:matcher.ys[-1]+1, matcher.xs[0]:matcher.xs[-1]+1]
    for component, expected in zip(actual, offset):
        np.testing.assert_allclose(component[roi], expected, atol=1e-12)


@pytest.mark.parametrize('direction', [-1, 1])
def test_continuous_diagonal_texture_registration(direction):
    y, x = np.indices((224, 256), dtype=float)
    def scene(x, y):
        return 300+35*np.cos((x+direction*y)/5)+15*np.sin((x-direction*y)/7)
    expected = np.array([.35, -.3])
    reference = scene(x, y)
    frame = scene(x-expected[0], y-expected[1])
    actual = LocalRegistration(reference).displacement(frame, (0., 0.))
    error = np.hypot(actual[0][72:-72, 72:-72]-expected[0],
                     actual[1][72:-72, 72:-72]-expected[1])
    # The quadratic is an approximation to this sinusoidal correlation surface:
    # require removal of at least 90% of the uncompensated displacement.
    assert np.sqrt(np.mean(error**2)) < .1*np.linalg.norm(expected)


def test_weak_diagonal_ridge_keeps_global_shift():
    y, x = np.indices((160, 192), dtype=float)
    reference = 300+30*np.cos(x/3)+35*np.sin(y/4)
    matcher = LocalRegistration(reference, use_cuda=True)
    matcher.texture_valid = np.ones(len(matcher.templates), dtype=bool)
    matcher.strength = np.ones(len(matcher.templates))
    cy, cx = np.indices((7, 7), dtype=float)-3
    costs = .95-.01*(cx+cy-.2)**2-.00001*(cx-cy)**2
    matcher._cuda_costs = lambda image: np.repeat(costs[None], len(matcher.templates), axis=0)
    for component, expected in zip(matcher.displacement(reference, (.25, -.125)), (.25, -.125)):
        np.testing.assert_array_equal(component, np.full(reference.shape, expected))
