import numpy as np

from tools.ap_noise_coupling import pair_stacks


def test_pair_swap_cancels_identical_clean_geometry_and_isolates_noise_coupling():
    rng = np.random.default_rng(18)
    clean, noise_a, noise_b = rng.normal(size=(3, 14, 17))
    def wa(a):
        return .7*a+.3*np.roll(a, 1, axis=0)
    def wb(a):
        return .2*a+.8*np.roll(a, -1, axis=1)
    same, cross = pair_stacks(wa(clean+noise_a), wa(clean+noise_b), wb(clean+noise_a), wb(clean+noise_b))
    expected = .5*(wa(noise_a-noise_b)-wb(noise_a-noise_b))
    np.testing.assert_allclose(same-cross, expected, atol=1e-15, rtol=0)
    twins = pair_stacks(wa(clean), wa(clean), wb(clean), wb(clean))
    np.testing.assert_array_equal(*twins)
    shared_field = pair_stacks(wa(clean+noise_a), wa(clean+noise_b), wa(clean+noise_a), wa(clean+noise_b))
    np.testing.assert_array_equal(*shared_field)
