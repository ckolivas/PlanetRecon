import numpy as np
import pytest
import torch
from scipy.ndimage import gaussian_filter

from planetrecon.pipeline.circular_align import CircularMultiscaleRegistration
from planetrecon.pipeline.local_align import LocalRegistration
from tools.blur_matched_registration import BlurMatchedRegistration, choose_blur
from tools.coherent_registration import CoherentRegistration


@pytest.fixture(autouse=True)
def limit_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def scene():
    return 100+30*gaussian_filter(np.random.default_rng(42).normal(size=(96,112)), 1.)


def test_blur_choice_tolerates_gain_and_offset():
    r = scene()
    bank = torch.tensor(np.stack([gaussian_filter(r, s) if s else r for s in (0, 1, 2)]))
    index, losses = choose_blur(bank, bank[1]*1.2+5, torch.ones(r.shape, dtype=torch.bool))
    assert index == 1
    assert losses[1] < 1e-10


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_fixed_geometry_zero_blur_parity_and_exact_blur(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    ref = scene()
    matcher = CircularMultiscaleRegistration(ref)
    baseline = CoherentRegistration(matcher, device=device, spacing=16., stiffness=.01, patch_average=True)
    engine = BlurMatchedRegistration(ref, matcher, device=device)
    design = engine.engine.fitter.design.copy()
    points = engine.engine.points.copy()
    frame = gaussian_filter(ref, 1.)
    expected = baseline.displacement(LocalRegistration.proxy(frame), (0.,0.), lambda:None)
    fields = engine.variants(frame, (0.,0.), lambda:None, known_sigma=1.)
    torch.testing.assert_close(fields['coherent'], expected, atol=1e-12, rtol=0)
    assert engine.stats[-1]['global_sigma'] == 1.
    torch.testing.assert_close(fields['known_blur'], torch.zeros_like(expected), atol=1e-12, rtol=0)
    np.testing.assert_array_equal(engine.engine.points, points)
    np.testing.assert_array_equal(engine.engine.fitter.design.toarray(), design.toarray())
    # Changing templates must also update reverse matching, and reset cleanly.
    engine.set_template(0.)
    torch.testing.assert_close(engine.engine.reference, baseline.reference, atol=0, rtol=0)


def test_blank_reference_and_invalid_blur():
    ref = np.zeros((40,48))
    engine = BlurMatchedRegistration(ref, CircularMultiscaleRegistration(ref), device='cpu')
    for value in (-1, np.nan, 3):
        with pytest.raises(ValueError):engine.set_template(value)
    fields = engine.variants(ref, (.3,-.2), lambda:None)
    for field in fields.values():
        np.testing.assert_array_equal(field[0].numpy(), .3)
        np.testing.assert_array_equal(field[1].numpy(), -.2)
