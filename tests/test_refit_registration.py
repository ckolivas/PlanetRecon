import numpy as np
import pytest
import torch
from scipy.ndimage import gaussian_filter

from planetrecon.pipeline.circular_align import CircularMultiscaleRegistration
from tools.refit_registration import RefitRegistration, guarded
from tools.validated_registration import ValidatedRegistration
from tools.validate_local_warp_centring import make_frame


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_ablation_preserves_baseline_and_refits_moving_scene(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    ref = 100+30*gaussian_filter(np.random.default_rng(94).normal(size=(96,112)),1.)
    frame, _, _ = make_frame(ref,.8,-.3,0.)
    baseline = ValidatedRegistration(ref,CircularMultiscaleRegistration(ref),device=device)
    engine = RefitRegistration(ref,CircularMultiscaleRegistration(ref),device=device)
    expected = baseline.displacement_raw(frame,(.1,-.1),lambda:None)
    variants = engine.variants(frame,(.1,-.1),lambda:None)
    # CUDA reductions may differ by a few float64 ULPs between instances.
    torch.testing.assert_close(variants['validated'],expected,rtol=0,atol=1e-12)
    assert engine.stats[-1]['accepted_halves'] == 2
    torch.testing.assert_close(variants['full_any'],variants['coherent'],rtol=0,atol=0)
    torch.testing.assert_close(variants['full_scaled'],variants['coherent'],rtol=0,atol=0)


def test_refit_does_not_bypass_stationary_rejection():
    ref = 100+30*gaussian_filter(np.random.default_rng(72).normal(size=(80,96)),1.)
    engine = RefitRegistration(ref,CircularMultiscaleRegistration(ref),device='cpu')
    variants = engine.variants(gaussian_filter(ref,.75),(0.,0.),lambda:None)
    assert engine.stats[-1]['accepted_halves'] == 0
    for key in ('validated','full_any','full_scaled'):
        np.testing.assert_array_equal(variants[key],0.)


def test_guard_preserves_global_translation_when_residual_is_invalid():
    origin = torch.tensor([.4,-.2],dtype=torch.float64)[:,None,None]
    field = origin.expand(2,40,48).clone()
    field[0,20,20] += 7.
    torch.testing.assert_close(guarded(field,origin),origin.expand_as(field))
