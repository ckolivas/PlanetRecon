import numpy as np
import pytest
import torch
from scipy.ndimage import gaussian_filter

from planetrecon.pipeline.circular_align import CircularMultiscaleRegistration
from tools.validated_registration import split_proxies,blur_model_loss,ValidatedRegistration
from tools.validate_local_warp_centring import make_frame


@pytest.fixture(autouse=True)
def threads():
    previous=torch.get_num_threads();torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def test_training_proxy_cannot_see_heldout_detector_samples():
    frame=np.random.default_rng(14).normal(size=(80,96))
    y,x=np.indices(frame.shape)
    changed=frame.copy();changed[(x+y)%2==1]+=100*np.sin(x/7.)[(x+y)%2==1]
    a,b=split_proxies(frame);aa,bb=split_proxies(changed)
    np.testing.assert_array_equal(a,aa)
    assert not np.allclose(b,bb)
    for proxy in split_proxies(np.full(frame.shape,17.)):
        np.testing.assert_allclose(proxy,0.,atol=1e-12)


def test_blur_score_fits_same_nuisance_family_without_modifying_observations():
    rng=np.random.default_rng(32)
    bank=torch.as_tensor(rng.normal(size=(4,40,48)))
    observed=1.2*(.7*bank[1]+.3*bank[2])+4
    original=observed.clone()
    loss,_=blur_model_loss(bank,observed,torch.ones_like(observed,dtype=torch.bool))
    assert loss<1e-12
    torch.testing.assert_close(observed,original,rtol=0,atol=0)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_known_blur_without_motion_is_rejected_as_displacement(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    reference=100+30*gaussian_filter(np.random.default_rng(72).normal(size=(80,96)),1.)
    engine=ValidatedRegistration(reference,CircularMultiscaleRegistration(reference),device=device)
    frame=gaussian_filter(reference,.75)
    field=engine.displacement_raw(frame,(0.,0.),lambda:None)
    np.testing.assert_allclose(field.cpu().numpy(),0.,atol=1e-12)
    assert engine.stats[-1]['accepted_halves']==0


def test_inverse_coordinates_preserve_nonzero_global_translation():
    reference=100+gaussian_filter(np.random.default_rng(12).normal(size=(80,96)),1.)
    engine=ValidatedRegistration(reference,CircularMultiscaleRegistration(reference),device='cpu')
    field=torch.stack((torch.full_like(engine.xx,.4),torch.full_like(engine.yy,-.2)))
    tx,ty,error=engine.inverse(field,engine.mask.bool())
    torch.testing.assert_close(tx,engine.xx-.4)
    torch.testing.assert_close(ty,engine.yy+.2)
    assert error<1e-12


def test_moving_validation_cpu_cuda_parity():
    if not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    reference=100+30*gaussian_filter(np.random.default_rng(94).normal(size=(96,112)),1.)
    frame,_,_=make_frame(reference,.8,-.3,0.)
    fields=[]
    for device in ('cpu','cuda'):
        engine=ValidatedRegistration(reference,CircularMultiscaleRegistration(reference),device=device)
        fields.append(engine.displacement_raw(frame,(.1,-.1),lambda:None).cpu().numpy())
        assert engine.stats[-1]['accepted_halves']==2
        assert engine.stats[-1]['field_guard_accepted']
    np.testing.assert_allclose(fields[0],fields[1],atol=1e-8)
    assert np.std(fields[0][0]-.1)>.01
