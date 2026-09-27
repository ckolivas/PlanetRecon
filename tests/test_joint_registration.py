import numpy as np
import pytest
import torch
from scipy.ndimage import gaussian_filter

from planetrecon.pipeline.circular_align import CircularMultiscaleRegistration
from tools.joint_registration import JointRegistration, photometric_loss, global_blur_fit


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def test_score_gain_and_offset_do_not_change_observations():
    x = torch.arange(100,dtype=torch.float64)
    observed = 1.2*x+7
    before = observed.clone()
    loss,gain,offset = photometric_loss(x,observed)
    assert float(loss) < 1e-20
    assert float(gain) == pytest.approx(1.2)
    assert float(offset) == pytest.approx(7)
    torch.testing.assert_close(observed,before,rtol=0,atol=0)


@pytest.mark.parametrize('gain',[.5,1.2,2.])
def test_stationary_model_has_same_continuous_blur_freedom(gain):
    rng = np.random.default_rng(94)
    bank = torch.tensor(rng.normal(size=(3,300)),dtype=torch.float64)
    observed = gain*(.3*bank[1]+.7*bank[2])+7.
    variance,g,offset,loss = global_blur_fit(bank,observed,np.array([0.,1.,4.]))
    assert variance == pytest.approx(3.1,abs=1e-9)
    assert g == pytest.approx(gain,abs=1e-9)
    assert offset == pytest.approx(7.,abs=1e-9)
    assert loss<1e-10


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_stationary_blur_and_integer_translation(device):
    if device == 'cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    ref = 100+20*gaussian_filter(np.random.default_rng(94).normal(size=(96,112)),1.)
    engine = JointRegistration(ref,CircularMultiscaleRegistration(ref),device=device,maxiter=30)
    for shift in ((0.,0.),(2.,-1.)):
        frame = np.roll(gaussian_filter(ref,1.),(int(shift[1]),int(shift[0])),axis=(0,1))
        field,stats = engine.fit(frame,shift,.03,lambda:None)
        assert not stats['fallback']
        assert stats['sigma'] == pytest.approx(1.,abs=1e-6)
        expected = torch.tensor(shift,device=device)[:,None,None].expand_as(field)
        torch.testing.assert_close(field,expected,rtol=0,atol=1e-6,check_dtype=False)
        assert stats['heldout_loss'] < 1e-10


def test_blank_reference_keeps_translation():
    ref = np.zeros((40,48))
    engine = JointRegistration(ref,CircularMultiscaleRegistration(ref),device='cpu')
    field,stats = engine.fit(ref,(.3,-.2),.03,lambda:None)
    assert stats['fallback']
    np.testing.assert_array_equal(field[0].numpy(),.3)
    np.testing.assert_array_equal(field[1].numpy(),-.2)


def test_unused_detector_samples_cannot_change_fitted_motion():
    rng = np.random.default_rng(48)
    ref = 100+20*gaussian_filter(rng.normal(size=(80,96)),1.)
    engine = JointRegistration(ref,CircularMultiscaleRegistration(ref),device='cpu',maxiter=8)
    frame = gaussian_filter(ref,.8)+rng.normal(0,2,ref.shape)
    changed = frame.copy()
    unused = np.ones(ref.shape,dtype=bool)
    unused[::2,::2] = False
    changed[unused] += rng.normal(0,20,unused.sum())
    a,sa = engine.fit(frame,(.2,-.1),.3,lambda:None)
    b,sb = engine.fit(changed,(.2,-.1),.3,lambda:None)
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert sa['sigma'] == sb['sigma']
    assert sa['heldout_loss'] != sb['heldout_loss']
