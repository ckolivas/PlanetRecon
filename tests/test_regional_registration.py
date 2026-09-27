import numpy as np
import pytest
import torch
from scipy.ndimage import gaussian_filter

from planetrecon.pipeline.circular_align import CircularMultiscaleRegistration
from tools.regional_registration import RegionalRegistration, partitions
from tools.refit_registration import RefitRegistration
from tools.validate_local_warp_centring import make_frame


@pytest.fixture(autouse=True)
def threads():
    n = torch.get_num_threads();torch.set_num_threads(2)
    yield
    torch.set_num_threads(n)


def test_partition_covers_image_and_blends_only_confidence():
    ref = np.ones((80,96))
    hard,blend,metadata = partitions(ref,ref>0)
    np.testing.assert_array_equal(hard.sum(0),1)
    np.testing.assert_allclose(blend.sum(0),1.,atol=1e-14)
    assert np.all(blend>=0)
    assert np.max(abs(np.diff(blend,axis=2)))<.05
    assert metadata['transition_sigma_px']==16.


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_whole_frame_baseline_matches_refit(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    ref=100+30*gaussian_filter(np.random.default_rng(94).normal(size=(96,112)),1.)
    frame,_,_=make_frame(ref,.8,-.3,0.)
    baseline=RefitRegistration(ref,CircularMultiscaleRegistration(ref),device=device)
    engine=RegionalRegistration(ref,CircularMultiscaleRegistration(ref),device=device)
    a=baseline.variants(frame,(.1,-.1),lambda:None)
    b=engine.variants(frame,(.1,-.1),lambda:None)
    for name in ('coherent','full_any'):
        torch.testing.assert_close(a[name],b[name],rtol=0,atol=1e-12)
    assert baseline.stats[-1]['accepted_halves']==engine.stats[-1]['accepted_halves']


def test_blank_reference_keeps_global_translation():
    ref=np.zeros((40,48))
    engine=RegionalRegistration(ref,CircularMultiscaleRegistration(ref),device='cpu')
    fields=engine.variants(ref,(.3,-.2),lambda:None)
    for field in fields.values():
        np.testing.assert_array_equal(field.numpy()[0],.3)
        np.testing.assert_array_equal(field.numpy()[1],-.2)
