import numpy as np
import pytest
from scipy.ndimage import gaussian_filter, shift

from planetrecon.pipeline.align import phase_correlation_shift
from planetrecon.detector import extract_green_proxy, cfa_labels


@pytest.mark.parametrize('offset', [(2.35,-3.7),(-7.25,5.4),(.3,.45)])
def test_noisy_bayer_registration_ignores_detector_lattice(offset):
    rng=np.random.default_rng(832)
    y,x=np.indices((96,112))
    disc=np.exp(-((x-54)**2+(y-48)**2)/250)
    texture=gaussian_filter(rng.normal(size=disc.shape),1.5)
    scene=100*disc*(1+.15*texture)
    moved=shift(scene,offset,order=3,mode='constant')
    labels=cfa_labels(*scene.shape,'RGGB')
    scale=np.where(labels=='R',.7,np.where(labels=='B',.5,1.))
    # Independent noise, with fixed detector-pattern contamination.
    fixed=3*((x+y)%2)
    ref=extract_green_proxy(scene*scale+fixed+rng.normal(0,3,scene.shape),'RGGB')
    img=extract_green_proxy(moved*scale+fixed+rng.normal(0,3,scene.shape),'RGGB')
    actual=phase_correlation_shift(ref,img)
    np.testing.assert_allclose(actual,offset[::-1],atol=.18)


def test_identical_and_flat_frames_do_not_invent_fractional_motion():
    rng=np.random.default_rng(6)
    a=rng.normal(size=(31,37))
    assert phase_correlation_shift(a,a)==(0.,0.)
    assert phase_correlation_shift(np.ones_like(a),np.ones_like(a))==(0.,0.)


@pytest.mark.hardware
def test_subpixel_cpu_cuda_agree(cuda_backend=None):
    import os
    if os.environ.get('PLANETRECON_TEST_GPU')!='1':pytest.skip('explicit GPU opt-in required')
    from planetrecon.backends.torch_accel import TorchBackend
    rng=np.random.default_rng(48)
    image=gaussian_filter(rng.normal(size=(80,96)),2)
    moved=shift(image,(1.3,-2.6),order=3,mode='wrap')
    np.testing.assert_allclose(TorchBackend().phase_correlation(image,moved),
                               phase_correlation_shift(image,moved),atol=1e-10)
