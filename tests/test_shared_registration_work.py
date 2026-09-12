"""Sharing geometry and FFT work preserves independent channel operators."""
import numpy as np
import pytest
from scipy.signal import correlate
from planetrecon.geometry.warp import bilinear_sample,bilinear_push
from planetrecon.pipeline.masked_align import observed_correlations


@pytest.mark.parametrize('channels',[2,3])
@pytest.mark.parametrize('fill',[0.,3.5])
def test_joint_sample_matches_separate_planes(channels,fill):
    rng=np.random.default_rng(52)
    image=rng.normal(size=(17,23,channels))
    y=rng.uniform(-2,18,(25,19));x=rng.uniform(-2,25,y.shape)
    y[0,0]=np.nan
    expected=np.stack([bilinear_sample(image[...,i],y,x,fill=fill) for i in range(channels)],axis=-1)
    np.testing.assert_array_equal(bilinear_sample(image,y,x,fill=fill),expected)


@pytest.mark.parametrize('channels',[2,3])
def test_joint_scatter_matches_separate_planes(channels):
    rng=np.random.default_rng(21)
    image=rng.normal(size=(17,23,channels))
    image[4,8,0]=np.nan
    y=rng.uniform(-2,18,(17,23));x=rng.uniform(-2,25,y.shape)
    y[0,0]=np.nan
    valid=rng.uniform(size=y.shape)>.2
    expected=[bilinear_push(image[...,i],y,x,(19,24),valid) for i in range(channels)]
    actual=bilinear_push(image,y,x,(19,24),valid)
    for i in range(2):
        np.testing.assert_array_equal(actual[i],np.stack([pair[i] for pair in expected],axis=-1))


@pytest.mark.parametrize('shape',[(31,47),(32,48),(1,15)])
def test_shared_fft_matches_independent_linear_correlations(shape):
    rng=np.random.default_rng(9)
    proxy=rng.normal(size=shape)
    template=rng.normal(size=shape)
    mask=(rng.uniform(size=shape)>.2).astype(float)
    expected=[correlate(a,b,mode='full',method='fft') for a,b in
              [(proxy,template),(proxy,mask),(proxy**2,mask),(np.ones(shape),mask)]]
    for actual,reference in zip(observed_correlations(proxy,template,mask),expected):
        np.testing.assert_allclose(actual,reference,atol=2e-12,rtol=1e-12)
